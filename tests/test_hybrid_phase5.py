import pytest
import asyncio
from typing import Dict, Any, List

from app.agent.graph_builder import compile_plan_graph
from app.state.plan_graph import PlanGraph, PlanNode, NodeStatus
from app.agent.graph_scheduler import run_plan_graph
from app.state.task_state_engine import get_task_state, clear_session_tasks
from app.agent.hybrid_controller import HybridAdvice

# Monkey patch execute_tool to avoid live execution in test
import app.agent.graph_scheduler
from app.agent.tool_executor import ToolResult

async def mock_execute_tool(tool_call, exec_ctx):
    return ToolResult(
        success=True,
        error=None,
        data=f"Mocked output for {tool_call.name}"
    )

app.agent.graph_scheduler.execute_tool = mock_execute_tool

class MockClient:
    async def __aenter__(self): return self
    async def __aexit__(self, exc_type, exc_val, exc_tb): pass
    
    class Chat:
        def __init__(self):
            class Completions:
                async def create(self, **kwargs):
                    all_prompts = [str(m.get("content", "")) for m in kwargs.get("messages", [])]
                    system_prompts = [m["content"] for m in kwargs.get("messages", []) if m["role"] == "system"]
                    user_prompts = [m["content"] for m in kwargs.get("messages", []) if m["role"] == "user"]

                    target_action = ""
                    for p in [*system_prompts, *user_prompts]:
                        if "Objective:" in p:
                            target_action = p.split("Objective:", 1)[1].strip()
                            break
                    if not target_action:
                        target_action = " ".join(all_prompts)
                    print(f"[TEST MOCK] target_action={target_action}")
                    
                    target_lower = target_action.lower()

                    if "deploy" in target_lower:
                        # Trigger non-transactional rollback
                        output = 'TOOL_CALL: SendWebhook("https://deploy.com", {})\n'
                    elif "search" in target_lower:
                        output = 'TOOL_CALL: search_web("python DAG")\n'
                    elif "write" in target_lower:
                        output = 'TOOL_CALL: WriteFile("test.py", "print(1)")\n'
                    else:
                        output = "I have completed the task."
                    print(f"[TEST MOCK] returning output={output.strip()}")
                        
                    class MockChunk:
                        class Choice:
                            class Delta:
                                content = output
                            delta = Delta()
                        choices = [Choice()]
                    
                    async def stream_gen():
                        yield MockChunk()
                        
                    return stream_gen()
            self.completions = Completions()
            
    @property
    def chat(self):
        return self.Chat()
        
class MockExecutionContext:
    def __init__(self):
        self.terminate = False
        self.forced_finalize = False
        self.degraded = False
        self.retry_count = 0
        self.tool_results = []

class MockAgentResult:
    def __init__(self):
        self.tool_allowed = True
        self.allowed_tools = ["search_web", "WriteFile", "SendWebhook"]
        self.execution_context = MockExecutionContext()


async def _run_graph_and_collect(**kwargs) -> str:
    chunks: List[str] = []
    async for chunk in run_plan_graph(**kwargs):
        chunks.append(str(chunk))
    return "".join(chunks)

def test_compile_linear_graph_execution():
    session_id = "test_phase5_session"
    task_id = "task_p5_1"
    clear_session_tasks(session_id)
    
    # 1. Compile DAG manually to avoid live LLM calls in unit test
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    graph.add_node(PlanNode("P1", "TOOL_CALL", {"instruction": "search python"}))
    graph.add_node(PlanNode("P2", "TOOL_CALL", {"instruction": "write a summary"}, ["P1"]))
    graph.add_node(PlanNode("FINAL", "REASONING", {"instruction": "Finalize"}, ["P2"]))
    
    assert len(graph.nodes) > 1
    
    client = MockClient()
    agent_result = MockAgentResult()
    strategy = HybridAdvice(
        planning_mode="ADAPTIVE_CODE_PLAN",
        reasoning_context={},
        repair_policy={"enabled": False, "max_attempts": 0}
    )
    
    messages = []
    # Run the DAG engine
    response = asyncio.run(_run_graph_and_collect(
        graph=graph,
        strategy=strategy,
        user_query="search python and then write a summary",
        messages=messages,
        agent_result=agent_result,
        client=client,
        model_to_use="mock",
        create_kwargs={"messages": messages}
    ))
    
    assert graph.is_fully_completed()
    assert graph.nodes["P1"].status == NodeStatus.COMPLETED
    assert graph.nodes["FINAL"].status == NodeStatus.COMPLETED

def test_non_transactional_hard_block_in_dag():
    session_id = "test_phase5_session"
    task_id = "task_p5_2"
    clear_session_tasks(session_id)
    
    from app.state.plan_graph import PlanGraph, PlanNode
    
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    graph.add_node(PlanNode("N1", "TOOL_CALL", {"instruction": "deploy the code"}))
    
    client = MockClient()
    agent_result = MockAgentResult()
    strategy = HybridAdvice(
        planning_mode="SEQUENTIAL_PLAN",
        reasoning_context={},
        repair_policy={"enabled": False}
    )
    
    messages = []
    response = asyncio.run(_run_graph_and_collect(
        graph=graph,
        strategy=strategy,
        user_query="deploy the code",
        messages=messages,
        agent_result=agent_result,
        client=client,
        model_to_use="mock",
        create_kwargs={"messages": messages}
    ))
    
    assert agent_result.execution_context.degraded is True
    assert graph.nodes["N1"].status == NodeStatus.FAILED
    assert not graph.is_fully_completed()
