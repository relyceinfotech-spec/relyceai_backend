import asyncio
from types import SimpleNamespace

from app.agent.graph_scheduler import run_plan_graph
from app.agent.tool_executor import parse_tool_calls
from app.agent.tools import local_ops
from app.state.plan_graph import NodeStatus, PlanGraph, PlanNode


class _ExecCtx:
    def __init__(self):
        self.terminate = False
        self.forced_finalize = False
        self.degraded = False
        self.retry_count = 0
        self.tool_results = []


class _AgentResult:
    def __init__(self):
        self.tool_allowed = True
        self.allowed_tools = ["search_web"]
        self.execution_context = _ExecCtx()


class _ClientNeverUsed:
    class _Chat:
        class _Completions:
            async def create(self, **kwargs):
                raise AssertionError("Client should not be called for blocked planned tools")

        completions = _Completions()

    chat = _Chat()


async def _collect_chunks(gen):
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks


def test_run_plan_graph_blocks_disallowed_planned_tool_node():
    graph = PlanGraph(graph_id="g_tool_block", session_id="s_tool_block")
    graph.add_node(
        PlanNode(
            node_id="N1",
            action_type="TOOL_CALL",
            payload={"tool": "search_patents", "instruction": "latest ai patents"},
        )
    )

    messages = []
    strategy = SimpleNamespace(repair_policy={"enabled": False})
    agent_result = _AgentResult()

    chunks = asyncio.run(
        _collect_chunks(
            run_plan_graph(
                graph=graph,
                strategy=strategy,
                user_query="latest ai patents",
                messages=messages,
                agent_result=agent_result,
                client=_ClientNeverUsed(),
                model_to_use="mock",
                create_kwargs={"messages": messages},
            )
        )
    )

    assert graph.nodes["N1"].status == NodeStatus.FAILED
    assert len(agent_result.execution_context.tool_results) == 0
    assert any('"event": "tool_blocked"' in chunk for chunk in chunks)


def test_parse_tool_calls_keeps_order_for_mixed_quoted_and_unquoted_calls():
    text = (
        'TOOL_CALL: search_web("nvidia stock today")\n'
        "TOOL_CALL: calculate(12*5)\n"
        'TOOL_CALL: summarize_url("https://example.com?q=\\"ai\\"")'
    )
    calls = parse_tool_calls(text)

    assert [c.name for c in calls] == ["search_web", "calculate", "summarize_url"]
    assert calls[0].args == "nvidia stock today"
    assert calls[1].args == "12*5"
    assert calls[2].args == 'https://example.com?q="ai"'


def test_local_ops_wrappers_support_sync_and_async_tool_impls(monkeypatch):
    def _sync_calc(args: str = ""):
        return {"status": "success", "data": {"result": 2}, "source": "calc", "confidence": "high"}

    async def _async_read(args: str = ""):
        return {"status": "success", "data": "ok", "source": "fs", "confidence": "high"}

    monkeypatch.setattr(local_ops, "_real_calculate", _sync_calc)
    monkeypatch.setattr(local_ops, "_real_read_file", _async_read)

    calc_out = asyncio.run(local_ops._tool_calculate("1+1", "session-x"))
    read_out = asyncio.run(local_ops._tool_read_file("notes.txt", "session-x"))

    assert calc_out["status"] == "success"
    assert calc_out["data"]["result"] == 2
    assert read_out["status"] == "success"
    assert read_out["data"] == "ok"
