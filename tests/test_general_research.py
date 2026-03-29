import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the backend directory to sys.path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.agent.task_manager import TaskManager
from app.llm.router import get_openrouter_client

load_dotenv(dotenv_path="backend/.env")

async def test_general_research():
    print("=== Testing Autonomous Research Loop ===")
    
    # We use gpt-4o-mini for speed and reliability in tests
    client = get_openrouter_client()
    # A research task that requires multiple steps (search + synthesis)
    goal = "Who won the 2024 F1 Drivers Championship and what was the points gap to second place?"
    
    # Create an AgentState to capture traces and workspace
    from app.agent.agent_state import AgentState
    state = AgentState(query=goal, goal=goal)
    
    task_manager = TaskManager(client, model_to_use="gpt-4o-mini")
    
    print(f"Goal: {goal}")
    print("-" * 50)
    
    async for status in task_manager.run_goal(goal, initial_state=state):
        print(status, end="", flush=True)

    print("\n" + "="*50)
    print("Test Complete")
    
    print(f"\n--- PRODUCTION HARDENING VERIFICATION ---")
    print(f"Trace ID: {state.trace_id}")
    print(f"Total Traces: {len(state.traces)}")
    print(f"Total Findings in Workspace: {len(state.workspace.knowledge)}")
    print(f"Total Sources tracked: {len(state.workspace.sources)}")
    
    if state.traces:
        print("\nLast 3 Trace Events:")
        for t in state.traces[-3:]:
             print(f"  - {t.action} | {t.latency_ms:.2f}ms | {t.result_status}")

    if state.workspace.sources:
        print("\nPrimary Source:")
        s = state.workspace.sources[0]
        print(f"  - URL: {s['url']}")
        print(f"  - Trust: {s['trust_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_general_research())
