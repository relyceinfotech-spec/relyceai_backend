"""
Test Autonomous Task Loop
Verifies the TaskManager's persistent loop, goal checking, and synthesis.
"""
import asyncio
import os
import sys
from typing import AsyncGenerator

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.task_manager import TaskManager
from app.llm.router import get_openrouter_client

async def run_loop():
    client = get_openrouter_client()
    # Use a more reliable model for autonomous testing
    model = "gpt-4o-mini"
    
    manager = TaskManager(client, model)
    
    goal = "Research the current stock price of NVIDIA and compare it with its price 1 month ago. Summarize the trend."
    
    print(f"Goal: {goal}")
    print("-" * 40)
    
    async for chunk in manager.run_goal(goal):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(run_loop())
