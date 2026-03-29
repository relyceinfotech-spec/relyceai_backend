import asyncio
import time
import sys
import os
sys.path.append(os.path.abspath(r'd:\finalai\testing\backend'))

from app.llm.processor import llm_processor
from app.agent.agent_orchestrator import run_agent_pipeline

async def run_queries():
    queries = [
        "Give me exactly 2 reasons to exercise",
        "What is today's date?",
        "Compare the best laptops",
        "Search for React documentation"
    ]
    
    user_settings = {"is_sandbox_test": True}
    results = []
    
    for q in queries:
        start_time = time.time()
        output = ""
        intel = ""
        try:
            async for token in llm_processor._process_agent_mode(
                user_query=q,
                context_messages=[],
                personality=None,
                user_settings=user_settings,
                user_id="test_user",
                session_id="test_session",
                start_time=start_time
            ):
                if token and token.startswith("[INFO]INTEL:"):
                    intel = token[len("[INFO]INTEL:"):]
                elif token:
                    output += token
        except Exception as e:
            output = f"[Error]: {e}"
            
        results.append((q, intel, output))

    with open("agent_results.txt", "w", encoding="utf-8") as f:
        f.write("AGENT EXECUTION TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        for q, intel, out in results:
            f.write(f"QUERY: {q}\n")
            f.write(f"INTEL: {intel}\n")
            f.write(f"RESPONSE:\n{out.strip()}\n")
            f.write("-" * 60 + "\n\n")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_queries())
