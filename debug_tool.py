import asyncio
import sys
import os
import time
sys.path.append(os.path.abspath(r'd:\finalai\testing\backend'))

from app.llm.processor import llm_processor
import app.llm.processor as proc_module

original_build = proc_module.build_agent_system_prompt

def hooked_build(*args, **kwargs):
    prompt = original_build(*args, **kwargs)
    with open("debug_output.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "="*70 + "\n")
        f.write("🧠 SYSTEM PROMPT (What the LLM sees):\n")
        f.write("="*70 + "\n")
        f.write(prompt + "\n")
        f.write("="*70 + "\n\n")
    return prompt

proc_module.build_agent_system_prompt = hooked_build

async def run_debug():
    # Clear file
    with open("debug_output.txt", "w", encoding="utf-8") as f:
        f.write("STARTING DEBUG...\n")

    query = "Search the web for the latest Next.js 15 features and calculate 145 * 12"
    with open("debug_output.txt", "a", encoding="utf-8") as f:
        f.write(f"👤 USER QUERY: {query}\n\n")
        f.write("🌊 STREAM OUTPUT:\n")
        f.write("-" * 70 + "\n")
    
    start = time.time()
    
    try:
        async for token in llm_processor._process_agent_mode(
            user_query=query,
            context_messages=[],
            start_time=start
        ):
            with open("debug_output.txt", "a", encoding="utf-8") as f:
                if token.startswith("[INFO]"):
                    f.write(f"\n\n🟢 [UI EVENT METADATA]: {token}\n\n")
                else:
                    f.write(token)
    except Exception as e:
        with open("debug_output.txt", "a", encoding="utf-8") as f:
            f.write(f"\n❌ Error during stream: {e}\n")
            
    with open("debug_output.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"⏱️ DONE. Total Time: {time.time() - start:.2f}s\n")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_debug())
