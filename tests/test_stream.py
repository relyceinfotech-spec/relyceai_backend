import sys, asyncio, traceback
from app.llm.processor import llm_processor

async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "who is the ceo of relyce infotech?"
    mode = sys.argv[2] if len(sys.argv) > 2 else "agent"
    print(f"Testing query: '{query}' in mode: '{mode}'")
    try:
        async for token in llm_processor.process_message_stream(query, mode=mode, context_messages=[], personality=None, user_settings={}, user_id="testuser", session_id="testsession"):
            if token.startswith("[INFO]"):
                print(f"\n{token}\n", end="")
            else:
                print(token, end="")
    except Exception as e:
        traceback.print_exc()

asyncio.run(main())
