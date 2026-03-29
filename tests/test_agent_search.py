import asyncio
import os
from app.llm.processor import LLMProcessor

async def run_agent():
    processor = LLMProcessor()
    async for chunk in processor.process_message_stream(
        user_query="recent attack iran vs israel",
        context_messages=[],
        mode="agent",
        user_id="test-user",
        session_id="test-session"
    ):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(run_agent())
