import asyncio
from app.llm.processor import LLMProcessor

async def main():
    processor = LLMProcessor()
    print("Testing processor initialization and simple query...")
    try:
        res = await processor.process_agent_query(
            user_query="Hi",
            mode="normal",
            sub_intent="general"
        )
        print("RESULT TYPE:", type(res))
        print("RESULT CONTENT:", res[:100] if isinstance(res, str) else res)
        
        print("\nTesting structured query...")
        res2 = await processor.process_agent_query(
            user_query="Compare CPU and GPU",
            mode="normal",
            sub_intent="general"
        )
        print("\nSTRUCTURED RESULT:\n")
        print(res2)
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
