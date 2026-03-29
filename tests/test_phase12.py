import asyncio
from app.llm.processor import LLMProcessor

async def test_semantic_headings():
    processor = LLMProcessor()
    
    query = "Compare CPU and GPU architectures"
    print(f"--- TESTING PHASE 12: {query} ---")
    
    response = await processor.process_agent_query(
        user_query=query,
        mode="normal",
        sub_intent="general"
    )
    
    print("\nFINAL RESPONSE OUTPUT:\n")
    print(response)
    
    with open("response_phase12.md", "w") as f:
        f.write(response)
    print("\nSaved to response_phase12.md")

if __name__ == "__main__":
    asyncio.run(test_semantic_headings())
