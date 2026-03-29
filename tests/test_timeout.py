import asyncio
import time
from unittest.mock import patch

from app.llm.processor import LLMProcessor
from app.context.query_planner import plan_query

async def mock_retrieve_graph(uid, q, limit=5):
    print("Graph retrieval hanging...")
    await asyncio.sleep(2.0)
    return ["mock_graph_triple"]

async def mock_retrieve_memories(uid, q, top_k, final_limit):
    print("Vector retrieval fast...")
    await asyncio.sleep(0.1)
    class MockMemory:
        def __init__(self):
            self.text = "mock_vector_memory"
    return [MockMemory()]

async def mock_fetch_and_extract(url):
    print("Web fetch hanging...")
    await asyncio.sleep(6.0)
    return {"status": "success", "data": {"content": "web content", "title": "test"}}

async def test_timeouts():
    processor = LLMProcessor()
    print("Testing retrieval timeouts...")
    
    with patch("app.context.query_planner.plan_query") as mock_planner, \
         patch("app.memory.knowledge_graph.retrieve_graph", side_effect=mock_retrieve_graph), \
         patch("app.memory.vector_memory.retrieve_memories", side_effect=mock_retrieve_memories), \
         patch("app.input_processing.web_fetch.fetch_and_extract", side_effect=mock_fetch_and_extract), \
         patch("app.llm.processor.LLMProcessor._call_model", return_value=(["mock response"], None)):
        
        # Force the planner to ask for both memory and web
        mock_planner.return_value = {
            "task_type": "general_chat",
            "needs_web": True,
            "needs_memory": True,
            "needs_chunking": False,
            "reasoning_depth": "low"
        }
        
        start = time.time()
        
        try:
            # We only want to test the context gathering part, but we can't easily break out early.
            # We'll run the full inference with mocked model.
            async for chunk in processor.stream_inference(
                mode="chat",
                user_id="test_user",
                user_query="https://example.com timeout test",
                message_history=[],
                personality={"id": "coding_buddy"},
                sub_intent="general_chat",
                user_profile=None
            ):
                pass
        except Exception as e:
            print(f"Failed with exception: {e}")
            raise e

        # The retrieval should wait max 600ms, and the web fetch max 5.0s
        duration = time.time() - start
        print(f"Total processing time: {duration:.2f}s")
        assert duration < 6.5, f"Pipeline stalled! Duration {duration:.2f}s > expected max 6.5s"
        print("\nTimeout test passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_timeouts())
