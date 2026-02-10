"""Test optimizations: fast path and LRU cache."""
import asyncio
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.llm.embeddings import classify_intent, load_intent_embeddings


async def test():
    print("=" * 60)
    print("Testing Optimizations: Fast Path + LRU Cache")
    print("=" * 60)
    print()
    
    # Load embeddings first
    await load_intent_embeddings()
    
    # 1. Test FAST PATH queries
    print("1. FAST PATH TESTS (should be instant):")
    print("-" * 40)
    fast_queries = [
        ("hi", "casual"),
        ("hello", "casual"),
        ("hey macha", "casual"),
        ("latest news", "web_factual"),
        ("price of bitcoin today", "web_factual"),
    ]
    
    for query, expected in fast_queries:
        start = time.time()
        result = await classify_intent(query)
        elapsed = (time.time() - start) * 1000
        match = "✅" if result["intent"] == expected else "❌"
        print(f"{match} \"{query}\" -> {result['intent']} ({elapsed:.1f}ms) path={result['path']}")
    
    print()
    
    # 2. Test LRU CACHE (second call should be instant)
    print("2. LRU CACHE TEST (repeat query):")
    print("-" * 40)
    
    # First call (cold)
    query = "explain this python function"
    start = time.time()
    result1 = await classify_intent(query)
    time1 = (time.time() - start) * 1000
    print(f"First call: {time1:.0f}ms (path={result1['path']})")
    
    # Second call (should be cached)
    start = time.time()
    result2 = await classify_intent(query)
    time2 = (time.time() - start) * 1000
    print(f"Second call: {time2:.1f}ms (path={result2['path']})")
    print(f"Speedup: {time1/time2:.0f}x faster!")
    
    print()
    print("=" * 60)
    print("Optimization test complete!")


if __name__ == "__main__":
    asyncio.run(test())
