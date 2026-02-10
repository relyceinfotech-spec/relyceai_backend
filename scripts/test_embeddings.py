"""Test script for embedding classification."""
import asyncio
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.llm.embeddings import classify_intent, load_intent_embeddings


async def test():
    print("Loading embeddings from Firestore...")
    await load_intent_embeddings()
    
    test_cases = [
        ("hi macha", "casual"),
        ("latest news today", "web_factual"),
        ("explain this code", "coding_simple"),
        ("debug this bug", "coding_complex"),
        ("business strategy advice", "business"),
        ("why is this happening", "analysis_internal"),
        ("analyze recent market trends", "web_analysis")
    ]
    
    print()
    print("Testing intent classification:")
    print("=" * 60)
    
    passed = 0
    for query, expected in test_cases:
        result = await classify_intent(query)
        match = "PASS" if result["intent"] == expected else "FAIL"
        if match == "PASS":
            passed += 1
        print(f"[{match}] \"{query}\" -> {result['intent']} (conf={result['confidence']:.2f})")
    
    print()
    print(f"Results: {passed}/{len(test_cases)} passed")
    print()
    
    # Benchmark
    print("Running benchmark (4 queries)...")
    start = time.time()
    for q, _ in test_cases[:4]:
        await classify_intent(q)
    elapsed = (time.time() - start) / 4 * 1000
    print(f"Average classification time: {elapsed:.0f}ms")


if __name__ == "__main__":
    asyncio.run(test())
