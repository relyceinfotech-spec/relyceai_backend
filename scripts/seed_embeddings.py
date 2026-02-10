"""
Seed script for embedding-based intent classification.
Run once to generate and store embeddings in Firestore.

Usage:
    cd d:\finalai\relyceai\backend
    python scripts/seed_embeddings.py
"""
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.llm.embeddings import save_intent_embeddings, INTENT_EXAMPLES


async def main():
    print("=" * 50)
    print("Relyce AI - Intent Embeddings Seed Script")
    print("=" * 50)
    print()
    
    # Show what will be embedded
    total_examples = sum(len(examples) for examples in INTENT_EXAMPLES.values())
    print(f"Intents to embed: {len(INTENT_EXAMPLES)}")
    print(f"Total examples: {total_examples}")
    print()
    
    for intent, examples in INTENT_EXAMPLES.items():
        print(f"  {intent}: {len(examples)} examples")
    
    print()
    print("Starting embedding generation...")
    print()
    
    success = await save_intent_embeddings()
    
    if success:
        print()
        print("=" * 50)
        print("✅ SUCCESS: All embeddings saved to Firestore")
        print("=" * 50)
    else:
        print()
        print("=" * 50)
        print("❌ FAILED: Check logs above for errors")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
