import asyncio
from app.memory.vector_memory import store_memory, retrieve_memories, bump_memory_scores
from app.memory.knowledge_graph import store_triples, retrieve_graph, bump_graph_scores, Triple

async def test_memory_hierarchy():
    user_id = "test_user_memory_123"
    
    print("\n--- 1. Testing Vector Memory Insertion ---")
    await store_memory(user_id, "I love using FastAPI for AI projects", "general", importance=0.8)
    await store_memory(user_id, "My backend uses litestar in production", "general", importance=0.9)
    await store_memory(user_id, "Deployments happen on AWS", "general", importance=0.6)
    
    print("\n--- 2. Testing Vector Memory Retrieval ---")
    vectors = await retrieve_memories(user_id, "What framework do I use?")
    print(f"Retrieved {len(vectors)} vectors.")
    vector_ids = [v.uuid for v in vectors if hasattr(v, 'uuid') and v.uuid]
    for i, v in enumerate(vectors):
        print(f" [{i+1}] {v.text} (Score: {v.final_score:.2f})")
    
    print("\n--- 3. Testing Vector Quality Scoring (Bump) ---")
    if vector_ids:
        await bump_memory_scores(user_id, vector_ids, positive=True)
        print(f"Bumped positive scores for {len(vector_ids)} vectors")
        
    print("\n--- 4. Testing Knowledge Graph Insertion ---")
    triples = [
        Triple("User", "building", "AI startup", importance=0.85),
        Triple("Backend", "uses", "Python", importance=0.75)
    ]
    count = await store_triples(user_id, triples)
    print(f"Stored {count} triples.")
    
    print("\n--- 5. Testing Knowledge Graph Retrieval ---")
    graph = await retrieve_graph(user_id, "AI startup developer python backend")
    print(f"Retrieved {len(graph) if graph else 0} graph triples.")
    
    print("\n✅ Memory Hierarchy & Scoring Test Complete")

if __name__ == "__main__":
    asyncio.run(test_memory_hierarchy())
