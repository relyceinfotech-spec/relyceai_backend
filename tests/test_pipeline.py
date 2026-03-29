import asyncio
import time
import sys

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for a in args:
            if isinstance(a, str):
                safe_args.append(a.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
            else:
                safe_args.append(a)
        print(*safe_args, **kwargs)

from app.llm.processor import llm_processor
from app.chat.context import get_context_for_llm
from app.memory.vector_memory import retrieve_memories

async def test_backend():
    user_id = "test_user_777"
    session_id = "test_session_1"
    
    print("\n" + "="*50)
    print("🚀 RUNNING FULL PIPELINE TEST")
    print("="*50)

    # 1. First question - establish context
    q1 = "I am a senior python developer building a new AI startup."
    print(f"\n[User]: {q1}")
    
    t0 = time.time()
    resp1 = ""
    async for chunk in llm_processor.process_message_stream(
        q1, 
        mode="normal", 
        context_messages=[], 
        personality=None, 
        user_settings={}, 
        user_id=user_id, 
        session_id=session_id
    ):
        if not chunk.startswith("[INFO]"):
            resp1 += chunk
            # print(chunk, end="", flush=True)
            
    t1 = time.time()
    safe_print(f"\n[AI]: {resp1[:150]}...")
    safe_print(f"⏱️  Time: {t1-t0:.2f}s | Tokens: ~{len(resp1)//4}")

    # Process background tasks manually for the test
    # from app.chat.context import update_context_with_exchange
    # update_context_with_exchange(user_id, session_id, q1, resp1, None)
    
    from app.memory.vector_memory import extract_and_store_memories
    from app.memory.knowledge_graph import extract_and_store_graph
    
    safe_print("\n⏳ Extracting memories...")
    await extract_and_store_memories(user_id, q1, resp1)
    await extract_and_store_graph(user_id, q1, resp1)
    
    # 2. Second question requiring reasoning + web + memory
    # Using complex query to trigger Semantic Router -> needs_web, needs_memory, needs_chunking
    q2 = "Based on what you know about me, should I use FastAPI or a new framework like Litestar for my backend? Please search the web for litestar vs fastapi benchmarks to support your answer."
    safe_print(f"\n[User]: {q2}")
    
    context = await asyncio.to_thread(get_context_for_llm, user_id, session_id, None)
    
    t2 = time.time()
    resp2 = ""
    async for chunk in llm_processor.process_message_stream(
        q2, 
        mode="normal", 
        context_messages=context, 
        personality=None, 
        user_settings={}, 
        user_id=user_id, 
        session_id=session_id
    ):
        if chunk.startswith("[INFO]"):
            print(f"  {chunk.strip()}")
        else:
            resp2 += chunk
            
    t3 = time.time()
    safe_print(f"\n[AI]:\n{resp2}")
    safe_print(f"\n⏱️  Time: {t3-t2:.2f}s | Tokens: ~{len(resp2)//4}")
    
    # update_context_with_exchange(user_id, session_id, q2, resp2, None)
    await extract_and_store_memories(user_id, q2, resp2)
    await extract_and_store_graph(user_id, q2, resp2)

    safe_print("\n" + "="*50)
    safe_print("🧠 CHECKING EXTRACTED MEMORY")
    safe_print("="*50)
    
    # Check stored vector memory
    memories = await retrieve_memories(user_id, "background", top_k=5, final_limit=5)
    safe_print(f"Vector Memories Found: {len(memories)}")
    for i, m in enumerate(memories):
        safe_print(f"  [{i+1}] {m.text} | Score: {m.final_score:.2f} | Imp: {m.importance:.2f} | Count: {m.retrieval_count}")
        
    # Check stored graph
    from app.memory.knowledge_graph import retrieve_graph
    graph = await retrieve_graph(user_id, "developer", limit=5)
    safe_print(f"\nGraph Triples Found: {len(graph) if graph else 0}")
    if graph:
        for i, t in enumerate(graph):
            safe_print(f"  [{i+1}] {t.subject} - {t.relation} - {t.object} | Imp: {t.importance:.2f} | Last Seen: {t.last_seen[:10]}")

if __name__ == "__main__":
    asyncio.run(test_backend())

