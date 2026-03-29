"""
Vector Memory Store — Weaviate-backed semantic memory with:
- Dedup (cosine > 0.85 → update)
- Ranked retrieval (similarity×0.7 + importance×0.2 + decay×0.1)
- Exponential aging: exp(-days/30)
- Memory type filtering
- Firestore keyword fallback if Weaviate unreachable
- Retrieval reinforcement (importance bump on use)
- Debug logging for retrieval
- Timeout safety (300ms max for vector search)
"""
import asyncio
import math
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from app.config import (
    MAX_MEMORIES_PER_USER,
    MEMORY_DEDUP_THRESHOLD,
)

# Global flag set on startup
VECTOR_MEMORY_ENABLED = False


@dataclass
class MemoryResult:
    """A retrieved memory with its ranking score."""
    text: str
    memory_type: str
    importance: float
    similarity: float
    final_score: float
    created_at: str = ""
    last_seen: str = ""
    retrieval_count: int = 0
    uuid: str = ""


def set_vector_enabled(enabled: bool):
    """Called from startup to set availability flag."""
    global VECTOR_MEMORY_ENABLED
    VECTOR_MEMORY_ENABLED = enabled


# ============================================
# STORE (Firestore primary + Weaviate acceleration)
# ============================================

async def store_memory(
    user_id: str,
    text: str,
    memory_type: str = "temporary_context",
    importance: float = 0.5,
    expires_at: Optional[str] = None,
) -> bool:
    """
    Store a memory. Firestore = canonical, Weaviate = acceleration index.
    Dedup: cosine > 0.85 → update existing instead of insert.
    """
    # Always store in Firestore first (canonical)
    firestore_ok = await _firestore_store(user_id, text, memory_type, importance, expires_at)
    
    if not VECTOR_MEMORY_ENABLED:
        return firestore_ok
    
    # Try Weaviate (acceleration, non-blocking failure)
    try:
        from app.memory.weaviate_client import get_weaviate_client, generate_embedding, COLLECTION_NAME
        
        client = await get_weaviate_client()
        if not client:
            return firestore_ok
        
        embedding = await generate_embedding(text)
        if not embedding:
            print(f"[VectorMemory] Embedding failed, Firestore-only for: {text[:50]}")
            return firestore_ok
        
        collection = client.collections.get(COLLECTION_NAME)
        now = datetime.utcnow().isoformat()
        
        # --- Deduplication ---
        existing = collection.query.near_vector(
            near_vector=embedding,
            limit=3,
            filters=_user_filter(user_id),
            return_metadata=["distance"],
        )
        
        for obj in existing.objects:
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            similarity = 1.0 - distance
            
            if similarity >= MEMORY_DEDUP_THRESHOLD:
                # Merge: keep newer text, bump importance
                collection.data.update(
                    uuid=obj.uuid,
                    properties={
                        "text": text,
                        "importance": min(1.0, max(importance, obj.properties.get("importance", 0)) + 0.05),
                        "last_seen": now,
                        "retrieval_count": obj.properties.get("retrieval_count", 0),
                    }
                )
                print(f"[VectorMemory] Dedup merge (sim={similarity:.3f}): {text[:50]}")
                return True
        
        # --- Capacity enforcement ---
        count_result = collection.aggregate.over_all(
            filters=_user_filter(user_id),
            total_count=True,
        )
        current_count = count_result.total_count or 0
        
        if current_count >= MAX_MEMORIES_PER_USER:
            evict_candidates = collection.query.fetch_objects(
                filters=_user_filter(user_id),
                sort=_importance_sort_asc(),
                limit=1,
            )
            if evict_candidates.objects:
                lowest = evict_candidates.objects[0]
                lowest_imp = lowest.properties.get("importance", 0)
                if lowest_imp < importance:
                    collection.data.delete_by_id(lowest.uuid)
                    print(f"[VectorMemory] Evicted (imp={lowest_imp:.2f}): {lowest.properties.get('text', '')[:40]}")
                else:
                    print(f"[VectorMemory] At capacity, new entry not important enough")
                    return firestore_ok
        
        # --- Insert ---
        collection.data.insert(
            properties={
                "text": text,
                "user_id": user_id,
                "memory_type": memory_type,
                "importance": importance,
                "created_at": now,
                "last_seen": now,
                "expires_at": expires_at or "",
                "retrieval_count": 0,
            },
            vector=embedding,
        )
        print(f"[VectorMemory] Stored [{memory_type}] (imp={importance:.2f}): {text[:50]}")
        return True
        
    except Exception as e:
        print(f"[VectorMemory] Weaviate store failed (Firestore OK): {e}")
        return firestore_ok


# ============================================
# RETRIEVE (Vector with Firestore fallback)
# ============================================

async def retrieve_memories(
    user_id: str,
    query: str,
    top_k: int = 10,
    final_limit: int = 3,
) -> List[MemoryResult]:
    """
    Retrieve semantically relevant memories.
    Vector search with 300ms timeout → fallback to Firestore keyword search.
    Ranking: similarity×0.7 + importance×0.2 + decay×0.1
    """
    if not user_id or user_id == "anonymous":
        return []
    
    if VECTOR_MEMORY_ENABLED:
        try:
            results = await asyncio.wait_for(
                _vector_retrieve(user_id, query, top_k, final_limit),
                timeout=0.3
            )
            if results:
                return results
        except asyncio.TimeoutError:
            print("[VectorMemory] Vector retrieval timed out (300ms), using Firestore fallback")
        except Exception as e:
            print(f"[VectorMemory] Vector retrieval failed, using fallback: {e}")
    
    # Firestore keyword fallback
    return await _firestore_fallback_retrieve(user_id, query, final_limit)


async def _vector_retrieve(
    user_id: str,
    query: str,
    top_k: int,
    final_limit: int,
) -> List[MemoryResult]:
    """Weaviate vector search with ranking."""
    from app.memory.weaviate_client import get_weaviate_client, generate_embedding, COLLECTION_NAME
    
    client = await get_weaviate_client()
    if not client:
        return []
    
    query_embedding = await generate_embedding(query)
    if not query_embedding:
        return []
    
    collection = client.collections.get(COLLECTION_NAME)
    
    results = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        filters=_user_filter(user_id),
        return_metadata=["distance"],
    )
    
    if not results.objects:
        return []
    
    now = datetime.utcnow()
    scored: List[MemoryResult] = []
    
    # Detect query category for memory type filtering
    query_lower = query.lower()
    is_debug_query = any(w in query_lower for w in ["debug", "error", "bug", "fix", "crash", "fail"])
    
    for obj in results.objects:
        props = obj.properties
        
        # Skip expired
        expires_at = props.get("expires_at", "")
        if expires_at:
            try:
                if datetime.fromisoformat(expires_at) < now:
                    asyncio.create_task(_delete_memory_safe(collection, obj.uuid))
                    continue
            except (ValueError, TypeError):
                pass
        
        # Memory type filtering: exclude temporary_context from non-debug queries
        mem_type = props.get("memory_type", "")
        if mem_type == "temporary_context" and not is_debug_query:
            continue
        
        # Similarity from distance
        distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
        similarity = max(0.0, 1.0 - distance)
        
        # Exponential decay: exp(-days_since_last_seen / 30)
        decay = 1.0
        last_seen = props.get("last_seen", "")
        if last_seen:
            try:
                days_ago = max(0, (now - datetime.fromisoformat(last_seen)).days)
                decay = math.exp(-days_ago / 30)
            except (ValueError, TypeError):
                pass
        
        importance = float(props.get("importance", 0.5))
        
        # Weekly importance decay: importance *= 0.98^weeks (Item #5)
        if last_seen:
            try:
                weeks_old = max(0, (now - datetime.fromisoformat(last_seen)).days) / 7
                importance = importance * (0.98 ** weeks_old)
            except (ValueError, TypeError):
                pass
                
        # Memory Hygiene: Skip memories below reinforcement threshold
        # Deleted in a background job to prevent concurrent read/write race conditions
        if importance < 0.15:
            continue
            
        importance = max(0.1, min(1.0, importance))
        
        # Weighted ranking: similarity dominates
        final_score = similarity * 0.7 + importance * 0.2 + decay * 0.1
        
        scored.append(MemoryResult(
            text=props.get("text", ""),
            memory_type=mem_type,
            importance=importance,
            similarity=similarity,
            final_score=final_score,
            created_at=props.get("created_at", ""),
            last_seen=last_seen,
            retrieval_count=int(props.get("retrieval_count", 0)),
            uuid=str(obj.uuid),
        ))
    
    scored.sort(key=lambda m: m.final_score, reverse=True)

    # --- Diversity filtering: skip near-duplicates ---
    diverse: List[MemoryResult] = []
    for candidate in scored:
        is_dup = False
        for existing in diverse:
            overlap = _text_overlap(candidate.text, existing.text)
            if overlap > 0.90:
                is_dup = True
                break
        if not is_dup:
            diverse.append(candidate)
        if len(diverse) >= final_limit:
            break

    if len(scored) != len(diverse):
        print(f"[VectorMemory] Diversity filter: {len(scored)} → {len(diverse)} (removed {len(scored) - len(diverse)} near-dupes)")

    top = diverse[:final_limit]
    
    # Debug logging
    if top:
        print(f"[VectorMemory] Retrieval for: \"{query[:60]}\"")
        for m in top:
            print(f"  sim={m.similarity:.2f} imp={m.importance:.2f} score={m.final_score:.2f} → {m.text[:60]}")
    
    # Reinforcement: bump importance + retrieval_count for used memories
    for mem in top:
        asyncio.create_task(_reinforce_memory(collection, mem.uuid, mem.importance))
    
    return top


# ============================================
# FIRESTORE FALLBACK
# ============================================

async def _firestore_store(
    user_id: str,
    text: str,
    memory_type: str,
    importance: float,
    expires_at: Optional[str],
) -> bool:
    """Store memory in Firestore (canonical store)."""
    try:
        from app.auth import get_firestore_db
        import hashlib
        
        db = get_firestore_db()
        if not db:
            return False
        
        content_hash = hashlib.md5(text.lower().strip().encode()).hexdigest()[:12]
        now = datetime.utcnow().isoformat()
        
        ref = db.collection("users").document(user_id).collection("vector_memories")
        
        # Check for duplicate by hash
        existing = ref.where("content_hash", "==", content_hash).limit(1).stream()
        for doc in existing:
            # Update existing
            doc.reference.update({
                "text": text,
                "importance": min(1.0, importance + 0.05),
                "last_seen": now,
            })
            return True
        
        # Insert new
        ref.add({
            "text": text,
            "user_id": user_id,
            "memory_type": memory_type,
            "importance": importance,
            "created_at": now,
            "last_seen": now,
            "expires_at": expires_at or "",
            "content_hash": content_hash,
            "retrieval_count": 0,
        })
        return True
        
    except Exception as e:
        print(f"[VectorMemory] Firestore store failed: {e}")
        return False


async def _firestore_fallback_retrieve(
    user_id: str,
    query: str,
    limit: int = 3,
) -> List[MemoryResult]:
    """Keyword-based retrieval from Firestore when Weaviate is unavailable."""
    try:
        from app.auth import get_firestore_db
        
        db = get_firestore_db()
        if not db:
            return []
        
        ref = db.collection("users").document(user_id).collection("vector_memories")
        docs = ref.order_by("importance", direction="DESCENDING").limit(50).stream()
        
        # Extract keywords (words > 4 chars)
        keywords = [w.lower() for w in query.split() if len(w) > 4]
        
        now = datetime.utcnow()
        candidates: List[MemoryResult] = []
        
        for doc in docs:
            data = doc.to_dict()
            text = data.get("text", "").lower()
            
            # Skip expired
            expires_at = data.get("expires_at", "")
            if expires_at:
                try:
                    if datetime.fromisoformat(expires_at) < now:
                        continue
                except (ValueError, TypeError):
                    pass
            
            # Keyword match score
            if keywords:
                match_count = sum(1 for k in keywords if k in text)
                if match_count == 0:
                    continue
                keyword_score = min(1.0, match_count / max(len(keywords), 1))
            else:
                keyword_score = 0.3  # Include high-importance memories even without keyword match
            
            importance = float(data.get("importance", 0.5))
            
            candidates.append(MemoryResult(
                text=data.get("text", ""),
                memory_type=data.get("memory_type", ""),
                importance=importance,
                similarity=keyword_score,
                final_score=keyword_score * 0.7 + importance * 0.3,
                created_at=data.get("created_at", ""),
                last_seen=data.get("last_seen", ""),
                retrieval_count=int(data.get("retrieval_count", 0)),
            ))
        
        candidates.sort(key=lambda m: m.final_score, reverse=True)
        
        if candidates:
            print(f"[VectorMemory] Firestore fallback returned {len(candidates[:limit])} memories")
        
        return candidates[:limit]
        
    except Exception as e:
        print(f"[VectorMemory] Firestore fallback error: {e}")

    return []


# ============================================
# RETRIEVAL QUALITY SCORING
# ============================================

async def bump_memory_scores(user_id: str, uuids: List[str], positive: bool = True):
    """
    Adjust the importance score and retrieval count of memories.
    Positive (+0.05) if cited by LLM, negative (-0.02) if retrieved but ignored.
    This prevents memory pollution over time.
    """
    if not VECTOR_MEMORY_ENABLED or not uuids:
        return
        
    try:
        from app.memory.weaviate_client import get_weaviate_client, COLLECTION_NAME
        client = await get_weaviate_client()
        if not client:
            return
            
        collection = client.collections.get(COLLECTION_NAME)
        
        for uuid_str in uuids:
            if not uuid_str or uuid_str.startswith("memory_"):
                continue  # Skip un-ID'd memories
                
            obj = collection.query.fetch_object_by_id(uuid_str)
            if not obj:
                continue
                
            current_imp = obj.properties.get("importance", 0.5)
            retrieval_count = obj.properties.get("retrieval_count", 0) + 1
            
            if positive:
                new_imp = min(1.0, current_imp + 0.05)
            else:
                new_imp = max(0.1, current_imp - 0.02)
                
            collection.data.update(
                uuid=uuid_str,
                properties={
                    "importance": new_imp,
                    "retrieval_count": retrieval_count
                }
            )
    except Exception as e:
        print(f"[VectorMemory] Failed to bump memory scores: {e}")


# ============================================
# CLEANUP
# ============================================

async def cleanup_expired(user_id: str) -> int:
    """Delete expired memories from both stores."""
    deleted = 0
    
    # Weaviate cleanup
    if VECTOR_MEMORY_ENABLED:
        try:
            from app.memory.weaviate_client import get_weaviate_client, COLLECTION_NAME
            client = await get_weaviate_client()
            if client:
                collection = client.collections.get(COLLECTION_NAME)
                results = collection.query.fetch_objects(
                    filters=_user_filter(user_id),
                    limit=MAX_MEMORIES_PER_USER,
                )
                now = datetime.utcnow()
                for obj in results.objects:
                    expires_at = obj.properties.get("expires_at", "")
                    if expires_at:
                        try:
                            if datetime.fromisoformat(expires_at) < now:
                                collection.data.delete_by_id(obj.uuid)
                                deleted += 1
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            print(f"[VectorMemory] Weaviate cleanup error: {e}")
    
    # Firestore cleanup
    try:
        from app.auth import get_firestore_db
        db = get_firestore_db()
        if db:
            ref = db.collection("users").document(user_id).collection("vector_memories")
            docs = ref.limit(MAX_MEMORIES_PER_USER).stream()
            now = datetime.utcnow()
            for doc in docs:
                data = doc.to_dict()
                expires_at = data.get("expires_at", "")
                if expires_at:
                    try:
                        if datetime.fromisoformat(expires_at) < now:
                            doc.reference.delete()
                            deleted += 1
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        print(f"[VectorMemory] Firestore cleanup error: {e}")
    
    if deleted:
        print(f"[VectorMemory] Cleaned {deleted} expired memories for {user_id}")
    return deleted


# ============================================
# PROMPT FORMATTING
# ============================================

def format_memories_for_prompt(memories: List[MemoryResult]) -> str:
    """Format retrieved memories as a compact XML block for system prompt injection."""
    if not memories:
        return ""
    
    lines = [f"- {mem.text}" for mem in memories]
    
    return (
        "\n<memories>\n"
        + "\n".join(lines)
        + "\n</memories>\n"
    )


# ============================================
# ORCHESTRATOR
# ============================================

async def extract_and_store_memories(
    user_id: str,
    user_message: str,
    assistant_response: str,
) -> int:
    """Full pipeline: extract → store (Firestore + Weaviate) → cleanup."""
    from app.memory.memory_extractor import extract_facts
    
    facts = await extract_facts(user_message, assistant_response, user_id)
    
    if not facts:
        return 0
    
    stored_count = 0
    for fact in facts:
        success = await store_memory(
            user_id=user_id,
            text=fact.text,
            memory_type=fact.memory_type,
            importance=fact.importance,
            expires_at=fact.expires_at,
        )
        if success:
            stored_count += 1
    
    # Background cleanup
    asyncio.create_task(cleanup_expired(user_id))
    
    return stored_count


# ============================================
# HELPERS
# ============================================

def _user_filter(user_id: str):
    from weaviate.classes.query import Filter
    return Filter.by_property("user_id").equal(user_id)


def _importance_sort_asc():
    from weaviate.classes.query import Sort
    return Sort.by_property("importance", ascending=True)


def _text_overlap(a: str, b: str) -> float:
    """Jaccard word overlap for diversity filtering."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


async def _delete_memory_safe(collection, uuid):
    try:
        collection.data.delete_by_id(uuid)
    except Exception:
        pass


async def _reinforce_memory(collection, uuid_str: str, current_importance: float):
    """Bump importance and retrieval_count when memory is used."""
    try:
        collection.data.update(
            uuid=uuid_str,
            properties={
                "last_seen": datetime.utcnow().isoformat(),
                "importance": min(1.0, current_importance + 0.05),
            }
        )
    except Exception:
        pass
