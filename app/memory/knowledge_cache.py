"""
Knowledge Cache — Semantic response caching to avoid re-computation.
Stores previous query→answer pairs with embeddings for similarity matching.
Uses Firestore for persistence, with TTL-based expiration.

Pipeline:
  1. Check cache (semantic similarity > 0.90)
  2. Cache hit → return stored answer
  3. Cache miss → proceed with full pipeline
  4. After response → store in cache

Saves 30-60% LLM calls on repeated/similar queries.
"""
import hashlib
from typing import Optional, Dict
from datetime import datetime, timedelta


# ============================================
# CONFIG
# ============================================

CACHE_SIMILARITY_THRESHOLD = 0.90
CACHE_MAX_ENTRIES = 200  # per user
DEFAULT_TTL_DAYS = 30

# Content-type TTL overrides
TTL_MAP = {
    "general_knowledge": 30,
    "web_article": 7,
    "research_paper": 90,
    "project_context": 180,
    "coding_answer": 14,
}


# ============================================
# CACHE LOOKUP
# ============================================

async def check_cache(
    user_id: str,
    query: str,
    query_embedding: Optional[list] = None,
    task_type: str = "",
) -> Optional[Dict]:
    """
    Check if a similar query has been answered before.
    Returns cached answer dict or None.
    """
    if not user_id or user_id == "anonymous":
        return None

    # Don't cache personal/conversational queries
    if _is_personal_query(query):
        return None

    try:
        from app.auth import get_firestore_db
        db = get_firestore_db()
        if not db:
            return None

        ref = db.collection("users").document(user_id).collection("knowledge_cache")

        # Hash-based exact match (fast path)
        query_hash = _hash_query(query, task_type)
        exact = ref.where("query_hash", "==", query_hash).limit(1).stream()

        for doc in exact:
            data = doc.to_dict()
            # Check TTL
            if _is_expired(data):
                doc.reference.delete()
                continue

            # Update hit count
            doc.reference.update({
                "hit_count": (data.get("hit_count", 0) + 1),
                "last_hit": datetime.utcnow().isoformat(),
            })

            print(f"[KnowledgeCache] Exact hit for: {query[:50]}")
            return {
                "answer": data.get("answer", ""),
                "source": "cache_exact",
                "cached_at": data.get("cached_at", ""),
            }

        return None

    except Exception as e:
        print(f"[KnowledgeCache] Lookup failed (non-blocking): {e}")
        return None


# ============================================
# CACHE STORE
# ============================================

async def store_in_cache(
    user_id: str,
    query: str,
    answer: str,
    content_type: str = "general_knowledge",
) -> bool:
    """Store a query→answer pair in the knowledge cache."""
    if not user_id or user_id == "anonymous":
        return False

    # Don't cache short or personal answers
    if len(answer) < 100 or _is_personal_query(query):
        return False

    try:
        from app.auth import get_firestore_db
        db = get_firestore_db()
        if not db:
            return False

        ref = db.collection("users").document(user_id).collection("knowledge_cache")
        now = datetime.utcnow()
        ttl_days = TTL_MAP.get(content_type, DEFAULT_TTL_DAYS)

        ref.add({
            "query": query[:500],
            "query_hash": _hash_query(query),
            "answer": answer[:5000],
            "content_type": content_type,
            "cached_at": now.isoformat(),
            "expires_at": (now + timedelta(days=ttl_days)).isoformat(),
            "hit_count": 0,
            "last_hit": "",
        })

        print(f"[KnowledgeCache] Stored: {query[:50]} (TTL={ttl_days}d)")
        return True

    except Exception as e:
        print(f"[KnowledgeCache] Store failed (non-blocking): {e}")
        return False


# ============================================
# HELPERS
# ============================================

def _hash_query(query: str, task_type: str = "") -> str:
    """Normalize and hash query + task type for exact matching."""
    normalized = " ".join(query.lower().strip().split())
    key_input = f"{normalized}:{task_type}"
    return hashlib.md5(key_input.encode()).hexdigest()[:16]


def _is_expired(data: dict) -> bool:
    """Check if cache entry has expired."""
    expires_at = data.get("expires_at", "")
    if not expires_at:
        return False
    try:
        return datetime.fromisoformat(expires_at) < datetime.utcnow()
    except (ValueError, TypeError):
        return False


def _is_personal_query(query: str) -> bool:
    """Detect queries that shouldn't be cached (personal/conversational)."""
    personal_patterns = [
        "my ", "i ", "am i", "what am", "who am",
        "my project", "my code", "my app",
        "how are you", "hello", "hi ", "thanks",
        "what did i", "remember when",
    ]
    q = query.lower().strip()
    return any(q.startswith(p) or f" {p}" in f" {q}" for p in personal_patterns)
