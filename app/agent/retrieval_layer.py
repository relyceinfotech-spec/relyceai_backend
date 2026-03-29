"""
Agent Retrieval Layer

Split-storage retrieval implementation:
  - Vector index: semantic lookup over topic embeddings (in-memory ANN-like scan).
  - Firestore: canonical topic metadata, summaries, claims, and sources.

Designed to degrade gracefully when Firestore or embedding model is unavailable.
"""
from __future__ import annotations

import math
import time
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.auth import get_firestore_db
from app.llm.embeddings import generate_embedding

try:
    from google.cloud.firestore_v1.base_query import FieldFilter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FieldFilter = None


TOPIC_COLLECTION = "agent_topic_knowledge"
TOPIC_VECTOR_COLLECTION = "agent_topic_vectors"


@dataclass
class RetrievalResult:
    hit: bool = False
    topic_id: str = ""
    similarity: float = 0.0
    record: Optional[Dict[str, Any]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _normalize_topic(query: str) -> str:
    return " ".join((query or "").lower().strip().split())


class TopicVectorIndex:
    """
    Lightweight vector index abstraction.

    v1 implementation keeps vectors in memory and persists them to Firestore.
    This avoids metadata scans on every query and keeps lookup bounded to loaded vectors.
    """

    def __init__(self):
        self._loaded = False
        self._vectors: Dict[str, List[float]] = {}  # topic_id -> embedding
        self._user_scope: str = "global"

    def _ensure_loaded(self, user_id: str = "global") -> None:
        if self._loaded and self._user_scope == user_id:
            return
        self._vectors = {}
        self._user_scope = user_id
        db = get_firestore_db()
        if not db:
            self._loaded = True
            return
        try:
            col = db.collection(TOPIC_VECTOR_COLLECTION)
            if FieldFilter is not None:
                ref = col.where(filter=FieldFilter("user_id", "==", user_id)).stream()
            else:
                ref = col.where("user_id", "==", user_id).stream()
            for doc in ref:
                data = doc.to_dict() or {}
                topic_id = str(data.get("topic_id", "")).strip()
                emb = data.get("embedding") or []
                if topic_id and isinstance(emb, list) and emb:
                    self._vectors[topic_id] = emb
        except Exception as e:
            print(f"[RetrievalLayer] vector load failed (non-blocking): {e}")
        self._loaded = True

    async def search(self, query: str, user_id: str = "global", top_k: int = 3) -> List[Tuple[str, float]]:
        self._ensure_loaded(user_id=user_id)
        qvec = await generate_embedding(query or "")
        if not qvec:
            return []
        scored: List[Tuple[str, float]] = []
        for topic_id, vec in self._vectors.items():
            scored.append((topic_id, _cosine_similarity(qvec, vec)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def upsert(self, topic_id: str, query_text: str, user_id: str = "global") -> None:
        if not topic_id:
            return
        vec = await generate_embedding(query_text or "")
        if not vec:
            return
        self._ensure_loaded(user_id=user_id)
        self._vectors[topic_id] = vec

        db = get_firestore_db()
        if not db:
            return
        try:
            db.collection(TOPIC_VECTOR_COLLECTION).document(topic_id).set(
                {
                    "topic_id": topic_id,
                    "user_id": user_id,
                    "embedding": vec,
                    "updated_at": _utc_now_iso(),
                }
            )
        except Exception as e:
            print(f"[RetrievalLayer] vector upsert failed (non-blocking): {e}")


class TopicKnowledgeStore:
    """
    Firestore metadata store for topic records.
    """

    def get(self, topic_id: str) -> Optional[Dict[str, Any]]:
        if not topic_id:
            return None
        db = get_firestore_db()
        if not db:
            return None
        try:
            snap = db.collection(TOPIC_COLLECTION).document(topic_id).get()
            if snap and snap.exists:
                data = snap.to_dict() or {}
                data["topic_id"] = topic_id
                return data
        except Exception as e:
            print(f"[RetrievalLayer] topic get failed (non-blocking): {e}")
        return None

    def upsert(self, topic_id: str, payload: Dict[str, Any]) -> None:
        if not topic_id:
            return
        db = get_firestore_db()
        if not db:
            return
        try:
            doc = db.collection(TOPIC_COLLECTION).document(topic_id)
            existing = doc.get()
            current = existing.to_dict() if existing.exists else {}
            merged = dict(current or {})
            merged.update(payload or {})
            merged["topic_id"] = topic_id
            merged["updated_at"] = _utc_now_iso()
            if "created_at" not in merged:
                merged["created_at"] = _utc_now_iso()
            doc.set(merged)
        except Exception as e:
            print(f"[RetrievalLayer] topic upsert failed (non-blocking): {e}")


class RetrievalLayer:
    def __init__(self, similarity_threshold: float = 0.90, min_confidence: float = 0.70, ttl_seconds: int = 86400):
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.ttl_seconds = ttl_seconds
        self.vector_index = TopicVectorIndex()
        self.knowledge_store = TopicKnowledgeStore()

    @staticmethod
    def _topic_id(query: str) -> str:
        norm = _normalize_topic(query)
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:24]
        return f"topic_{digest}"

    def _is_fresh(self, record: Dict[str, Any]) -> bool:
        updated = record.get("updated_at") or record.get("timestamp")
        if not updated:
            return False
        try:
            ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00")).timestamp()
            return (time.time() - ts) <= float(record.get("ttl_seconds", self.ttl_seconds))
        except Exception:
            return False

    async def retrieve(self, query: str, user_id: str = "global") -> RetrievalResult:
        # Fast path: exact topic id.
        topic_id = self._topic_id(query)
        exact = self.knowledge_store.get(topic_id)
        if exact and self._is_fresh(exact) and float(exact.get("confidence", 0.0)) >= self.min_confidence:
            return RetrievalResult(hit=True, topic_id=topic_id, similarity=1.0, record=exact)

        # Semantic path.
        candidates = await self.vector_index.search(query, user_id=user_id, top_k=3)
        for cand_id, sim in candidates:
            if sim < self.similarity_threshold:
                continue
            rec = self.knowledge_store.get(cand_id)
            if not rec:
                continue
            if not self._is_fresh(rec):
                continue
            if float(rec.get("confidence", 0.0)) < self.min_confidence:
                continue
            return RetrievalResult(hit=True, topic_id=cand_id, similarity=sim, record=rec)

        return RetrievalResult(hit=False)

    async def upsert_topic(
        self,
        query: str,
        summary: str,
        claims: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        confidence: float,
        user_id: str = "global",
    ) -> str:
        topic_id = self._topic_id(query)
        payload = {
            "topic": _normalize_topic(query),
            "summary": summary,
            "claims": claims,
            "sources": sources,
            "confidence": float(confidence),
            "timestamp": _utc_now_iso(),
            "ttl_seconds": self.ttl_seconds,
            "user_id": user_id,
        }
        self.knowledge_store.upsert(topic_id, payload)
        await self.vector_index.upsert(topic_id, query_text=query, user_id=user_id)
        return topic_id
