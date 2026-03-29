from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from app.auth import get_firestore_db
from app.retrieval.vector_index import VectorIndex


TOPIC_COLLECTION = "agent_topic_knowledge"


class KnowledgeStore:
    """Firestore-backed topic knowledge store."""

    def __init__(self, vector_index: Optional[VectorIndex] = None, collection_name: str = TOPIC_COLLECTION):
        self.collection_name = collection_name
        self.vector_index = vector_index or VectorIndex()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def make_topic_id(topic: str, user_id: str = "global") -> str:
        normalized = " ".join((topic or "").lower().strip().split())
        digest = hashlib.sha1(f"{user_id}:{normalized}".encode("utf-8")).hexdigest()[:24]
        return f"topic_{digest}"

    async def get(self, topic_id: str) -> Optional[Dict[str, Any]]:
        if not topic_id:
            return None

        db = get_firestore_db()
        if not db:
            return None

        try:
            snap = db.collection(self.collection_name).document(topic_id).get()
            if snap and snap.exists:
                payload = snap.to_dict() or {}
                payload["topic_id"] = topic_id
                return payload
        except Exception as exc:
            print(f"[KnowledgeStore] get failed (non-blocking): {exc}")

        return None

    async def get_similar(self, query_embedding: List[float], user_id: str = "global", min_similarity: float = 0.8) -> Optional[Dict[str, Any]]:
        candidates = await self.vector_index.search(query_embedding=query_embedding, user_id=user_id, top_k=3)
        for topic_id, sim in candidates:
            if sim < min_similarity:
                continue
            record = await self.get(topic_id)
            if not record:
                continue
            record["similarity"] = sim
            return record
        return None

    async def put(self, record: Dict[str, Any]) -> str:
        topic = str(record.get("topic", "")).strip()
        user_id = str(record.get("user_id", "global"))
        if not topic:
            raise ValueError("record.topic is required")

        topic_id = str(record.get("topic_id", "")).strip() or self.make_topic_id(topic, user_id=user_id)

        payload = {
            "topic": topic,
            "summary": str(record.get("summary", "")),
            "claims": list(record.get("claims", [])),
            "sources": list(record.get("sources", [])),
            "confidence": float(record.get("confidence", 0.0)),
            "timestamp": str(record.get("timestamp", "")) or self._now_iso(),
            "updated_at": self._now_iso(),
            "created_at": str(record.get("created_at", "")) or self._now_iso(),
            "user_id": user_id,
        }

        db = get_firestore_db()
        if db:
            try:
                db.collection(self.collection_name).document(topic_id).set(payload, merge=True)
            except Exception as exc:
                print(f"[KnowledgeStore] put failed (non-blocking): {exc}")

        return topic_id

    async def update(self, topic_id: str, record: Dict[str, Any]) -> None:
        if not topic_id:
            return

        db = get_firestore_db()
        if not db:
            return

        payload = dict(record or {})
        payload["updated_at"] = self._now_iso()

        try:
            db.collection(self.collection_name).document(topic_id).set(payload, merge=True)
        except Exception as exc:
            print(f"[KnowledgeStore] update failed (non-blocking): {exc}")
