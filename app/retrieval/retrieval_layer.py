from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.vector_index import VectorIndex
from app.retrieval.knowledge_store import KnowledgeStore


@dataclass
class RetrievalHit:
    hit: bool = False
    topic_id: str = ""
    similarity: float = 0.0
    record: Optional[Dict[str, Any]] = None


class RetrievalLayer:
    """Query normalization + semantic retrieval + knowledge upsert."""

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        min_confidence: float = 0.8,
        ttl_seconds: int = 86400,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.ttl_seconds = ttl_seconds
        self.embedding_service = EmbeddingService()
        self.vector_index = VectorIndex()
        self.knowledge_store = KnowledgeStore(vector_index=self.vector_index)

    @staticmethod
    def normalize_query(query: str) -> str:
        q = (query or "").lower().strip()
        q = re.sub(r"[^a-z0-9\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _topic_id(self, normalized_query: str, user_id: str) -> str:
        return self.knowledge_store.make_topic_id(normalized_query, user_id=user_id)

    def _is_fresh(self, record: Dict[str, Any]) -> bool:
        updated = record.get("updated_at") or record.get("timestamp")
        if not updated:
            return False
        try:
            dt = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
            age_seconds = (datetime.now(timezone.utc) - dt).total_seconds()
            ttl = float(record.get("ttl_seconds", self.ttl_seconds))
            return age_seconds <= ttl
        except Exception:
            return False

    def _is_confident(self, record: Dict[str, Any]) -> bool:
        return float(record.get("confidence", 0.0)) >= self.min_confidence

    async def lookup(self, query: str, user_id: str = "global") -> RetrievalHit:
        normalized = self.normalize_query(query)
        if not normalized:
            return RetrievalHit(hit=False)

        topic_id = self._topic_id(normalized, user_id=user_id)
        exact = await self.knowledge_store.get(topic_id)
        if exact and self._is_fresh(exact) and self._is_confident(exact):
            return RetrievalHit(hit=True, topic_id=topic_id, similarity=1.0, record=exact)

        query_embedding = await self.embedding_service.embed(normalized)
        if not query_embedding:
            return RetrievalHit(hit=False)

        similar = await self.knowledge_store.get_similar(
            query_embedding=query_embedding,
            user_id=user_id,
            min_similarity=self.similarity_threshold,
        )
        if similar and self._is_fresh(similar) and self._is_confident(similar):
            return RetrievalHit(
                hit=True,
                topic_id=str(similar.get("topic_id", "")),
                similarity=float(similar.get("similarity", 0.0)),
                record=similar,
            )

        return RetrievalHit(hit=False)

    async def upsert_topic(
        self,
        query: str,
        summary: str,
        claims: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        confidence: float,
        user_id: str = "global",
    ) -> str:
        normalized = self.normalize_query(query)
        topic_id = self._topic_id(normalized, user_id=user_id)
        payload: Dict[str, Any] = {
            "topic_id": topic_id,
            "topic": normalized,
            "summary": summary,
            "claims": claims,
            "sources": sources,
            "confidence": float(confidence),
            "timestamp": self._now_iso(),
            "ttl_seconds": self.ttl_seconds,
            "user_id": user_id,
        }

        await self.knowledge_store.update(topic_id, payload)

        emb = await self.embedding_service.embed(normalized)
        if emb:
            await self.vector_index.upsert(topic_id=topic_id, embedding=emb, user_id=user_id)

        return topic_id
