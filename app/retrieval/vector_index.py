from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Any

from app.auth import get_firestore_db

try:
    from google.cloud.firestore_v1.base_query import FieldFilter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    FieldFilter = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VectorIndex:
    """
    Topic vector index with FAISS acceleration when available.
    Falls back to cosine scan when FAISS/numpy is unavailable.
    """

    def __init__(self, collection_name: str = "agent_topic_vectors"):
        self.collection_name = collection_name
        self._loaded_user: Optional[str] = None
        self._all_vectors: Dict[str, List[float]] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._active_dim: Optional[int] = None
        self._faiss_index: Optional[Any] = None
        self._faiss_ids: List[str] = []

    def _build_faiss(self) -> None:
        self._faiss_index = None
        self._faiss_ids = []
        if faiss is None or np is None or not self._vectors:
            return

        ids = list(self._vectors.keys())
        matrix = np.array([self._vectors[i] for i in ids], dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return

        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        self._faiss_index = index
        self._faiss_ids = ids

    def _set_active_dim(self, dim: int) -> None:
        """Switch active search space to vectors matching the requested dimension."""
        self._active_dim = dim
        self._vectors = {
            topic_id: emb
            for topic_id, emb in self._all_vectors.items()
            if isinstance(emb, list) and len(emb) == dim
        }
        self._build_faiss()

    def _ensure_loaded(self, user_id: str) -> None:
        if self._loaded_user == user_id:
            return

        self._all_vectors = {}
        self._vectors = {}
        self._active_dim = None
        self._loaded_user = user_id

        db = get_firestore_db()
        if not db:
            self._build_faiss()
            return

        try:
            col = db.collection(self.collection_name)
            if FieldFilter is not None:
                docs = col.where(filter=FieldFilter("user_id", "==", user_id)).stream()
            else:
                docs = col.where("user_id", "==", user_id).stream()
            for d in docs:
                payload = d.to_dict() or {}
                topic_id = str(payload.get("topic_id", d.id)).strip()
                emb = payload.get("embedding")
                if topic_id and isinstance(emb, list) and emb:
                    self._all_vectors[topic_id] = emb
        except Exception as exc:
            print(f"[VectorIndex] load failed (non-blocking): {exc}")

        # Pick most common embedding dimension as default active set.
        dim_counts: Dict[int, int] = {}
        for emb in self._all_vectors.values():
            dim = len(emb)
            if dim > 0:
                dim_counts[dim] = dim_counts.get(dim, 0) + 1
        if dim_counts:
            default_dim = max(dim_counts, key=dim_counts.get)
            self._set_active_dim(default_dim)
        else:
            self._build_faiss()

    async def search(self, query_embedding: List[float], user_id: str = "global", top_k: int = 3) -> List[Tuple[str, float]]:
        if not query_embedding:
            return []

        self._ensure_loaded(user_id=user_id)
        query_dim = len(query_embedding)

        # Auto-switch active index if query uses a different embedding dimension
        # (e.g. after moving from local MiniLM to OpenRouter embeddings).
        if query_dim > 0 and query_dim != self._active_dim:
            self._set_active_dim(query_dim)

        if not self._vectors:
            return []

        if self._faiss_index is not None and np is not None:
            q = np.array([query_embedding], dtype="float32")
            faiss.normalize_L2(q)
            try:
                scores, indices = self._faiss_index.search(q, max(1, top_k))
            except AssertionError:
                # Last-resort guard: if FAISS still sees a dim mismatch for any reason,
                # rebuild against query dimension and retry once.
                self._set_active_dim(query_dim)
                if self._faiss_index is None:
                    return []
                scores, indices = self._faiss_index.search(q, max(1, top_k))

            out: List[Tuple[str, float]] = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0 or idx >= len(self._faiss_ids):
                    continue
                out.append((self._faiss_ids[idx], float(score)))
            return out

        scored = [(topic_id, _cosine_similarity(query_embedding, emb)) for topic_id, emb in self._vectors.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def upsert(self, topic_id: str, embedding: List[float], user_id: str = "global") -> None:
        if not topic_id or not embedding:
            return

        self._ensure_loaded(user_id=user_id)
        self._all_vectors[topic_id] = embedding
        if self._active_dim != len(embedding):
            self._set_active_dim(len(embedding))
        else:
            self._vectors[topic_id] = embedding
            self._build_faiss()

        db = get_firestore_db()
        if not db:
            return

        try:
            db.collection(self.collection_name).document(topic_id).set(
                {
                    "topic_id": topic_id,
                    "user_id": user_id,
                    "embedding": embedding,
                },
                merge=True,
            )
        except Exception as exc:
            print(f"[VectorIndex] upsert failed (non-blocking): {exc}")
