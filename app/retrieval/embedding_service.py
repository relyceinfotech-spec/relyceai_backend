from __future__ import annotations

from typing import List

from app.llm.embeddings import generate_embedding


class EmbeddingService:
    """Async embedding adapter used by retrieval components."""

    async def embed(self, text: str) -> List[float]:
        return await generate_embedding(text or "")
