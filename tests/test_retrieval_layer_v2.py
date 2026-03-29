import pytest

from app.retrieval.retrieval_layer import RetrievalLayer


@pytest.mark.asyncio
async def test_normalize_query_strips_punctuation_and_case():
    layer = RetrievalLayer()
    assert layer.normalize_query("  Who won, the 2024 F1 Championship?! ") == "who won the 2024 f1 championship"


@pytest.mark.asyncio
async def test_lookup_rejects_low_confidence_record(monkeypatch):
    layer = RetrievalLayer(min_confidence=0.8)

    async def fake_get(topic_id):
        return {
            "topic_id": topic_id,
            "summary": "cached",
            "confidence": 0.5,
            "updated_at": "2099-01-01T00:00:00+00:00",
        }

    async def fake_embed(_):
        return []

    monkeypatch.setattr(layer.knowledge_store, "get", fake_get)
    monkeypatch.setattr(layer.embedding_service, "embed", fake_embed)

    result = await layer.lookup("test query", user_id="u1")
    assert result.hit is False
