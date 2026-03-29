import asyncio

from app.agent.task_manager import TaskManager
from app.retrieval.retrieval_layer import RetrievalHit


def test_task_manager_query_cache_route():
    async def _run():
        tm = TaskManager(client=None)
        key = tm._cache_key_for_goal(tm.retrieval.normalize_query("who won 2024 f1"))
        tm.response_cache[key] = "cached-answer"

        chunks = []
        async for c in tm.run_goal("who won 2024 f1"):
            chunks.append(c)

        payload = "".join(chunks)
        assert "query_cache_hit" in payload
        assert "cached-answer" in payload

    asyncio.run(_run())


def test_task_manager_retrieval_hit_route(monkeypatch):
    async def _run():
        tm = TaskManager(client=None)

        async def fake_lookup(query, user_id="global"):
            return RetrievalHit(
                hit=True,
                topic_id="topic_1",
                similarity=0.93,
                record={"summary": "retrieval-answer", "confidence": 0.9, "sources": [], "claims": []},
            )

        monkeypatch.setattr(tm.retrieval, "lookup", fake_lookup)

        chunks = []
        async for c in tm.run_goal("who won 2024 f1"):
            chunks.append(c)

        payload = "".join(chunks)
        assert "retrieval_hit" in payload
        assert "retrieval-answer" in payload

    asyncio.run(_run())


def test_task_manager_fast_path_route(monkeypatch):
    async def _run():
        tm = TaskManager(client=None)

        async def fake_lookup(query, user_id="global"):
            return RetrievalHit(hit=False)

        async def fake_fast_path(goal, query_type, user_scope, metrics):
            return ("fast-path-answer", 0.88)

        monkeypatch.setattr(tm.retrieval, "lookup", fake_lookup)
        monkeypatch.setattr(tm, "_execute_fast_path", fake_fast_path)

        chunks = []
        async for c in tm.run_goal("weather in chennai"):
            chunks.append(c)

        payload = "".join(chunks)
        assert "\"mode\": \"fast_path\"" in payload
        assert "fast-path-answer" in payload

    asyncio.run(_run())
