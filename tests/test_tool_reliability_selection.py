from __future__ import annotations

from app.agent import graph_scheduler as gs
from app.agent import tool_confidence_store as tcs


def _reset_store() -> tcs.ToolConfidenceStore:
    tcs._tool_confidence_singleton = tcs.ToolConfidenceStore()
    return tcs.get_tool_confidence_store()


def test_tool_selection_score_prefers_high_success_low_error_low_latency():
    store = _reset_store()
    for _ in range(12):
        store.record(tool_name="search_web", success=True, relevance=0.9, latency_ms=650, hard_error=False)
    for _ in range(12):
        store.record(tool_name="search_news", success=False, relevance=0.2, latency_ms=2200, hard_error=True)

    web = store.get("search_web")
    news = store.get("search_news")

    assert web is not None and news is not None
    assert float(web.selection_score()) > float(news.selection_score())
    assert float(web.success_rate()) > float(news.success_rate())
    assert float(web.error_rate()) < float(news.error_rate())


def test_reliability_router_swaps_from_weak_to_strong_tool():
    store = _reset_store()
    # Strong candidate
    for _ in range(10):
        store.record(tool_name="search_web", success=True, relevance=0.85, latency_ms=700, hard_error=False)
    # Weak current tool
    for _ in range(10):
        store.record(tool_name="search_news", success=False, relevance=0.1, latency_ms=2600, hard_error=True)

    chosen = gs._choose_reliability_preferred_retrieval_tool(
        current_tool="search_news",
        allowed_tools={"search_web", "search_news", "search_scholar"},
        min_score=0.30,
        min_improvement=0.08,
    )
    assert chosen == "search_web"


def test_reliability_router_keeps_current_when_not_materially_better():
    store = _reset_store()
    for _ in range(8):
        store.record(tool_name="search_news", success=True, relevance=0.8, latency_ms=850, hard_error=False)
    for _ in range(8):
        store.record(tool_name="search_web", success=True, relevance=0.82, latency_ms=900, hard_error=False)

    chosen = gs._choose_reliability_preferred_retrieval_tool(
        current_tool="search_news",
        allowed_tools={"search_web", "search_news"},
        min_score=0.30,
        min_improvement=0.08,
    )
    assert chosen == "search_news"

