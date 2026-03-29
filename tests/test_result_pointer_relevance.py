from app.llm.result_pointer_store import store_tool_result_pointer, list_result_pointers
from app.llm.relevance_selector import select_relevant_result_pointers


def test_result_pointer_store_and_list():
    sid = "sess_pointer_test"
    store_tool_result_pointer(
        session_id=sid,
        tool_name="search_web",
        status="success",
        summary="Top competitor pricing tiers extracted",
        payload={"items": [1, 2, 3]},
        source="search_web",
    )
    rows = list_result_pointers(sid, limit=10)
    assert len(rows) >= 1
    assert rows[-1]["result_id"].startswith("res_")
    assert rows[-1]["summary"]


def test_relevance_selector_prefers_matching_intent():
    pointers = [
        {"result_id": "r1", "tool": "search_web", "summary": "Competitor pricing tiers", "status": "success", "ts": 1000},
        {"result_id": "r2", "tool": "memory_query", "summary": "User likes dark theme", "status": "success", "ts": 1001},
    ]
    out = select_relevant_result_pointers(step_intent="extract competitor pricing", pointers=pointers, top_k=1)
    assert len(out) == 1
    assert out[0]["result_id"] == "r1"
