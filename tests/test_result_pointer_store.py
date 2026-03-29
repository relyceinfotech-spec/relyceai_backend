from app.llm.result_pointer_store import (
    get_result_payload,
    list_result_pointers,
    store_tool_result_pointer,
)


def test_pointer_store_roundtrip_memory_fallback():
    row = store_tool_result_pointer(
        session_id="test_sess",
        tool_name="search_web",
        status="success",
        summary="example summary",
        payload={"k": "v"},
        source="search_web",
    )
    assert row.get("result_id")

    pointers = list_result_pointers("test_sess", limit=5)
    assert pointers
    rid = pointers[0]["result_id"]

    payload = get_result_payload("test_sess", rid)
    assert payload == {"k": "v"}
