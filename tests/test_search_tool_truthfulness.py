import asyncio

import app.agent.tool_executor as te
import app.llm.router as router


def test_search_web_fails_when_results_have_no_links(monkeypatch):
    async def _fake_serper(endpoint, queries, param_key="q"):
        return [{"organic": [{"title": "No link result", "snippet": "missing link", "link": ""}]}]

    monkeypatch.setattr(router, "execute_serper_batch", _fake_serper)
    monkeypatch.setattr(router, "get_tools_for_mode", lambda mode: {"Search": "mock://search"})

    out = asyncio.run(te._tool_search_web("iran pakistan mediator", session_id="s-test"))
    assert out["status"] == "failure"
    assert "non-verifiable" in str(out.get("data", "")).lower()


def test_serper_generic_fails_when_provider_returns_no_items(monkeypatch):
    async def _fake_serper(endpoint, queries, param_key="q"):
        return [{}]

    monkeypatch.setattr(router, "execute_serper_batch", _fake_serper)

    out = asyncio.run(te._tool_serper_generic("latest conflict updates", session_id="s-test", tool_key="Search"))
    assert out["status"] == "failure"
    assert "no usable search results" in str(out.get("data", "")).lower()


def test_execute_tool_surfaces_failure_reason_from_tool_data():
    async def _failing_tool(args: str = "", session_id: str = ""):
        return {
            "status": "failure",
            "data": "search provider unavailable",
            "source": "unit_test_search",
            "confidence": "low",
        }

    tool_name = "unit_test_fail_reason_tool"
    te.TOOLS[tool_name] = {
        "func": _failing_tool,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    }
    try:
        result = asyncio.run(te.execute_tool(te.ToolCall(name=tool_name, args="q", session_id="s-test")))
        assert result.success is False
        assert "search provider unavailable" in str(result.error).lower()
    finally:
        te.TOOLS.pop(tool_name, None)
