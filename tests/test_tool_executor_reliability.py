import asyncio
import time

from app.agent.execution import tool_executor as exec_tools
from app.tools.registry import registry


async def _ok_tool(args: str = "", session_id: str = "", user_id: str = ""):
    return {"status": "success", "data": {"ok": True}, "source": "test", "confidence": "high"}


async def _fail_tool(args: str = "", session_id: str = "", user_id: str = ""):
    return {"status": "failure", "data": None, "source": "failure", "confidence": "low"}


def test_tool_backoff_and_circuit_transitions():
    tool_name = "unit_fail_tool"
    exec_tools._TOOL_HEALTH.pop(tool_name, None)

    exec_tools._tool_on_failure(tool_name, "x")
    st = exec_tools._TOOL_HEALTH[tool_name]
    assert st["failures"] == 1
    assert st["backoff_until"] > time.time()
    assert st["state"] == "CLOSED"

    exec_tools._tool_on_failure(tool_name, "x")
    exec_tools._tool_on_failure(tool_name, "x")
    st = exec_tools._TOOL_HEALTH[tool_name]
    assert st["state"] == "OPEN"
    assert st["opened_until"] > time.time()


def test_execute_tool_returns_reliability_metadata_when_blocked():
    tool_name = "unit_blocked_tool"
    registry.register(tool_name, _ok_tool)
    exec_tools._TOOL_HEALTH[tool_name] = {
        "state": "OPEN",
        "failures": 3,
        "backoff_until": 0.0,
        "opened_until": time.time() + 60,
        "last_error": "boom",
    }

    result = asyncio.run(exec_tools.execute_tool(exec_tools.ToolCall(name=tool_name, args="{}")))
    assert result.success is False
    assert result.skipped_reason.startswith("circuit_open_until")
    assert result.circuit_state == "OPEN"
    assert result.opened_until > time.time()


def test_execute_tool_success_includes_reliability_metadata():
    tool_name = "unit_ok_tool"
    registry.register(tool_name, _ok_tool)
    exec_tools._TOOL_HEALTH.pop(tool_name, None)

    result = asyncio.run(exec_tools.execute_tool(exec_tools.ToolCall(name=tool_name, args="{}")))
    assert result.success is True
    assert result.circuit_state == "CLOSED"
    assert result.backoff_until == 0.0


def test_execute_tool_failure_updates_reliability_metadata():
    tool_name = "unit_retry_fail_tool"
    registry.register(tool_name, _fail_tool)
    exec_tools._TOOL_HEALTH.pop(tool_name, None)

    result = asyncio.run(exec_tools.execute_tool(exec_tools.ToolCall(name=tool_name, args="{}")))
    assert result.success is False
    assert result.backoff_until > 0.0
    assert exec_tools._TOOL_HEALTH[tool_name]["failures"] >= 1

def test_execute_tool_idempotency_reuses_cached_result():
    tool_name = "unit_idempotent_tool"
    call_count = {"n": 0}

    async def _idempotent_tool(args: str = "", session_id: str = "", user_id: str = ""):
        call_count["n"] += 1
        return {"status": "success", "data": {"value": 42}, "source": "idempotent", "confidence": "high"}

    registry.register(tool_name, _idempotent_tool)

    call = exec_tools.ToolCall(name=tool_name, args='{"x":1}', session_id="s1", user_id="u1", idempotency_key="idem-1")
    first = asyncio.run(exec_tools.execute_tool(call))
    second = asyncio.run(exec_tools.execute_tool(call))

    assert first.success is True
    assert second.success is True
    assert call_count["n"] == 1
    assert str(second.source).endswith(":idempotent")
