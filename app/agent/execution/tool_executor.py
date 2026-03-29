from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.tools.registry import registry

_BACKOFF_SECONDS = 1.0
_OPEN_SECONDS = 30.0
_OPEN_THRESHOLD = 3

_TOOL_HEALTH: Dict[str, Dict[str, Any]] = {}
_IDEMPOTENCY_CACHE: Dict[tuple[str, str], "ToolResult"] = {}


@dataclass
class ToolCall:
    name: str
    args: str = ""
    session_id: str = ""
    user_id: str = ""
    idempotency_key: str = ""


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    source: str = ""
    confidence: str = "low"
    error: str = ""
    skipped_reason: str = ""
    circuit_state: str = "CLOSED"
    opened_until: float = 0.0
    backoff_until: float = 0.0


def _default_health() -> Dict[str, Any]:
    return {
        "state": "CLOSED",
        "failures": 0,
        "backoff_until": 0.0,
        "opened_until": 0.0,
        "last_error": "",
    }


def _tool_on_failure(tool_name: str, error: str) -> None:
    now = time.time()
    state = _TOOL_HEALTH.setdefault(tool_name, _default_health())
    state["failures"] = int(state.get("failures", 0)) + 1
    state["last_error"] = str(error or "")
    state["backoff_until"] = now + _BACKOFF_SECONDS
    if state["failures"] >= _OPEN_THRESHOLD:
        state["state"] = "OPEN"
        state["opened_until"] = now + _OPEN_SECONDS
    else:
        state["state"] = "CLOSED"


def _tool_on_success(tool_name: str) -> None:
    _TOOL_HEALTH[tool_name] = _default_health()


async def execute_tool(tool_call: ToolCall) -> ToolResult:
    now = time.time()
    tool_name = str(tool_call.name or "").strip()
    state = _TOOL_HEALTH.setdefault(tool_name, _default_health())

    if state.get("state") == "OPEN" and float(state.get("opened_until", 0.0)) > now:
        return ToolResult(
            success=False,
            source=tool_name,
            confidence="low",
            error=str(state.get("last_error") or "circuit open"),
            skipped_reason=f"circuit_open_until:{int(state.get('opened_until', 0.0))}",
            circuit_state="OPEN",
            opened_until=float(state.get("opened_until", 0.0)),
            backoff_until=float(state.get("backoff_until", 0.0)),
        )

    cache_key = (tool_name, str(tool_call.idempotency_key or "").strip())
    if cache_key[1] and cache_key in _IDEMPOTENCY_CACHE:
        cached = _IDEMPOTENCY_CACHE[cache_key]
        return ToolResult(
            success=cached.success,
            data=cached.data,
            source=f"{cached.source}:idempotent",
            confidence=cached.confidence,
            error=cached.error,
            skipped_reason=cached.skipped_reason,
            circuit_state=cached.circuit_state,
            opened_until=cached.opened_until,
            backoff_until=cached.backoff_until,
        )

    handler = registry.get(tool_name)
    if handler is None:
        _tool_on_failure(tool_name, "unknown tool")
        state = _TOOL_HEALTH.get(tool_name, _default_health())
        return ToolResult(
            success=False,
            source=tool_name,
            confidence="low",
            error="unknown tool",
            circuit_state=str(state.get("state", "CLOSED")),
            opened_until=float(state.get("opened_until", 0.0)),
            backoff_until=float(state.get("backoff_until", 0.0)),
        )

    try:
        raw = await handler(
            args=str(tool_call.args or ""),
            session_id=str(tool_call.session_id or ""),
            user_id=str(tool_call.user_id or ""),
        )
    except Exception as exc:
        _tool_on_failure(tool_name, str(exc))
        state = _TOOL_HEALTH.get(tool_name, _default_health())
        return ToolResult(
            success=False,
            source=tool_name,
            confidence="low",
            error=str(exc),
            circuit_state=str(state.get("state", "CLOSED")),
            opened_until=float(state.get("opened_until", 0.0)),
            backoff_until=float(state.get("backoff_until", 0.0)),
        )

    ok = isinstance(raw, dict) and str(raw.get("status", "")).lower() == "success"
    if not ok:
        _tool_on_failure(tool_name, str((raw or {}).get("data") or "tool failure"))
        state = _TOOL_HEALTH.get(tool_name, _default_health())
        result = ToolResult(
            success=False,
            data=(raw or {}).get("data"),
            source=str((raw or {}).get("source") or tool_name),
            confidence=str((raw or {}).get("confidence") or "low"),
            error=str((raw or {}).get("data") or "tool failure"),
            circuit_state=str(state.get("state", "CLOSED")),
            opened_until=float(state.get("opened_until", 0.0)),
            backoff_until=float(state.get("backoff_until", 0.0)),
        )
    else:
        _tool_on_success(tool_name)
        state = _TOOL_HEALTH.get(tool_name, _default_health())
        result = ToolResult(
            success=True,
            data=(raw or {}).get("data"),
            source=str((raw or {}).get("source") or tool_name),
            confidence=str((raw or {}).get("confidence") or "high"),
            circuit_state=str(state.get("state", "CLOSED")),
            opened_until=float(state.get("opened_until", 0.0)),
            backoff_until=float(state.get("backoff_until", 0.0)),
        )

    if cache_key[1]:
        _IDEMPOTENCY_CACHE[cache_key] = result
    return result

