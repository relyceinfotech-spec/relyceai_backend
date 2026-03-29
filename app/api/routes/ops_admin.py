from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app.platform import get_task_queue

_FEATURE_FLAGS: Dict[str, Any] = {}
_QUEUE_CONTROL = {"paused": False}


def _is_admin(user_info: Optional[Dict[str, Any]]) -> bool:
    claims = (user_info or {}).get("claims") if isinstance((user_info or {}).get("claims"), dict) else {}
    return bool(claims.get("admin") or claims.get("role") == "admin" or claims.get("superadmin"))


def _require_admin(user_info: Optional[Dict[str, Any]]) -> None:
    if not _is_admin(user_info):
        raise HTTPException(status_code=403, detail="admin required")


def _audit(*args, **kwargs):
    return None


def _load_trace_entries(run_id: str) -> List[Dict[str, Any]]:
    return []


async def update_feature_flags(*, payload: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    flags = dict(payload or {}).get("flags", {})
    if isinstance(flags, dict):
        _FEATURE_FLAGS.update(flags)
    _audit(action="update_feature_flags", by=(user_info or {}).get("uid"), flags=flags)
    return {"success": True, "flags": dict(_FEATURE_FLAGS)}


async def get_feature_flags(*, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    return {"success": True, "flags": dict(_FEATURE_FLAGS)}


async def queue_control(*, payload: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    action = str(dict(payload or {}).get("action") or "").strip().lower()
    if action == "pause":
        _QUEUE_CONTROL["paused"] = True
    elif action == "resume":
        _QUEUE_CONTROL["paused"] = False
    else:
        raise HTTPException(status_code=400, detail="unsupported action")
    return {"success": True, "queue": dict(_QUEUE_CONTROL)}


async def export_sign_verify(*, payload: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    envelope = dict(payload or {}).get("envelope", {})
    signature = str(dict(payload or {}).get("signature") or "")
    algo = str(dict(payload or {}).get("algo") or "sha256").lower()
    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"), default=str)
    if algo == "sha256":
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    else:
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"success": True, "valid": signature == expected, "algo": algo}


async def policy_lint(*, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    return {"success": True, "findings": []}


async def inspect_run(run_id: str, *, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    queue = get_task_queue()
    task = await queue.get_task(run_id)
    events = await queue.get_events(run_id, after_seq=0) if hasattr(queue, "get_events") else []
    traces = _load_trace_entries(run_id)

    tool_selection = [t for t in traces if str(t.get("stage")) == "PLAN_CREATED"]
    tool_calls = [t for t in traces if str(t.get("stage")) in {"TOOL_CALL", "TOOL_EXEC_START"}]
    tool_results = [t for t in traces if str(t.get("stage")) in {"TOOL_RESULT", "TOOL_EXEC_COMPLETE"}]

    created = float(getattr(task, "created_at", 0.0) or 0.0) if task is not None else 0.0
    started = float(getattr(task, "started_at", 0.0) or 0.0) if task is not None else 0.0
    completed = float(getattr(task, "completed_at", 0.0) or 0.0) if task is not None else 0.0
    run_duration_ms = int(max(0.0, completed - started) * 1000) if started and completed else 0
    created_to_start_ms = int(max(0.0, started - created) * 1000) if created and started else 0

    return {
        "success": True,
        "run_id": run_id,
        "status": str(getattr(task, "status", "unknown")),
        "tool_selection": tool_selection,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "events": events,
        "run_duration_ms": run_duration_ms,
        "created_to_start_ms": created_to_start_ms,
    }


async def replay_run(run_id: str, *, payload: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    mode = str(dict(payload or {}).get("mode") or "trace_only")
    if mode == "full_execution":
        raise HTTPException(status_code=403, detail="full execution replay disabled")
    return {"success": True, "mode": "trace_only", "side_effects": False, "replayed": False, "run_id": run_id}


async def dry_run_agent(*, payload: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    query = str(dict(payload or {}).get("query") or "").strip()
    mode = str(dict(payload or {}).get("mode") or "smart").strip().lower()
    selected_tools = ["search_web"] if any(x in query.lower() for x in ("latest", "news", "today")) else ["memory_query"]
    if mode == "agent":
        selected_tools.append("task_planner")
    token_estimate = max(32, len(query.split()) * 12)
    _audit(action="dry_run_agent", query=query, selected_tools=selected_tools, ts=time.time())
    return {
        "success": True,
        "dry_run": True,
        "side_effects": False,
        "selected_tools": selected_tools,
        "token_estimate": token_estimate,
    }

