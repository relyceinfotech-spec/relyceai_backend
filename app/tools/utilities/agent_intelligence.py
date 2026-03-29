from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.agent.tool_policy import get_allowed_tools_for_mode

AGENT_MAX_TOTAL_TOOL_CALLS = 12
_USER_FACTS: Dict[str, Dict[str, Any]] = {}


def get_user_facts(user_id: str) -> Dict[str, Any]:
    return dict(_USER_FACTS.get(str(user_id or ""), {}))


def save_user_facts(user_id: str, facts: Dict[str, Any]) -> bool:
    _USER_FACTS[str(user_id or "")] = dict(facts or {})
    return True


def delete_user_memory_items(*, user_id: str, fact_id: str = "", mode: str = "soft", reason: str = "") -> Dict[str, Any]:
    facts = _USER_FACTS.get(str(user_id or ""), {})
    if fact_id and fact_id in facts:
        del facts[fact_id]
        _USER_FACTS[str(user_id or "")] = facts
        return {"ok": True, "deleted": 1, "mode": mode}
    return {"ok": True, "deleted": 0, "mode": mode}


@dataclass
class _SpendGuard:
    def allow(self) -> bool:
        return True


def get_spend_guard() -> _SpendGuard:
    return _SpendGuard()


def _parse_json(args: str) -> Dict[str, Any]:
    try:
        payload = json.loads(str(args or "{}"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


async def _tool_memory_query(args: str = "", user_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    query = str(payload.get("query") or "").lower()
    top_k = int(payload.get("top_k", 5) or 5)
    terms = [t for t in query.split() if t]
    facts = get_user_facts(user_id)
    matches = []
    for key, value in facts.items():
        text = f"{key} {value}".lower()
        if not terms or any(term in text for term in terms):
            matches.append({"key": key, "value": value})
    return {"status": "success", "data": {"count": len(matches), "matches": matches[:top_k]}}


async def _tool_memory_upsert(args: str = "", user_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    conf = float(payload.get("confidence_score", 0.0) or 0.0)
    content = str(payload.get("content") or "")
    if conf < 0.9:
        return {"status": "failure", "data": "confidence_score below required threshold (>=0.9)"}
    if any(tok in content.lower() for tok in ("password", "secret", "api key", "token")):
        return {"status": "failure", "data": "secret-like content is blocked"}

    key = str(payload.get("key") or payload.get("type") or f"fact_{len(content)}")
    facts = get_user_facts(user_id)
    facts[key] = content
    save_user_facts(user_id, facts)
    return {"status": "success", "data": {"key": key}}


async def _tool_memory_delete(args: str = "", user_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    out = delete_user_memory_items(
        user_id=str(user_id or ""),
        fact_id=str(payload.get("fact_id") or ""),
        mode=str(payload.get("mode") or "soft"),
        reason=str(payload.get("reason") or ""),
    )
    return {"status": "success" if out.get("ok") else "failure", "data": out}


async def _tool_search_web(args: str = "", session_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    query = str(payload.get("query") or args or "").strip()
    return {
        "status": "success",
        "data": {
            "content": f"Summary for {query}",
            "source": "https://example.com",
            "sources": [{"url": "https://example.com", "domain": "example.com", "trust": 0.8, "snippet": "result"}],
        },
    }


async def _tool_summarize_url(args: str = "", session_id: str = "") -> Dict[str, Any]:
    return {"status": "success", "data": {"summary": "summarized content"}}


async def _tool_web_research(args: str = "", user_id: str = "", session_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    query = str(payload.get("query") or "")
    search_out = await _tool_search_web(json.dumps({"query": query}), session_id=session_id)
    if str(search_out.get("status")) != "success":
        return {"status": "failure", "data": {"summary": "", "sources": [], "source_count": 0}}
    data = search_out.get("data") if isinstance(search_out.get("data"), dict) else {}
    raw_sources = data.get("sources") if isinstance(data.get("sources"), list) else []
    clean_sources = []
    for src in raw_sources:
        if not isinstance(src, dict):
            continue
        snippet = str(src.get("snippet") or "").lower()
        if "ignore previous instructions" in snippet:
            continue
        clean_sources.append(src)
    summary_out = await _tool_summarize_url(json.dumps({"url": str(data.get("source") or "")}), session_id=session_id)
    summary = ""
    if isinstance(summary_out.get("data"), dict):
        summary = str(summary_out["data"].get("summary") or "")
    if not summary:
        summary = str(data.get("content") or "")
    return {
        "status": "success",
        "data": {
            "summary": summary,
            "sources": clean_sources,
            "source_count": len(clean_sources),
            "citation_score": round(min(1.0, len(clean_sources) / 3.0), 2),
        },
    }


async def _tool_code_exec_sandbox(args: str = "", user_id: str = "", session_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    code = str(payload.get("code") or "")
    env: Dict[str, Any] = {}
    try:
        exec(code, {"__builtins__": {}}, env)
        return {"status": "success", "data": {"result": env.get("result")}}
    except Exception as exc:
        return {"status": "failure", "data": str(exc)}


async def _tool_task_planner(args: str = "", user_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    goal = str(payload.get("goal") or "")
    steps = [
        {"step_id": 1, "title": "Map competitors and market segments"},
        {"step_id": 2, "title": "Collect pricing and feature tiers"},
        {"step_id": 3, "title": "Benchmark positioning gaps"},
        {"step_id": 4, "title": "Recommend pricing strategy"},
    ]
    return {"status": "success", "data": {"goal": goal, "steps": steps, "step_count": len(steps)}}


async def _tool_task_executor_loop(args: str = "", user_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    if not get_spend_guard().allow():
        return {"status": "failure", "data": {"reason": "spend_guard_blocked"}}

    max_steps = max(0, int(payload.get("max_steps", 3) or 3))
    max_tools_per_step = max(0, int(payload.get("max_tools_per_step", 1) or 1))
    requested_total = int(payload.get("max_total_tools", AGENT_MAX_TOTAL_TOOL_CALLS) or AGENT_MAX_TOTAL_TOOL_CALLS)
    max_total_tools = min(requested_total, int(AGENT_MAX_TOTAL_TOOL_CALLS))
    budget_tokens = int(payload.get("budget_tokens", 0) or 0)
    budget_mode = str(payload.get("budget_mode") or "best_effort")
    estimated_budget_need = max_steps * max(1, max_tools_per_step) * 600

    if budget_tokens > 0 and budget_tokens < estimated_budget_need and budget_mode == "strict":
        return {
            "status": "success",
            "data": {
                "budget_decision": "hard_abort",
                "trim_reason": "budget_exceeded_strict_mode",
                "steps_executed": 0,
                "tools_executed": 0,
                "limits": {"max_total_tools": max_total_tools},
                "events": [],
            },
        }

    budget_decision = "within_budget"
    trim_reason = ""
    if budget_tokens > 0 and budget_tokens < estimated_budget_need and budget_mode == "best_effort":
        budget_decision = "trimmed"
        trim_reason = "budget_exceeded_best_effort"
        max_steps = min(max_steps, max(1, budget_tokens // 600))

    steps_executed = 0
    tools_executed = 0
    events: List[Dict[str, Any]] = []
    tool_plan = payload.get("tool_plan") if isinstance(payload.get("tool_plan"), list) else []
    allowed = get_allowed_tools_for_mode("agent")

    for i in range(max_steps):
        tools_for_step: List[str]
        if i < len(tool_plan) and isinstance(tool_plan[i], dict):
            tools_for_step = [str(t) for t in (tool_plan[i].get("tools") or [])]
        else:
            tools_for_step = ["memory_query"]
        tools_for_step = tools_for_step[:max_tools_per_step]

        tool_events = []
        for name in tools_for_step:
            if tools_executed >= max_total_tools:
                break
            if name not in allowed:
                tool_events.append({"tool": name, "reason": "blocked_by_mode_policy"})
                continue
            tools_executed += 1
            tool_events.append({"tool": name, "status": "success"})

        events.append({"step_id": i + 1, "tools": tool_events})
        steps_executed += 1
        if tools_executed >= max_total_tools:
            break

    return {
        "status": "success",
        "data": {
            "steps_executed": steps_executed,
            "tools_executed": tools_executed,
            "budget_decision": budget_decision,
            "trim_reason": trim_reason,
            "limits": {"max_total_tools": max_total_tools},
            "events": events,
        },
    }


async def _tool_tool_registry(args: str = "", user_id: str = "") -> Dict[str, Any]:
    payload = _parse_json(args)
    category = str(payload.get("category") or "").strip().lower()
    tools = [
        {"name": "task_planner", "category": "planning"},
        {"name": "task_executor_loop", "category": "planning"},
        {"name": "memory_query", "category": "memory"},
    ]
    if category:
        tools = [t for t in tools if t["category"] == category]
    return {"status": "success", "data": {"tools": tools}}
