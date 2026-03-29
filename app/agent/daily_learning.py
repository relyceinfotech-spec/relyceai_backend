"""
Daily Learning Batch

Reads run history and updates versioned runtime config artifacts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict

from app.agent.learning_config_store import load_active_config, write_new_version, activate_version
from app.learning.run_history import get_recent_runs


def _compute_tool_priors(runs: list[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "success": 0.0, "latency_ms": 0.0})
    for run in runs:
        for t in run.get("tools_used", []) or []:
            name = str(t.get("name", "")) if isinstance(t, dict) else str(t)
            if not name:
                continue
            stats[name]["calls"] += 1.0
            if bool(t.get("success", True)) if isinstance(t, dict) else True:
                stats[name]["success"] += 1.0
            if isinstance(t, dict):
                stats[name]["latency_ms"] += float(t.get("latency_ms", 0.0))

    priors: Dict[str, Dict[str, float]] = {}
    for tool, s in stats.items():
        calls = max(1.0, s["calls"])
        priors[tool] = {
            "success_rate": round(s["success"] / calls, 4),
            "avg_latency_ms": round(s["latency_ms"] / calls, 2),
            "calls": int(calls),
        }
    return priors


def _compute_planner_patterns(runs: list[Dict[str, Any]]) -> Dict[str, list[str]]:
    by_type = defaultdict(list)
    for run in runs:
        qtype = str(run.get("query_type", "research"))
        plan = run.get("plan") or []
        if isinstance(plan, list) and plan:
            by_type[qtype].append(tuple(plan))

    patterns: Dict[str, list[str]] = {}
    for qtype, plans in by_type.items():
        if not plans:
            continue
        most_common = Counter(plans).most_common(1)[0][0]
        patterns[qtype] = list(most_common)

    if "research" not in patterns:
        patterns["research"] = ["search", "search", "extract_facts"]
    if "simple_fact" not in patterns:
        patterns["simple_fact"] = ["search"]
    return patterns


def run_daily_learning(user_id: str = "global", activate: bool = True) -> Dict[str, Any]:
    runs = get_recent_runs(user_id=user_id, limit=1000)
    if not runs:
        return {"updated": False, "reason": "no_runs"}

    base = load_active_config()
    next_cfg = dict(base)
    next_cfg["planner_patterns"] = _compute_planner_patterns(runs)
    next_cfg["tool_priors"] = _compute_tool_priors(runs)

    version = write_new_version(next_cfg, note="daily-learning-batch")
    did_activate = bool(activate and activate_version(version))
    return {"updated": True, "version": version, "activated": did_activate, "runs_processed": len(runs)}
