from __future__ import annotations

from typing import Iterable, Set


_MODE_TOOL_ALLOWLIST = {
    "normal": {"search_web", "search_news", "memory_query"},
    "smart": {"search_web", "search_news", "memory_query"},
    "agent": {"search_web", "search_news", "memory_query", "task_planner", "task_executor_loop"},
    "coding": {
        "search_web",
        "search_news",
        "memory_query",
        "validate_code",
        "code_exec_sandbox",
        "generate_tests",
    },
    "research_pro": {"search_web", "search_news", "memory_query", "task_planner"},
}


def _normalize_mode(mode: str) -> str:
    raw = str(mode or "normal").strip().lower()
    aliases = {
        "hybrid_main": "smart",
        "business": "smart",
        "deepsearch": "research_pro",
        "research": "research_pro",
    }
    return aliases.get(raw, raw)


def get_allowed_tools_for_mode(mode: str) -> Set[str]:
    normalized = _normalize_mode(mode)
    return set(_MODE_TOOL_ALLOWLIST.get(normalized, _MODE_TOOL_ALLOWLIST["normal"]))


def filter_tools_by_mode(mode: str, selected_tools: Iterable[str]) -> Set[str]:
    selected = {str(x or "").strip() for x in (selected_tools or []) if str(x or "").strip()}
    return selected.intersection(get_allowed_tools_for_mode(mode))

