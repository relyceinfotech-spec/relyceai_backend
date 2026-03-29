from __future__ import annotations

from typing import Dict, Set


CANONICAL_CHAT_MODES: Set[str] = {"smart", "agent", "research_pro"}
_LEGACY_TO_CANONICAL: Dict[str, str] = {
    "auto": "smart",
    "normal": "smart",
    "hybrid_main": "smart",
    "business": "smart",
    "deepsearch": "research_pro",
    "research": "research_pro",
}


def normalize_chat_mode(mode: str, default: str = "smart") -> str:
    raw = str(mode or "").strip().lower()
    if raw in CANONICAL_CHAT_MODES:
        return raw
    if raw in _LEGACY_TO_CANONICAL:
        return _LEGACY_TO_CANONICAL[raw]
    return default


def is_agent_premium_mode(mode: str) -> bool:
    normalized = normalize_chat_mode(mode)
    return normalized in {"agent", "research_pro"}


def is_structured_mode(mode: str) -> bool:
    normalized = normalize_chat_mode(mode)
    return normalized in {"smart", "agent", "research_pro"}
