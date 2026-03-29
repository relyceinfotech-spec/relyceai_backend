from __future__ import annotations

import re

from app.chat.mode_mapper import normalize_chat_mode

_RESEARCH_HEAVY_TRIGGER_RE = re.compile(
    r"\b(latest|today|current|breaking|recent|news|update|updates|new|new data|as of|with sources|source-backed|"
    r"cite|citation|proof|evidence|verify|official|conflict|war|attack|sanction|research|analysis)\b",
    re.IGNORECASE,
)
_TASK_HEAVY_TRIGGER_RE = re.compile(
    r"\b(build|generate|analyze|fix|debug|implement|create|code|refactor|optimize|execute|run|automate)\b",
    re.IGNORECASE,
)
_YEAR_TRIGGER_RE = re.compile(r"\b20\d{2}\b")


def select_queue_lane(chat_mode: str, user_query: str) -> str:
    """
    Predict queue lane at submit-time.
    Mirrors queue routing behavior used by AgentTaskQueue.
    """
    raw_mode = str(chat_mode or "").strip().lower()
    query = str(user_query or "").strip()
    if raw_mode == "auto":
        if query and (
            len(query) > 80
            or bool(_RESEARCH_HEAVY_TRIGGER_RE.search(query))
            or bool(_TASK_HEAVY_TRIGGER_RE.search(query))
            or bool(_YEAR_TRIGGER_RE.search(query))
        ):
            return "heavy"
        return "fast"

    mode = normalize_chat_mode(str(chat_mode or "smart"))
    if mode in {"agent", "research_pro"}:
        return "heavy"

    if mode == "smart":
        if query:
            if (
                len(query) > 120
                or bool(_RESEARCH_HEAVY_TRIGGER_RE.search(query))
                or bool(_YEAR_TRIGGER_RE.search(query))
                or bool(_TASK_HEAVY_TRIGGER_RE.search(query))
            ):
                return "heavy"
    return "fast"
