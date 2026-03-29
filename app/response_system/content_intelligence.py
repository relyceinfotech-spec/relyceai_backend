"""Content intelligence layer for response presentation decisions."""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _is_comparison_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ("compare", "comparison", " vs ", "versus", "difference between"))


def _is_timeline_query(query: str) -> bool:
    q = (query or "").lower()
    return any(token in q for token in ("timeline", "history", "chronology", "chronological"))


def detect_content_plan(
    *,
    user_query: str,
    text: str,
    key_points: List[str],
    has_table: bool,
    has_timeline_events: bool,
    has_card_fields: bool,
) -> Dict[str, Any]:
    """
    Decide best visualization plan before block building.

    Returns a small deterministic plan with:
    - primary: one of text/list/table/timeline/card
    - secondary: ordered additional block types
    - reason: human/debug readable explanation
    """
    query = user_query or ""
    body = text or ""

    if has_table or _is_comparison_query(query):
        return {
            "primary": "table",
            "secondary": ["text", "list"],
            "reason": "comparison_detected",
        }

    if has_timeline_events or _is_timeline_query(query):
        return {
            "primary": "timeline",
            "secondary": ["text", "list"],
            "reason": "chronological_detected",
        }

    if has_card_fields and not key_points:
        return {
            "primary": "card",
            "secondary": ["text"],
            "reason": "entity_fields_detected",
        }

    if len(key_points) >= 3:
        return {
            "primary": "list",
            "secondary": ["text"],
            "reason": "multi_fact_detected",
        }

    # Fallback for short explanations
    if len(re.split(r"(?<=[.!?])\s+", body.strip())) <= 2:
        return {
            "primary": "text",
            "secondary": [],
            "reason": "simple_answer_detected",
        }

    return {
        "primary": "text",
        "secondary": ["list"],
        "reason": "default_explanation",
    }
