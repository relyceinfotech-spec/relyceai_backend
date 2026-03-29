from __future__ import annotations

import re
from typing import Any, Dict, List


def _terms(value: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(value or "").lower()) if len(t) > 2]


def select_relevant_result_pointers(
    *,
    step_intent: str,
    pointers: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    intent_terms = set(_terms(step_intent))
    scored = []
    for row in pointers or []:
        summary = str((row or {}).get("summary") or "")
        row_terms = set(_terms(summary))
        overlap = len(intent_terms.intersection(row_terms))
        recency = float((row or {}).get("ts") or 0.0)
        score = overlap * 10 + min(recency / 1_000_000_000, 10)
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[: max(1, int(top_k))]]

