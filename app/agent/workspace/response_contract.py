from __future__ import annotations

from typing import Any, Dict

from app.context.response_contract import normalize_chat_response as _normalize_chat_response


def normalize_chat_response(raw_result: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
    out = _normalize_chat_response(raw_result, user_query=user_query)
    answer = str(out.get("answer") or "")
    lowered = answer.lower()
    if any(token in lowered for token in ("developer instruction", "internal policy", "system prompt")):
        safe = "I can't share internal system details."
        out["answer"] = safe
        out["response"] = safe
        out["summary"] = safe
        out["key_points"] = []
        out["sources"] = []
    return out

