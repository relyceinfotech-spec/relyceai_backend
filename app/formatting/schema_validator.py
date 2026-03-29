from __future__ import annotations

from typing import Any, Dict


class ResponseSchemaError(ValueError):
    pass


def validate_final_payload(payload: Dict[str, Any], chat_mode: str = "smart") -> Dict[str, Any]:
    data = dict(payload or {})
    key_points = data.get("key_points")
    if key_points is not None and not isinstance(key_points, list):
        raise ResponseSchemaError("key_points must be a list")
    sources = data.get("sources")
    if sources is not None and not isinstance(sources, list):
        raise ResponseSchemaError("sources must be a list")
    if "answer" in data and not isinstance(data.get("answer"), str):
        raise ResponseSchemaError("answer must be a string")
    return data


def build_schema_fallback(chat_mode: str = "smart") -> Dict[str, Any]:
    mode = str(chat_mode or "smart").strip().lower()
    if mode in {"smart", "normal"}:
        mode_label = "normal" if mode == "normal" else "smart"
        return {
            "mode": mode_label,
            "answer_type": "plain",
            "summary": "",
            "answer": "",
            "response": "",
            "key_points": [],
            "sources": [],
        }
    return {
        "summary": "",
        "answer": "",
        "response": "",
        "key_points": [],
        "sources": [],
        "confidence": 0.0,
    }
