from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from app.auth import get_firestore_db


RUN_COLLECTION = "agent_run_history"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(record or {})
    payload.setdefault("query", "")
    payload.setdefault("route_type", "agent_pipeline")
    payload.setdefault("plan", [])
    payload.setdefault("tools_used", [])
    payload.setdefault("claims", [])
    payload.setdefault("verification_result", {})
    payload.setdefault("sources", [])
    payload.setdefault("latency_ms", 0)
    payload.setdefault("confidence", 0.0)
    payload.setdefault("input_token_count", 0)
    payload.setdefault("output_token_count", 0)
    payload.setdefault("queue_wait_ms", 0)
    payload.setdefault("model_latency_ms", 0)
    payload.setdefault("compression_applied", False)
    payload.setdefault("compression_ratio", 0.0)
    return payload


def log_run(record: Dict[str, Any], user_id: str = "global") -> bool:
    db = get_firestore_db()
    if not db:
        return False
    try:
        payload = _normalize_record(record)
        payload["user_id"] = user_id
        payload["logged_at"] = _now_iso()
        db.collection(RUN_COLLECTION).add(payload)
        return True
    except Exception as exc:
        print(f"[RunHistory] log failed (non-blocking): {exc}")
        return False


def get_recent_runs(user_id: str = "global", limit: int = 500) -> List[Dict[str, Any]]:
    db = get_firestore_db()
    if not db:
        return []
    try:
        docs = (
            db.collection(RUN_COLLECTION)
            .where("user_id", "==", user_id)
            .order_by("logged_at", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        return [doc.to_dict() or {} for doc in docs]
    except Exception as exc:
        print(f"[RunHistory] fetch failed (non-blocking): {exc}")
        return []
