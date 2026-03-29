from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.auth import get_firestore_db


ADAPTIVE_CASES_COLLECTION = "adaptive_learning_cases"
ADAPTIVE_STATE_COLLECTION = "adaptive_learning_state"

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
_WS_RE = re.compile(r"\s+")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_for_fingerprint(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    without_urls = _URL_RE.sub(" ", raw)
    without_numbers = _NUMBER_RE.sub(" <num> ", without_urls)
    normalized = _WS_RE.sub(" ", without_numbers).strip()
    return normalized


def query_fingerprint(text: str, length: int = 8) -> str:
    normalized = normalize_for_fingerprint(text)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return digest[: max(4, int(length or 8))]


def adaptive_bucket_key(*, input_text: str, intent: str, mode: str) -> str:
    fp = query_fingerprint(input_text)
    intent_key = str(intent or "general").strip().lower() or "general"
    mode_key = str(mode or "smart").strip().lower() or "smart"
    return f"{fp}:{intent_key}:{mode_key}"


class AdaptiveLearningStore:
    def _db(self):
        try:
            return get_firestore_db()
        except Exception:
            return None

    def log_case(self, *, user_id: str, case_payload: Dict[str, Any]) -> bool:
        db = self._db()
        if not db:
            return False
        try:
            payload = dict(case_payload or {})
            payload["user_id"] = str(user_id or "global")
            payload["logged_at"] = _now_iso()
            db.collection(ADAPTIVE_CASES_COLLECTION).add(payload)
            return True
        except Exception:
            return False

    def upsert_state(self, *, user_id: str, bucket_key: str, state_payload: Dict[str, Any]) -> bool:
        db = self._db()
        if not db:
            return False
        try:
            payload = dict(state_payload or {})
            payload["user_id"] = str(user_id or "global")
            payload["bucket_key"] = str(bucket_key or "")
            payload["updated_at"] = _now_iso()
            doc_id = f"{payload['user_id']}::{payload['bucket_key']}"
            db.collection(ADAPTIVE_STATE_COLLECTION).document(doc_id).set(payload, merge=True)
            return True
        except Exception:
            return False

    def get_state(self, *, user_id: str, bucket_key: str) -> Optional[Dict[str, Any]]:
        db = self._db()
        if not db:
            return None
        try:
            doc_id = f"{str(user_id or 'global')}::{str(bucket_key or '')}"
            doc = db.collection(ADAPTIVE_STATE_COLLECTION).document(doc_id).get()
            if not doc.exists:
                return None
            data = doc.to_dict() or {}
            return data
        except Exception:
            return None

    def list_recent_states(self, *, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        db = self._db()
        if not db:
            return []
        try:
            docs = (
                db.collection(ADAPTIVE_STATE_COLLECTION)
                .where("user_id", "==", str(user_id or "global"))
                .order_by("updated_at", direction="DESCENDING")
                .limit(max(1, int(limit or 50)))
                .stream()
            )
            return [d.to_dict() or {} for d in docs]
        except Exception:
            return []
