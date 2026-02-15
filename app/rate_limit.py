"""
Relyce AI - Chat Rate Limiter (Firestore-backed)
Distributed-safe rate limiting with fixed windows
"""
from datetime import datetime, timedelta, timezone
from typing import Tuple

from firebase_admin import firestore
from app.auth import get_firestore_db
from app.config import RATE_LIMIT_PER_MINUTE, RATE_LIMIT_WINDOW_SECONDS, RATE_LIMIT_FAIL_OPEN

COLLECTION_NAME = "chat_rate_limits"

def _get_window(now: datetime) -> Tuple[int, datetime]:
    bucket = int(now.timestamp() // RATE_LIMIT_WINDOW_SECONDS)
    window_start = datetime.fromtimestamp(bucket * RATE_LIMIT_WINDOW_SECONDS, tz=timezone.utc)
    return bucket, window_start

@firestore.transactional
def _check_and_increment(transaction, doc_ref, now: datetime, window_start: datetime) -> bool:
    doc = doc_ref.get(transaction=transaction)
    if doc.exists:
        data = doc.to_dict() or {}
        count = int(data.get("count", 0) or 0)
        if count >= RATE_LIMIT_PER_MINUTE:
            return False
        transaction.update(doc_ref, {
            "count": firestore.Increment(1),
            "lastRequestAt": now,
        })
        return True

    transaction.set(doc_ref, {
        "count": 1,
        "windowStart": window_start,
        "lastRequestAt": now,
        # TTL field (configure in Firestore if desired)
        "expiresAt": window_start + timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS * 2),
    })
    return True

def check_rate_limit(user_id: str) -> bool:
    """
    Firestore-backed fixed-window rate limit.
    Returns True if allowed, False if rate limit exceeded.
    """
    if not user_id:
        user_id = "anonymous"

    db = get_firestore_db()
    if not db:
        return RATE_LIMIT_FAIL_OPEN

    now = datetime.now(timezone.utc)
    bucket, window_start = _get_window(now)
    doc_ref = db.collection(COLLECTION_NAME).document(f"{user_id}:{bucket}")
    try:
        transaction = db.transaction()
        return _check_and_increment(transaction, doc_ref, now, window_start)
    except Exception as e:
        print(f"[RateLimit] Firestore error: {e}")
        return RATE_LIMIT_FAIL_OPEN
