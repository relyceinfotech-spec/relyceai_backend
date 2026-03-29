"""
Relyce AI - Login Rate Limiter
Firestore-based rate limiting with progressive delays.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from app.auth import get_firestore_db


DELAY_CONFIG = {
    1: 0,
    2: 0,
    3: 30,
    4: 120,
    5: 600,
}
MAX_ATTEMPTS = 5
LOCKOUT_DURATION = 600
EXPIRY_MINUTES = 30
COLLECTION_NAME = "login_attempts"


def _get_attempt_key(email: str, ip: str) -> str:
    combined = f"{email.lower().strip()}:{ip}"
    return hashlib.sha256(combined.encode()).hexdigest()


def _format_time(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = seconds // 60
    remainder = seconds % 60
    if remainder == 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{minutes}:{remainder:02d}"


def check_rate_limit(email: str, ip: str) -> dict:
    db = get_firestore_db()
    if not db:
        return {"allowed": True, "wait_seconds": 0, "attempts": 0, "message": ""}

    key = _get_attempt_key(email, ip)
    doc_ref = db.collection(COLLECTION_NAME).document(key)
    doc = doc_ref.get()
    if not doc.exists:
        return {"allowed": True, "wait_seconds": 0, "attempts": 0, "message": ""}

    data = doc.to_dict() or {}
    attempts = int(data.get("attempts", 0) or 0)
    last_attempt = data.get("last_attempt")
    locked_until = data.get("locked_until")
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    if last_attempt:
        last_time = last_attempt if isinstance(last_attempt, datetime) else last_attempt.replace(tzinfo=None)
        if (now - last_time) > timedelta(minutes=EXPIRY_MINUTES):
            doc_ref.delete()
            return {"allowed": True, "wait_seconds": 0, "attempts": 0, "message": ""}

    if locked_until:
        lock_time = locked_until if isinstance(locked_until, datetime) else locked_until.replace(tzinfo=None)
        if now < lock_time:
            wait_seconds = int((lock_time - now).total_seconds())
            return {
                "allowed": False,
                "wait_seconds": wait_seconds,
                "attempts": attempts,
                "message": f"Too many failed attempts. Try again in {_format_time(wait_seconds)}.",
            }

    return {"allowed": True, "wait_seconds": 0, "attempts": attempts, "message": ""}


def record_failed_attempt(email: str, ip: str) -> dict:
    db = get_firestore_db()
    if not db:
        return {"attempts": 0, "wait_seconds": 0, "message": ""}

    key = _get_attempt_key(email, ip)
    doc_ref = db.collection(COLLECTION_NAME).document(key)
    doc = doc_ref.get()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    attempts = int((doc.to_dict() or {}).get("attempts", 0) or 0) + 1 if doc.exists else 1
    delay_seconds = DELAY_CONFIG.get(attempts, LOCKOUT_DURATION)
    locked_until = now + timedelta(seconds=delay_seconds) if delay_seconds > 0 else None

    doc_ref.set(
        {
            "email_hash": hashlib.sha256(email.lower().strip().encode()).hexdigest(),
            "attempts": attempts,
            "last_attempt": now,
            "locked_until": locked_until,
        }
    )

    if attempts >= MAX_ATTEMPTS:
        message = f"Account temporarily locked. Try again in {_format_time(delay_seconds)}."
    elif delay_seconds > 0:
        message = f"Too many attempts. Please wait {_format_time(delay_seconds)}."
    else:
        remaining = MAX_ATTEMPTS - attempts
        message = f"Incorrect password. {remaining} attempt{'s' if remaining != 1 else ''} remaining."

    return {"attempts": attempts, "wait_seconds": delay_seconds, "message": message}


def clear_attempts(email: str, ip: str) -> bool:
    db = get_firestore_db()
    if not db:
        return False
    try:
        db.collection(COLLECTION_NAME).document(_get_attempt_key(email, ip)).delete()
        return True
    except Exception as exc:
        print(f"[RateLimiter] Error clearing attempts: {exc}")
        return False
