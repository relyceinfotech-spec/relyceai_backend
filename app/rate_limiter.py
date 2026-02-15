"""
Relyce AI - Login Rate Limiter
Firestore-based rate limiting with progressive delays
Layer 2 Security: Server-side brute-force protection
"""
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Tuple
from app.auth import get_firestore_db

# Progressive delay configuration (in seconds)
DELAY_CONFIG = {
    1: 0,      # 1st-2nd attempt: no delay
    2: 0,
    3: 30,     # 3rd attempt: 30 seconds
    4: 120,    # 4th attempt: 2 minutes
    5: 600,    # 5th+ attempt: 10 minutes
}
MAX_ATTEMPTS = 5
LOCKOUT_DURATION = 600  # 10 minutes for 5+ attempts
EXPIRY_MINUTES = 30     # Auto-expire after 30 min of inactivity

COLLECTION_NAME = "login_attempts"


def _get_attempt_key(email: str, ip: str) -> str:
    """Generate a unique key for email+IP combination using SHA-256"""
    combined = f"{email.lower().strip()}:{ip}"
    return hashlib.sha256(combined.encode()).hexdigest()


def check_rate_limit(email: str, ip: str) -> dict:
    """
    Check if login attempt is allowed for this email+IP.
    
    Returns:
        {
            "allowed": bool,
            "wait_seconds": int (0 if allowed),
            "attempts": int,
            "message": str
        }
    """
    db = get_firestore_db()
    if not db:
        # If Firestore unavailable, allow (fail-open for UX)
        return {"allowed": True, "wait_seconds": 0, "attempts": 0, "message": ""}
    
    key = _get_attempt_key(email, ip)
    doc_ref = db.collection(COLLECTION_NAME).document(key)
    doc = doc_ref.get()
    
    if not doc.exists:
        return {"allowed": True, "wait_seconds": 0, "attempts": 0, "message": ""}
    
    data = doc.to_dict()
    attempts = data.get("attempts", 0)
    last_attempt = data.get("last_attempt")
    locked_until = data.get("locked_until")
    
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # Check if record expired (30 min inactivity)
    if last_attempt:
        last_time = last_attempt if isinstance(last_attempt, datetime) else last_attempt.replace(tzinfo=None)
        if (now - last_time) > timedelta(minutes=EXPIRY_MINUTES):
            # Expired, clear and allow
            doc_ref.delete()
            return {"allowed": True, "wait_seconds": 0, "attempts": 0, "message": ""}
    
    # Check if currently locked
    if locked_until:
        lock_time = locked_until if isinstance(locked_until, datetime) else locked_until.replace(tzinfo=None)
        if now < lock_time:
            wait_seconds = int((lock_time - now).total_seconds())
            return {
                "allowed": False,
                "wait_seconds": wait_seconds,
                "attempts": attempts,
                "message": f"Too many failed attempts. Try again in {_format_time(wait_seconds)}."
            }
    
    return {"allowed": True, "wait_seconds": 0, "attempts": attempts, "message": ""}


def record_failed_attempt(email: str, ip: str) -> dict:
    """
    Record a failed login attempt and apply progressive delays.
    
    Returns:
        {
            "attempts": int,
            "wait_seconds": int,
            "message": str
        }
    """
    db = get_firestore_db()
    if not db:
        return {"attempts": 0, "wait_seconds": 0, "message": ""}
    
    key = _get_attempt_key(email, ip)
    doc_ref = db.collection(COLLECTION_NAME).document(key)
    doc = doc_ref.get()
    
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    if doc.exists:
        data = doc.to_dict()
        attempts = data.get("attempts", 0) + 1
    else:
        attempts = 1
    
    # Determine delay based on attempt count
    delay_seconds = DELAY_CONFIG.get(attempts, LOCKOUT_DURATION)
    locked_until = now + timedelta(seconds=delay_seconds) if delay_seconds > 0 else None
    
    # Update Firestore
    doc_ref.set({
        "email_hash": hashlib.sha256(email.lower().strip().encode()).hexdigest(),
        "attempts": attempts,
        "last_attempt": now,
        "locked_until": locked_until
    })
    
    # Build response message
    if attempts >= MAX_ATTEMPTS:
        message = f"Account temporarily locked. Try again in {_format_time(delay_seconds)}."
    elif delay_seconds > 0:
        message = f"Too many attempts. Please wait {_format_time(delay_seconds)}."
    else:
        remaining = MAX_ATTEMPTS - attempts
        message = f"Incorrect password. {remaining} attempt{'s' if remaining != 1 else ''} remaining."
    
    return {
        "attempts": attempts,
        "wait_seconds": delay_seconds,
        "message": message
    }


def clear_attempts(email: str, ip: str) -> bool:
    """
    Clear login attempts on successful login.
    
    Returns:
        True if cleared successfully
    """
    db = get_firestore_db()
    if not db:
        return False
    
    key = _get_attempt_key(email, ip)
    doc_ref = db.collection(COLLECTION_NAME).document(key)
    
    try:
        doc_ref.delete()
        return True
    except Exception as e:
        print(f"[RateLimiter] Error clearing attempts: {e}")
        return False


def _format_time(seconds: int) -> str:
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if remaining_seconds == 0:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{minutes}:{remaining_seconds:02d}"
