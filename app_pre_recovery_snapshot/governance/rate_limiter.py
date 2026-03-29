"""
Phase 10: Rate Limiter
Burst control — prevents spam-level request rates per user.

Design:
  - Sliding window (last 60 seconds)
  - Hard cap per minute
  - No dynamic adjustment — deterministic
  - Thread-safe via usage store
"""
from __future__ import annotations

from typing import Optional, Tuple
from app.governance.usage_store import UsageStore, get_usage_store


# Default: 20 requests per minute
DEFAULT_MAX_REQUESTS_PER_MINUTE = 20


class RateLimiter:
    """
    Burst-level rate limiter using sliding window.

    Usage:
        limiter = RateLimiter()
        allowed = limiter.allow("user123")
        if not allowed:
            return 429
    """

    def __init__(
        self,
        max_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE,
        store: Optional[UsageStore] = None,
    ):
        self.max_per_minute = max_per_minute
        self.store = store or get_usage_store()

    def allow(self, user_id: str) -> bool:
        """
        Check if user is within burst limit.
        Returns True if allowed, False if rate-limited.
        """
        count = self.store.get_recent_request_count(user_id, window_seconds=60)
        if count >= self.max_per_minute:
            return False
        return True

    def check_and_record(self, user_id: str) -> Tuple[bool, Optional[str]]:
        """
        Combined check + record in one call.
        Returns (allowed, reason_if_denied).
        """
        if not self.allow(user_id):
            return False, "RATE_LIMIT_EXCEEDED"
        self.store.increment_requests(user_id)
        return True, None


# ============================================
# MODULE SINGLETON
# ============================================

_default_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter()
    return _default_limiter
