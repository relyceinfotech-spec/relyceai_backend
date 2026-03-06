"""
Phase 10: Quota Manager
Daily caps for requests and tokens per user.

Design:
  - Hard daily limits
  - Tier-aware (FREE / PRO)
  - Concurrent task guard
  - Deterministic — no AI decisions
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict
from app.governance.usage_store import UsageStore, UserUsage, get_usage_store


# ============================================
# TIER CONFIGURATION
# ============================================

TIER_LIMITS: Dict[str, Dict] = {
    "free": {
        "daily_requests": 50,
        "daily_tokens": 50_000,
        "max_concurrent": 1,
    },
    "pro": {
        "daily_requests": 500,
        "daily_tokens": 500_000,
        "max_concurrent": 3,
    },
    "enterprise": {
        "daily_requests": 5000,
        "daily_tokens": 5_000_000,
        "max_concurrent": 10,
    },
}

DEFAULT_TIER = "free"


class QuotaManager:
    """
    Enforces daily request/token quotas and concurrent task limits.

    Usage:
        quota = QuotaManager()
        allowed, reason = quota.check("user123", tier="free")
    """

    def __init__(self, store: Optional[UsageStore] = None):
        self.store = store or get_usage_store()

    def check(self, user_id: str, tier: str = DEFAULT_TIER) -> Tuple[bool, Optional[str]]:
        """
        Check if user is within all quota limits.
        Returns (allowed, denial_reason).
        """
        limits = TIER_LIMITS.get(tier, TIER_LIMITS[DEFAULT_TIER])
        usage = self.store.get_or_create(user_id)

        # Daily request cap
        if usage.daily_requests >= limits["daily_requests"]:
            return False, "DAILY_REQUEST_LIMIT"

        # Daily token cap
        if usage.daily_tokens >= limits["daily_tokens"]:
            return False, "DAILY_TOKEN_LIMIT"

        # Concurrent task cap
        if usage.concurrent_tasks >= limits["max_concurrent"]:
            return False, "CONCURRENT_LIMIT_REACHED"

        return True, None

    def get_remaining(self, user_id: str, tier: str = DEFAULT_TIER) -> Dict:
        """Return remaining quota for a user."""
        limits = TIER_LIMITS.get(tier, TIER_LIMITS[DEFAULT_TIER])
        usage = self.store.get_or_create(user_id)
        return {
            "requests_remaining": max(0, limits["daily_requests"] - usage.daily_requests),
            "tokens_remaining": max(0, limits["daily_tokens"] - usage.daily_tokens),
            "concurrent_remaining": max(0, limits["max_concurrent"] - usage.concurrent_tasks),
            "tier": tier,
        }


# ============================================
# MODULE SINGLETON
# ============================================

_default_quota: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    global _default_quota
    if _default_quota is None:
        _default_quota = QuotaManager()
    return _default_quota
