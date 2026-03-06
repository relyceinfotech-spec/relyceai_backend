"""
Phase 10: Spend Guard
Global system-wide budget protection kill switch.

Design:
  - Hard daily cap on total system token consumption
  - Returns 503 when exceeded — no exceptions
  - Checked BEFORE any LLM call
  - This is your financial kill switch
"""
from __future__ import annotations

from typing import Optional
from app.governance.usage_store import UsageStore, get_usage_store


# Global system-level daily token cap
DEFAULT_MAX_DAILY_SYSTEM_TOKENS = 10_000_000


class SpendGuard:
    """
    System-wide spend protection.

    Usage:
        guard = SpendGuard()
        if not guard.allow():
            return 503
    """

    def __init__(
        self,
        max_daily_tokens: int = DEFAULT_MAX_DAILY_SYSTEM_TOKENS,
        store: Optional[UsageStore] = None,
    ):
        self.max_daily_tokens = max_daily_tokens
        self.store = store or get_usage_store()

    def allow(self) -> bool:
        """
        Check if system-wide daily token budget is within limits.
        Returns True if more LLM calls are allowed.
        """
        total = self.store.get_system_token_total()
        return total < self.max_daily_tokens

    def get_status(self) -> dict:
        """Return current spend status."""
        total = self.store.get_system_token_total()
        return {
            "total_tokens_today": total,
            "max_daily_tokens": self.max_daily_tokens,
            "remaining": max(0, self.max_daily_tokens - total),
            "utilization_pct": round((total / self.max_daily_tokens) * 100, 2) if self.max_daily_tokens > 0 else 0,
            "budget_exhausted": total >= self.max_daily_tokens,
        }


# ============================================
# MODULE SINGLETON
# ============================================

_default_guard: Optional[SpendGuard] = None


def get_spend_guard() -> SpendGuard:
    global _default_guard
    if _default_guard is None:
        _default_guard = SpendGuard()
    return _default_guard
