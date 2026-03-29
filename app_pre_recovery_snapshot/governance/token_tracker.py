"""
Phase 10: Token Tracker
LLM token accounting — records usage ONLY after successful LLM calls.

Design:
  - Only counts on SUCCESS (never on failure/timeout)
  - Feeds into usage_store for per-user tracking
  - Feeds into spend_guard for system-wide tracking
  - Thread-safe
"""
from __future__ import annotations

import threading
from typing import Optional
from app.governance.usage_store import UsageStore, get_usage_store


class TokenTracker:
    """
    Records LLM token consumption per user and system-wide.

    Usage:
        tracker = TokenTracker()
        tracker.record("user123", prompt_tokens=150, completion_tokens=300)
    """

    def __init__(self, store: Optional[UsageStore] = None):
        self._lock = threading.Lock()
        self.store = store or get_usage_store()
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_calls: int = 0

    def record(self, user_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """
        Record token usage for a successful LLM call.
        ONLY call this after a confirmed successful response.
        """
        total = prompt_tokens + completion_tokens
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1
        # Update per-user store
        self.store.increment_tokens(user_id, total)

    def get_totals(self) -> dict:
        """Return system-wide token totals."""
        with self._lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_calls": self.total_calls,
            }

    def reset(self) -> None:
        """Reset all counters (testing)."""
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_calls = 0


# ============================================
# MODULE SINGLETON
# ============================================

_default_tracker: Optional[TokenTracker] = None


def get_token_tracker() -> TokenTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = TokenTracker()
    return _default_tracker
