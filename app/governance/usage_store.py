"""
Phase 10: Usage Store
In-memory per-user usage state with automatic daily reset.

MVP: In-memory dictionary (swappable to Firestore/Redis later).
Thread-safe. Atomic increments.

Tracks per user:
  - daily_requests
  - daily_tokens
  - concurrent_tasks
  - minute-window request timestamps (for burst control)
  - last_reset date
"""
from __future__ import annotations

import time
import threading
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class UserUsage:
    """Usage record for a single user."""
    user_id: str = ""
    daily_requests: int = 0
    daily_tokens: int = 0
    concurrent_tasks: int = 0
    last_reset: str = ""  # ISO date string
    request_timestamps: List[float] = field(default_factory=list)


class UsageStore:
    """
    Thread-safe in-memory usage store.
    Auto-resets daily counters when date rolls over.

    Usage:
        store = UsageStore()
        usage = store.get_or_create("user123")
        store.increment_requests("user123")
        store.increment_tokens("user123", 500)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._users: Dict[str, UserUsage] = {}

    def get_or_create(self, user_id: str) -> UserUsage:
        """Get user usage, creating if not exists. Auto-resets on new day."""
        with self._lock:
            today = date.today().isoformat()
            if user_id not in self._users:
                self._users[user_id] = UserUsage(user_id=user_id, last_reset=today)
            usage = self._users[user_id]
            # Auto-reset if day has changed
            if usage.last_reset != today:
                usage.daily_requests = 0
                usage.daily_tokens = 0
                usage.request_timestamps = []
                usage.last_reset = today
            return usage

    def increment_requests(self, user_id: str) -> None:
        """Increment daily request count and record timestamp."""
        with self._lock:
            usage = self._get_safe(user_id)
            usage.daily_requests += 1
            usage.request_timestamps.append(time.time())

    def increment_tokens(self, user_id: str, token_count: int) -> None:
        """Add tokens to daily total."""
        with self._lock:
            usage = self._get_safe(user_id)
            usage.daily_tokens += token_count

    def increment_concurrent(self, user_id: str) -> None:
        """Mark a new concurrent task starting."""
        with self._lock:
            usage = self._get_safe(user_id)
            usage.concurrent_tasks += 1

    def decrement_concurrent(self, user_id: str) -> None:
        """Mark a concurrent task finishing. Floor at 0."""
        with self._lock:
            usage = self._get_safe(user_id)
            usage.concurrent_tasks = max(0, usage.concurrent_tasks - 1)

    def get_recent_request_count(self, user_id: str, window_seconds: int = 60) -> int:
        """Count requests within the last N seconds (for burst control)."""
        with self._lock:
            usage = self._get_safe(user_id)
            cutoff = time.time() - window_seconds
            # Prune old timestamps while counting
            recent = [t for t in usage.request_timestamps if t > cutoff]
            usage.request_timestamps = recent
            return len(recent)

    def _get_safe(self, user_id: str) -> UserUsage:
        """Internal: get or create without lock (caller must hold lock)."""
        today = date.today().isoformat()
        if user_id not in self._users:
            self._users[user_id] = UserUsage(user_id=user_id, last_reset=today)
        usage = self._users[user_id]
        if usage.last_reset != today:
            usage.daily_requests = 0
            usage.daily_tokens = 0
            usage.request_timestamps = []
            usage.last_reset = today
        return usage

    def get_system_token_total(self) -> int:
        """Sum of all daily tokens across all users."""
        with self._lock:
            return sum(u.daily_tokens for u in self._users.values())

    def reset_user(self, user_id: str) -> None:
        """Force reset a user's usage (admin/testing)."""
        with self._lock:
            if user_id in self._users:
                del self._users[user_id]


# ============================================
# MODULE SINGLETON
# ============================================

_default_store: Optional[UsageStore] = None


def get_usage_store() -> UsageStore:
    global _default_store
    if _default_store is None:
        _default_store = UsageStore()
    return _default_store
