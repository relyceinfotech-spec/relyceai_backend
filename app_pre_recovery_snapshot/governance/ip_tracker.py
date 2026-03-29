"""
Advanced Hardening: IP Tracker
Per-IP request counter with sliding window for abuse detection.

Same architecture as usage_store.py but keyed on IP address.
Thread-safe. Handles X-Forwarded-For parsing.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class IPRecord:
    """Track state for a single IP address."""
    ip: str = ""
    request_timestamps: List[float] = field(default_factory=list)
    failure_timestamps: List[float] = field(default_factory=list)
    associated_user_ids: List[str] = field(default_factory=list)
    flagged: bool = False
    flag_reason: str = ""


class IPTracker:
    """
    Thread-safe per-IP tracking with sliding window.

    Usage:
        tracker = IPTracker()
        tracker.record_request("1.2.3.4", user_id="u1")
        if tracker.get_request_count("1.2.3.4", window=60) > 100:
            block()
    """

    def __init__(self, max_per_minute: int = 60):
        self._lock = threading.Lock()
        self._ips: Dict[str, IPRecord] = {}
        self.max_per_minute = max_per_minute

    def record_request(self, ip: str, user_id: str = "") -> None:
        """Record a request from an IP."""
        with self._lock:
            record = self._get_or_create(ip)
            record.request_timestamps.append(time.time())
            if user_id and user_id not in record.associated_user_ids:
                record.associated_user_ids.append(user_id)

    def record_failure(self, ip: str) -> None:
        """Record a failed request from an IP."""
        with self._lock:
            record = self._get_or_create(ip)
            record.failure_timestamps.append(time.time())

    def get_request_count(self, ip: str, window_seconds: int = 60) -> int:
        """Count requests within sliding window."""
        with self._lock:
            record = self._get_or_create(ip)
            cutoff = time.time() - window_seconds
            recent = [t for t in record.request_timestamps if t > cutoff]
            record.request_timestamps = recent
            return len(recent)

    def get_failure_count(self, ip: str, window_seconds: int = 300) -> int:
        """Count failures within sliding window (default 5 min)."""
        with self._lock:
            record = self._get_or_create(ip)
            cutoff = time.time() - window_seconds
            recent = [t for t in record.failure_timestamps if t > cutoff]
            record.failure_timestamps = recent
            return len(recent)

    def get_associated_users(self, ip: str) -> List[str]:
        """Get all user IDs that have used this IP."""
        with self._lock:
            record = self._get_or_create(ip)
            return list(record.associated_user_ids)

    def flag_ip(self, ip: str, reason: str) -> None:
        """Flag an IP for abuse."""
        with self._lock:
            record = self._get_or_create(ip)
            record.flagged = True
            record.flag_reason = reason

    def is_flagged(self, ip: str) -> bool:
        """Check if an IP is flagged."""
        with self._lock:
            if ip in self._ips:
                return self._ips[ip].flagged
            return False

    def _get_or_create(self, ip: str) -> IPRecord:
        """Internal: get or create IP record (caller must hold lock)."""
        if ip not in self._ips:
            self._ips[ip] = IPRecord(ip=ip)
        return self._ips[ip]

    @staticmethod
    def extract_client_ip(headers: dict) -> str:
        """
        Safely extract client IP from request headers.
        Handles X-Forwarded-For (takes first IP = original client).
        """
        forwarded = headers.get("x-forwarded-for", "")
        if forwarded:
            # First IP in chain is the original client
            return forwarded.split(",")[0].strip()
        return headers.get("x-real-ip", headers.get("remote-addr", "unknown"))


# ============================================
# MODULE SINGLETON
# ============================================

_default_tracker: Optional[IPTracker] = None


def get_ip_tracker() -> IPTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = IPTracker()
    return _default_tracker
