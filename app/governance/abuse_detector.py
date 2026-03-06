"""
Advanced Hardening: Abuse Detector
Behavioral abuse detection — deterministic pattern matching, no AI.

Detection modes:
  1. IP throttling (max requests per IP per minute)
  2. Velocity anomaly (current rate > baseline * 5x)
  3. Repeated failure detection (consecutive failures in window)
  4. Multi-account per IP (same IP, many user IDs)

All detection is LOGGING + FLAGGING only (no auto-ban in MVP).
Feeds into event_logger as ANOMALY_ALERT.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.governance.ip_tracker import IPTracker, get_ip_tracker
from app.governance.usage_store import UsageStore, get_usage_store
from app.observability.event_types import EventType
from app.observability.event_logger import ExecutionEventLogger, get_event_logger


# ============================================
# CONFIGURATION
# ============================================

DEFAULT_ABUSE_THRESHOLDS = {
    "ip_max_per_minute": 60,           # Max requests per IP per minute
    "velocity_multiplier": 5,          # Flag if rate > baseline * 5
    "velocity_baseline_window": 300,   # 5 min window for baseline
    "velocity_current_window": 60,     # 1 min window for current rate
    "failure_threshold": 10,           # Consecutive failures in 5 min
    "failure_window": 300,             # 5 min failure window
    "multi_account_threshold": 3,      # >3 user IDs from same IP
}


@dataclass
class AbuseAlert:
    """Structured abuse detection result."""
    alert_type: str
    ip: str
    user_id: str
    severity: str  # "low", "medium", "high"
    details: Dict
    timestamp: float = field(default_factory=time.time)


class AbuseDetector:
    """
    Deterministic abuse pattern detector.

    Usage:
        detector = AbuseDetector()
        alerts = detector.evaluate_request(ip="1.2.3.4", user_id="u1")
        if alerts:
            log_and_flag(alerts)
    """

    def __init__(
        self,
        ip_tracker: Optional[IPTracker] = None,
        usage_store: Optional[UsageStore] = None,
        event_logger: Optional[ExecutionEventLogger] = None,
        thresholds: Optional[Dict] = None,
    ):
        self.ip_tracker = ip_tracker or get_ip_tracker()
        self.usage_store = usage_store or get_usage_store()
        self.event_logger = event_logger or get_event_logger()
        self.thresholds = thresholds or DEFAULT_ABUSE_THRESHOLDS

    def evaluate_request(self, ip: str, user_id: str = "") -> List[AbuseAlert]:
        """
        Run all abuse checks for an incoming request.
        Returns list of triggered alerts (empty if clean).
        """
        alerts: List[AbuseAlert] = []

        # Record the request
        self.ip_tracker.record_request(ip, user_id=user_id)

        # 1. IP throttle
        alert = self._check_ip_throttle(ip, user_id)
        if alert:
            alerts.append(alert)

        # 2. Velocity anomaly
        alert = self._check_velocity_anomaly(ip, user_id)
        if alert:
            alerts.append(alert)

        # 3. Repeated failures
        alert = self._check_repeated_failures(ip, user_id)
        if alert:
            alerts.append(alert)

        # 4. Multi-account from same IP
        alert = self._check_multi_account(ip, user_id)
        if alert:
            alerts.append(alert)

        # Emit events for all alerts
        for a in alerts:
            self.event_logger.emit(EventType.ANOMALY_ALERT, {
                "abuse_type": a.alert_type,
                "ip": a.ip,
                "user_id": a.user_id,
                "severity": a.severity,
                "details": a.details,
            })

        return alerts

    def record_failure(self, ip: str, user_id: str = "") -> None:
        """Record a failed request for abuse tracking."""
        self.ip_tracker.record_failure(ip)

    def _check_ip_throttle(self, ip: str, user_id: str) -> Optional[AbuseAlert]:
        """Check if IP exceeds per-minute request limit."""
        count = self.ip_tracker.get_request_count(ip, window_seconds=60)
        limit = self.thresholds["ip_max_per_minute"]
        if count > limit:
            self.ip_tracker.flag_ip(ip, "IP_THROTTLE")
            return AbuseAlert(
                alert_type="IP_THROTTLE",
                ip=ip,
                user_id=user_id,
                severity="high",
                details={"count": count, "limit": limit},
            )
        return None

    def _check_velocity_anomaly(self, ip: str, user_id: str) -> Optional[AbuseAlert]:
        """
        Detect velocity spikes using moving average baseline.
        If current_rate > baseline_rate * multiplier → flag.
        """
        baseline_window = self.thresholds["velocity_baseline_window"]
        current_window = self.thresholds["velocity_current_window"]
        multiplier = self.thresholds["velocity_multiplier"]

        baseline_count = self.ip_tracker.get_request_count(ip, window_seconds=baseline_window)
        current_count = self.ip_tracker.get_request_count(ip, window_seconds=current_window)

        # Normalize to per-minute rate
        baseline_rate = (baseline_count / baseline_window) * 60 if baseline_window > 0 else 0
        current_rate = (current_count / current_window) * 60 if current_window > 0 else 0

        # Need minimum baseline to avoid false positives on first requests
        if baseline_rate > 0.5 and current_rate > baseline_rate * multiplier:
            return AbuseAlert(
                alert_type="VELOCITY_SPIKE",
                ip=ip,
                user_id=user_id,
                severity="medium",
                details={
                    "baseline_rate_per_min": round(baseline_rate, 2),
                    "current_rate_per_min": round(current_rate, 2),
                    "multiplier": multiplier,
                },
            )
        return None

    def _check_repeated_failures(self, ip: str, user_id: str) -> Optional[AbuseAlert]:
        """Flag IPs with too many failures in a time window."""
        window = self.thresholds["failure_window"]
        threshold = self.thresholds["failure_threshold"]
        failure_count = self.ip_tracker.get_failure_count(ip, window_seconds=window)

        if failure_count >= threshold:
            self.ip_tracker.flag_ip(ip, "REPEATED_FAILURE")
            return AbuseAlert(
                alert_type="REPEATED_FAILURE",
                ip=ip,
                user_id=user_id,
                severity="medium",
                details={"failure_count": failure_count, "threshold": threshold, "window_seconds": window},
            )
        return None

    def _check_multi_account(self, ip: str, user_id: str) -> Optional[AbuseAlert]:
        """Flag IPs associated with too many user accounts."""
        users = self.ip_tracker.get_associated_users(ip)
        threshold = self.thresholds["multi_account_threshold"]

        if len(users) > threshold:
            self.ip_tracker.flag_ip(ip, "MULTI_ACCOUNT")
            return AbuseAlert(
                alert_type="MULTI_ACCOUNT",
                ip=ip,
                user_id=user_id,
                severity="high",
                details={"user_count": len(users), "threshold": threshold, "user_ids": users[:5]},
            )
        return None


# ============================================
# MODULE SINGLETON
# ============================================

_default_detector: Optional[AbuseDetector] = None


def get_abuse_detector() -> AbuseDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = AbuseDetector()
    return _default_detector
