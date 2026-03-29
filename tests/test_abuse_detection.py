"""
Advanced Hardening: Abuse & Bot Detection Tests
Verifies IP throttling, velocity spike detection, repeated failure
flagging, multi-account detection, and X-Forwarded-For parsing.
"""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.governance.ip_tracker import IPTracker
from app.governance.abuse_detector import AbuseDetector, AbuseAlert, DEFAULT_ABUSE_THRESHOLDS
from app.governance.usage_store import UsageStore
from app.observability.event_logger import ExecutionEventLogger
from app.observability.event_types import EventType


# ============================================
# IP TRACKER TESTS
# ============================================

def test_ip_tracker_record_request():
    """Request should be recorded with timestamp."""
    tracker = IPTracker()
    tracker.record_request("1.2.3.4", user_id="u1")
    assert tracker.get_request_count("1.2.3.4", window_seconds=60) == 1


def test_ip_tracker_sliding_window():
    """Old requests should be pruned from window."""
    tracker = IPTracker()
    record = tracker._get_or_create("1.2.3.4")
    record.request_timestamps = [time.time() - 120]  # 2 min ago
    tracker.record_request("1.2.3.4")
    count = tracker.get_request_count("1.2.3.4", window_seconds=60)
    assert count == 1  # Only the recent one


def test_ip_tracker_user_association():
    """Multiple user IDs should be tracked per IP."""
    tracker = IPTracker()
    tracker.record_request("1.2.3.4", user_id="u1")
    tracker.record_request("1.2.3.4", user_id="u2")
    tracker.record_request("1.2.3.4", user_id="u1")  # Duplicate
    users = tracker.get_associated_users("1.2.3.4")
    assert len(users) == 2
    assert "u1" in users
    assert "u2" in users


def test_ip_tracker_failure_count():
    """Failures should be tracked separately."""
    tracker = IPTracker()
    tracker.record_failure("1.2.3.4")
    tracker.record_failure("1.2.3.4")
    assert tracker.get_failure_count("1.2.3.4", window_seconds=60) == 2


def test_ip_tracker_flagging():
    """Flag should persist."""
    tracker = IPTracker()
    assert tracker.is_flagged("1.2.3.4") is False
    tracker.flag_ip("1.2.3.4", "ABUSE")
    assert tracker.is_flagged("1.2.3.4") is True


def test_ip_tracker_x_forwarded_for():
    """Should extract first IP from X-Forwarded-For."""
    headers = {"x-forwarded-for": "203.0.113.1, 70.41.3.18, 150.172.238.178"}
    ip = IPTracker.extract_client_ip(headers)
    assert ip == "203.0.113.1"


def test_ip_tracker_x_forwarded_for_single():
    """Single IP in X-Forwarded-For."""
    headers = {"x-forwarded-for": "192.168.1.1"}
    ip = IPTracker.extract_client_ip(headers)
    assert ip == "192.168.1.1"


def test_ip_tracker_no_forwarded():
    """Fallback when no X-Forwarded-For."""
    headers = {"x-real-ip": "10.0.0.1"}
    ip = IPTracker.extract_client_ip(headers)
    assert ip == "10.0.0.1"


# ============================================
# ABUSE DETECTOR TESTS
# ============================================

def test_abuse_clean_request():
    """Normal request should trigger zero alerts."""
    tracker = IPTracker()
    store = UsageStore()
    logger = ExecutionEventLogger()
    detector = AbuseDetector(
        ip_tracker=tracker, usage_store=store, event_logger=logger
    )
    alerts = detector.evaluate_request(ip="1.2.3.4", user_id="u1")
    assert len(alerts) == 0


def test_abuse_ip_throttle():
    """IP exceeding per-minute limit should be flagged."""
    tracker = IPTracker()
    store = UsageStore()
    logger = ExecutionEventLogger()
    detector = AbuseDetector(
        ip_tracker=tracker, usage_store=store, event_logger=logger,
        thresholds={**DEFAULT_ABUSE_THRESHOLDS,
                    "ip_max_per_minute": 5,
                    "velocity_multiplier": 5, "velocity_baseline_window": 300,
                    "velocity_current_window": 60, "failure_threshold": 10,
                    "failure_window": 300, "multi_account_threshold": 3},
    )

    # 5 requests = at limit, 6th triggers
    for i in range(6):
        alerts = detector.evaluate_request(ip="1.2.3.4", user_id="u1")

    assert len(alerts) >= 1
    assert any(a.alert_type == "IP_THROTTLE" for a in alerts)
    assert tracker.is_flagged("1.2.3.4")


def test_abuse_repeated_failures():
    """IP with many failures should be flagged."""
    tracker = IPTracker()
    store = UsageStore()
    logger = ExecutionEventLogger()
    detector = AbuseDetector(
        ip_tracker=tracker, usage_store=store, event_logger=logger,
        thresholds={
            "ip_max_per_minute": 100,
            "velocity_multiplier": 5, "velocity_baseline_window": 300,
            "velocity_current_window": 60,
            "failure_threshold": 3, "failure_window": 300,
            "multi_account_threshold": 10,
        },
    )

    # Record failures
    for _ in range(3):
        detector.record_failure("1.2.3.4")

    alerts = detector.evaluate_request(ip="1.2.3.4", user_id="u1")
    assert any(a.alert_type == "REPEATED_FAILURE" for a in alerts)


def test_abuse_multi_account():
    """Same IP with many accounts should be flagged."""
    tracker = IPTracker()
    store = UsageStore()
    logger = ExecutionEventLogger()
    detector = AbuseDetector(
        ip_tracker=tracker, usage_store=store, event_logger=logger,
        thresholds={
            "ip_max_per_minute": 100,
            "velocity_multiplier": 5, "velocity_baseline_window": 300,
            "velocity_current_window": 60, "failure_threshold": 100,
            "failure_window": 300,
            "multi_account_threshold": 2,
        },
    )

    detector.evaluate_request(ip="1.2.3.4", user_id="u1")
    detector.evaluate_request(ip="1.2.3.4", user_id="u2")
    alerts = detector.evaluate_request(ip="1.2.3.4", user_id="u3")

    assert any(a.alert_type == "MULTI_ACCOUNT" for a in alerts)


def test_abuse_emits_events():
    """Abuse alerts should be emitted as ANOMALY_ALERT events."""
    tracker = IPTracker()
    store = UsageStore()
    logger = ExecutionEventLogger()
    detector = AbuseDetector(
        ip_tracker=tracker, usage_store=store, event_logger=logger,
        thresholds={
            "ip_max_per_minute": 2,
            "velocity_multiplier": 5, "velocity_baseline_window": 300,
            "velocity_current_window": 60, "failure_threshold": 100,
            "failure_window": 300, "multi_account_threshold": 10,
        },
    )

    for _ in range(3):
        detector.evaluate_request(ip="10.0.0.1", user_id="u1")

    events = logger.get_by_type(EventType.ANOMALY_ALERT)
    assert len(events) >= 1
    assert any("abuse_type" in e.get("payload", {}) for e in events)


def test_abuse_no_false_positive_normal_usage():
    """Normal diverse usage should not trigger any alerts."""
    tracker = IPTracker()
    store = UsageStore()
    logger = ExecutionEventLogger()
    detector = AbuseDetector(
        ip_tracker=tracker, usage_store=store, event_logger=logger,
    )

    # 5 requests from different IPs
    for i in range(5):
        alerts = detector.evaluate_request(ip=f"192.168.1.{i}", user_id=f"u{i}")
        assert len(alerts) == 0
