"""
Phase 7: Observability & Telemetry Engine Tests
Verifies event logging, metrics aggregation, anomaly detection,
and integration with the execution kernel.
"""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.observability.event_types import EventType
from app.observability.event_logger import ExecutionEventLogger
from app.observability.metrics_collector import MetricsCollector
from app.observability.anomaly_detector import AnomalyDetector


# ============================================
# EVENT LOGGER TESTS
# ============================================

def test_event_emit_and_retrieve():
    """Events emitted should be retrievable."""
    logger = ExecutionEventLogger(max_buffer=100)
    logger.emit(EventType.NODE_STARTED, {"node_id": "N1", "task_id": "t1"})
    logger.emit(EventType.NODE_COMPLETED, {"node_id": "N1", "latency_ms": 42.5})

    events = logger.get_recent(limit=10)
    assert len(events) == 2
    assert events[0]["event_type"] == "NODE_STARTED"
    assert events[1]["event_type"] == "NODE_COMPLETED"
    assert events[1]["payload"]["latency_ms"] == 42.5


def test_event_buffer_max_enforced():
    """Circular buffer should not exceed maxlen."""
    logger = ExecutionEventLogger(max_buffer=5)
    for i in range(20):
        logger.emit(EventType.NODE_STARTED, {"i": i})

    assert logger.count() == 5
    events = logger.get_recent(limit=10)
    assert len(events) == 5
    # Should have the LAST 5 events (indices 15-19)
    assert events[0]["payload"]["i"] == 15


def test_event_type_enforcement():
    """Only EventType enum members should be accepted."""
    logger = ExecutionEventLogger()
    try:
        logger.emit("random_string", {})  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_event_filter_by_type():
    """get_by_type should return only matching events."""
    logger = ExecutionEventLogger()
    logger.emit(EventType.NODE_STARTED, {"n": 1})
    logger.emit(EventType.NODE_COMPLETED, {"n": 2})
    logger.emit(EventType.NODE_STARTED, {"n": 3})
    logger.emit(EventType.REPAIR_TRIGGERED, {"n": 4})

    started = logger.get_by_type(EventType.NODE_STARTED)
    assert len(started) == 2
    assert started[0]["payload"]["n"] == 1
    assert started[1]["payload"]["n"] == 3


def test_event_clear():
    """Clear should flush all events."""
    logger = ExecutionEventLogger()
    logger.emit(EventType.TASK_STARTED, {})
    logger.emit(EventType.TASK_COMPLETED, {})
    assert logger.count() == 2
    logger.clear()
    assert logger.count() == 0


# ============================================
# METRICS COLLECTOR TESTS
# ============================================

def test_metrics_node_execution():
    """Node execution should increment count and accumulate latency."""
    metrics = MetricsCollector()
    metrics.record_node_execution(50.0)
    metrics.record_node_execution(100.0)
    metrics.record_node_execution(150.0)

    m = metrics.get_metrics()
    assert m["total_nodes_executed"] == 3
    assert m["avg_node_latency_ms"] == 100.0


def test_metrics_rates():
    """Computed rates should be correct."""
    metrics = MetricsCollector()
    metrics.record_node_execution(10.0)
    metrics.record_node_execution(20.0)
    metrics.record_node_execution(30.0)
    metrics.record_node_execution(40.0)
    metrics.record_node_failure()

    m = metrics.get_metrics()
    assert m["node_failure_rate"] == 0.25  # 1/4


def test_metrics_sandbox_kill_rate():
    """Sandbox kill rate should be computed correctly."""
    metrics = MetricsCollector()
    metrics.record_sandbox_execution()
    metrics.record_sandbox_execution()
    metrics.record_sandbox_execution()
    metrics.record_sandbox_kill()

    m = metrics.get_metrics()
    assert m["sandbox_kill_rate"] == round(1/3, 4)


def test_metrics_zero_division_safe():
    """Empty metrics should return 0 for all rates, not crash."""
    metrics = MetricsCollector()
    m = metrics.get_metrics()
    assert m["avg_node_latency_ms"] == 0.0
    assert m["node_failure_rate"] == 0.0
    assert m["strategy_drift_rate"] == 0.0
    assert m["sandbox_kill_rate"] == 0.0
    assert m["research_cache_rate"] == 0.0


def test_metrics_reset():
    """Reset should zero all counters."""
    metrics = MetricsCollector()
    metrics.record_node_execution(100.0)
    metrics.record_repair()
    metrics.record_rollback()
    metrics.record_sandbox_kill()
    metrics.reset()

    m = metrics.get_metrics()
    assert m["total_nodes_executed"] == 0
    assert m["total_repairs"] == 0
    assert m["total_rollbacks"] == 0
    assert m["sandbox_kills"] == 0


# ============================================
# ANOMALY DETECTOR TESTS
# ============================================

def test_anomaly_strategy_drift():
    """High drift rate should trigger an alert."""
    metrics = MetricsCollector()
    logger = ExecutionEventLogger()

    # 5 nodes, 2 drifts = 0.4 rate > 0.2 threshold
    for _ in range(5):
        metrics.record_node_execution(10.0)
    metrics.record_strategy_drift()
    metrics.record_strategy_drift()

    detector = AnomalyDetector(metrics, logger)
    alerts = detector.evaluate()

    assert len(alerts) >= 1
    assert any(a["anomaly"] == "HIGH_STRATEGY_DRIFT" for a in alerts)

    # Verify event was logged
    anomaly_events = logger.get_by_type(EventType.ANOMALY_ALERT)
    assert len(anomaly_events) >= 1


def test_anomaly_sandbox_kills():
    """Excessive sandbox kills should trigger an alert."""
    metrics = MetricsCollector()
    logger = ExecutionEventLogger()

    for _ in range(15):
        metrics.record_sandbox_execution()
    for _ in range(12):
        metrics.record_sandbox_kill()

    detector = AnomalyDetector(metrics, logger)
    alerts = detector.evaluate()

    anomaly_names = [a["anomaly"] for a in alerts]
    assert "EXCESSIVE_SANDBOX_KILLS" in anomaly_names
    assert "HIGH_SANDBOX_KILL_RATE" in anomaly_names


def test_anomaly_no_false_positives():
    """Healthy metrics should trigger zero alerts."""
    metrics = MetricsCollector()
    logger = ExecutionEventLogger()

    for _ in range(100):
        metrics.record_node_execution(10.0)
    metrics.record_sandbox_execution()

    detector = AnomalyDetector(metrics, logger)
    alerts = detector.evaluate()
    assert len(alerts) == 0


def test_anomaly_rollback_spike():
    """Rollback spike should trigger alert."""
    metrics = MetricsCollector()
    logger = ExecutionEventLogger()

    for _ in range(6):
        metrics.record_rollback()

    detector = AnomalyDetector(metrics, logger)
    alerts = detector.evaluate()

    assert any(a["anomaly"] == "ROLLBACK_SPIKE" for a in alerts)


def test_anomaly_repair_exhaustion():
    """High repair exhaustion rate should trigger alert."""
    metrics = MetricsCollector()
    logger = ExecutionEventLogger()

    for _ in range(4):
        metrics.record_repair()
    for _ in range(3):
        metrics.record_repair_exhausted()

    detector = AnomalyDetector(metrics, logger)
    alerts = detector.evaluate()

    assert any(a["anomaly"] == "HIGH_REPAIR_EXHAUSTION" for a in alerts)
