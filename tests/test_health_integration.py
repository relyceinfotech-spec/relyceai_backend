"""
Phase 8: Health Governance Integration Tests
Verifies circuit breaker, health monitor, mitigation policies,
worker lifecycle, and their interactions.
"""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.health.health_monitor import HealthMonitor, HEALTHY, SANDBOX_UNSTABLE, HIGH_NODE_FAILURE, REPAIR_SYSTEM_STRESSED, ROLLBACK_SPIKE
from app.health.circuit_breaker import CircuitBreaker
from app.health.mitigation_policies import MitigationPolicies
from app.health.worker_lifecycle import WorkerLifecycle
from app.observability.metrics_collector import MetricsCollector
from app.observability.event_logger import ExecutionEventLogger
from app.observability.event_types import EventType


# ============================================
# HEALTH MONITOR TESTS
# ============================================

def test_health_monitor_healthy():
    """Clean metrics should return HEALTHY."""
    metrics = MetricsCollector()
    for _ in range(10):
        metrics.record_node_execution(10.0)
    monitor = HealthMonitor(metrics=metrics)
    assert monitor.evaluate() == HEALTHY


def test_health_monitor_sandbox_unstable():
    """High sandbox kill rate should return SANDBOX_UNSTABLE."""
    metrics = MetricsCollector()
    for _ in range(4):
        metrics.record_sandbox_execution()
    for _ in range(2):
        metrics.record_sandbox_kill()
    # kill rate = 2/4 = 0.5 > 0.25 threshold
    monitor = HealthMonitor(metrics=metrics)
    assert monitor.evaluate() == SANDBOX_UNSTABLE


def test_health_monitor_high_node_failure():
    """High node failure rate should return HIGH_NODE_FAILURE."""
    metrics = MetricsCollector()
    for _ in range(10):
        metrics.record_node_execution(10.0)
    for _ in range(4):
        metrics.record_node_failure()
    # failure rate = 4/10 = 0.4 > 0.35 threshold
    monitor = HealthMonitor(metrics=metrics)
    assert monitor.evaluate() == HIGH_NODE_FAILURE


def test_health_monitor_repair_stressed():
    """High repair exhaustion should return REPAIR_SYSTEM_STRESSED."""
    metrics = MetricsCollector()
    for _ in range(10):
        metrics.record_node_execution(10.0)
    for _ in range(5):
        metrics.record_repair()
    for _ in range(2):
        metrics.record_repair_exhausted()
    # exhaustion rate = 2/5 = 0.4 > 0.2 threshold
    monitor = HealthMonitor(metrics=metrics)
    assert monitor.evaluate() == REPAIR_SYSTEM_STRESSED


def test_health_monitor_rollback_spike():
    """Excessive rollbacks should return ROLLBACK_SPIKE."""
    metrics = MetricsCollector()
    for _ in range(10):
        metrics.record_node_execution(10.0)
    for _ in range(6):
        metrics.record_rollback()
    monitor = HealthMonitor(metrics=metrics)
    assert monitor.evaluate() == ROLLBACK_SPIKE


def test_health_monitor_severity_priority():
    """SANDBOX_UNSTABLE should take priority over ROLLBACK_SPIKE."""
    metrics = MetricsCollector()
    for _ in range(4):
        metrics.record_sandbox_execution()
    for _ in range(2):
        metrics.record_sandbox_kill()
    for _ in range(10):
        metrics.record_rollback()
    monitor = HealthMonitor(metrics=metrics)
    # Sandbox is more severe, should be returned first
    assert monitor.evaluate() == SANDBOX_UNSTABLE


# ============================================
# CIRCUIT BREAKER TESTS
# ============================================

def test_breaker_starts_closed():
    """Breaker should start in CLOSED state."""
    breaker = CircuitBreaker()
    assert breaker.allow_execution() is True
    assert breaker.get_state()["state"] == "CLOSED"


def test_breaker_opens_on_unhealthy():
    """Breaker should OPEN when health signal is not HEALTHY."""
    breaker = CircuitBreaker()
    breaker.evaluate(SANDBOX_UNSTABLE)
    assert breaker.allow_execution() is False
    assert breaker.get_state()["state"] == "OPEN"
    assert breaker.get_state()["open_count"] == 1


def test_breaker_blocks_execution_when_open():
    """Open breaker must block new task execution."""
    breaker = CircuitBreaker()
    breaker.evaluate(HIGH_NODE_FAILURE)
    assert breaker.allow_execution() is False


def test_breaker_cooldown_prevents_flapping():
    """Breaker should NOT close immediately even if signal goes HEALTHY."""
    breaker = CircuitBreaker(cooldown_seconds=10)
    breaker.evaluate(SANDBOX_UNSTABLE)
    assert breaker.allow_execution() is False

    # Signal healthy, but cooldown hasn't elapsed
    breaker.evaluate(HEALTHY)
    assert breaker.allow_execution() is False  # Still open — cooldown active


def test_breaker_closes_after_cooldown():
    """Breaker should close after cooldown period elapses."""
    breaker = CircuitBreaker(cooldown_seconds=0)  # 0s cooldown for test
    breaker.evaluate(SANDBOX_UNSTABLE)
    assert breaker.allow_execution() is False

    # Simulate cooldown completion
    breaker.last_open_time = time.time() - 1  # 1s ago
    breaker.evaluate(HEALTHY)
    assert breaker.allow_execution() is True
    assert breaker.get_state()["state"] == "CLOSED"


def test_breaker_force_open_close():
    """Manual override should work."""
    breaker = CircuitBreaker()
    breaker.force_open()
    assert breaker.allow_execution() is False
    breaker.force_close()
    assert breaker.allow_execution() is True


def test_breaker_refreshes_open_time_on_repeated_unhealthy():
    """If already OPEN and signal still unhealthy, refresh timestamp."""
    breaker = CircuitBreaker(cooldown_seconds=30)
    breaker.evaluate(SANDBOX_UNSTABLE)
    first_time = breaker.last_open_time
    time.sleep(0.01)
    breaker.evaluate(HIGH_NODE_FAILURE)
    second_time = breaker.last_open_time
    assert second_time > first_time  # Timestamp refreshed


# ============================================
# MITIGATION POLICIES TESTS
# ============================================

def test_mitigation_disables_sandbox_on_unstable():
    """SANDBOX_UNSTABLE should disable sandbox."""
    import app.config as config
    original = config.SANDBOX_ENABLED
    config.SANDBOX_ENABLED = True

    try:
        policy = MitigationPolicies()
        actions = policy.apply(SANDBOX_UNSTABLE)
        assert "SANDBOX_DISABLED" in actions
        assert config.SANDBOX_ENABLED is False
    finally:
        config.SANDBOX_ENABLED = original


def test_mitigation_reduces_retries_on_repair_stress():
    """REPAIR_SYSTEM_STRESSED should reduce MAX_RETRIES."""
    import app.agent.tool_executor as te
    original = te.MAX_RETRIES

    try:
        te.MAX_RETRIES = 2
        policy = MitigationPolicies()
        actions = policy.apply(REPAIR_SYSTEM_STRESSED)
        assert "MAX_RETRIES_REDUCED" in actions
        assert te.MAX_RETRIES == 1
    finally:
        te.MAX_RETRIES = original


def test_mitigation_restores_on_healthy():
    """HEALTHY signal should restore original config values."""
    import app.config as config
    import app.agent.tool_executor as te
    original_sandbox = config.SANDBOX_ENABLED
    original_retries = te.MAX_RETRIES

    try:
        config.SANDBOX_ENABLED = True
        te.MAX_RETRIES = 2

        policy = MitigationPolicies()

        # First degrade
        policy.apply(SANDBOX_UNSTABLE)
        assert config.SANDBOX_ENABLED is False

        # Then recover
        actions = policy.apply(HEALTHY)
        assert "SANDBOX_RESTORED" in actions
        assert config.SANDBOX_ENABLED is True
    finally:
        config.SANDBOX_ENABLED = original_sandbox
        te.MAX_RETRIES = original_retries


def test_mitigation_no_action_on_already_healthy():
    """No mitigations should be applied when system is already healthy."""
    policy = MitigationPolicies()
    actions = policy.apply(HEALTHY)
    # No originals stored, so nothing to restore
    assert actions == []


# ============================================
# WORKER LIFECYCLE TESTS
# ============================================

def test_worker_recycle_on_task_count():
    """Worker should signal recycle after max_tasks."""
    lifecycle = WorkerLifecycle(max_tasks=3, max_uptime_seconds=9999)
    lifecycle.record_task()
    lifecycle.record_task()
    assert lifecycle.should_recycle() is False
    lifecycle.record_task()
    assert lifecycle.should_recycle() is True


def test_worker_recycle_on_uptime():
    """Worker should signal recycle after max_uptime."""
    lifecycle = WorkerLifecycle(max_tasks=9999, max_uptime_seconds=0)
    # Uptime is already > 0
    time.sleep(0.01)
    assert lifecycle.should_recycle() is True


def test_worker_state_snapshot():
    """State snapshot should contain required fields."""
    lifecycle = WorkerLifecycle()
    lifecycle.record_task()
    state = lifecycle.get_state()
    assert "tasks_processed" in state
    assert state["tasks_processed"] == 1
    assert "uptime_seconds" in state
    assert "recycle_signaled" in state


def test_worker_reset():
    """Reset should clear counters."""
    lifecycle = WorkerLifecycle(max_tasks=2)
    lifecycle.record_task()
    lifecycle.record_task()
    assert lifecycle.should_recycle() is True
    lifecycle.reset()
    assert lifecycle.should_recycle() is False
    assert lifecycle.get_state()["tasks_processed"] == 0


# ============================================
# END-TO-END INTEGRATION TEST
# ============================================

def test_full_health_pipeline():
    """
    Simulate a full health degradation → breaker open → mitigation → recovery cycle.
    This is the critical integration test.
    """
    import app.config as config
    original_sandbox = config.SANDBOX_ENABLED
    config.SANDBOX_ENABLED = True

    try:
        # Fresh instances
        metrics = MetricsCollector()
        monitor = HealthMonitor(metrics=metrics)
        breaker = CircuitBreaker(cooldown_seconds=0)
        policy = MitigationPolicies()

        # Step 1: System is healthy
        signal = monitor.evaluate()
        assert signal == HEALTHY
        breaker.evaluate(signal)
        assert breaker.allow_execution() is True

        # Step 2: Sandbox starts failing
        for _ in range(4):
            metrics.record_sandbox_execution()
        for _ in range(2):
            metrics.record_sandbox_kill()

        signal = monitor.evaluate()
        assert signal == SANDBOX_UNSTABLE

        # Step 3: Breaker opens
        breaker.evaluate(signal)
        assert breaker.allow_execution() is False

        # Step 4: Mitigation kicks in
        actions = policy.apply(signal)
        assert "SANDBOX_DISABLED" in actions
        assert config.SANDBOX_ENABLED is False

        # Step 5: After cooldown, system recovers
        breaker.last_open_time = time.time() - 1
        metrics.reset()
        signal = monitor.evaluate()
        assert signal == HEALTHY

        breaker.evaluate(signal)
        assert breaker.allow_execution() is True

        # Step 6: Mitigation restores
        actions = policy.apply(signal)
        assert "SANDBOX_RESTORED" in actions
        assert config.SANDBOX_ENABLED is True

    finally:
        config.SANDBOX_ENABLED = original_sandbox
