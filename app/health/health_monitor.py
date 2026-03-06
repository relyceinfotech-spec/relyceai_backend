"""
Phase 8: Health Monitor
Central evaluator that classifies system health from telemetry metrics.

Design rules:
  - Hard thresholds only — no dynamic learning
  - Returns a deterministic health signal string
  - Called at request boundary (main.py) or periodic background
  - Never modifies execution state directly
"""
from __future__ import annotations

from typing import Optional
from app.observability.metrics_collector import MetricsCollector, get_metrics_collector


# ============================================
# HEALTH SIGNALS
# ============================================

HEALTHY = "HEALTHY"
SANDBOX_UNSTABLE = "SANDBOX_UNSTABLE"
HIGH_NODE_FAILURE = "HIGH_NODE_FAILURE"
REPAIR_SYSTEM_STRESSED = "REPAIR_SYSTEM_STRESSED"
ROLLBACK_SPIKE = "ROLLBACK_SPIKE"


# ============================================
# THRESHOLD CONFIGURATION
# ============================================

DEFAULT_HEALTH_THRESHOLDS = {
    "sandbox_kill_rate": 0.25,
    "node_failure_rate": 0.35,
    "repair_exhaustion_rate": 0.20,
    "rollback_spike": 5,
}


class HealthMonitor:
    """
    Evaluates system health from MetricsCollector snapshot.
    Returns the MOST SEVERE signal found.

    Usage:
        monitor = HealthMonitor()
        signal = monitor.evaluate()  # → "HEALTHY" | "SANDBOX_UNSTABLE" | ...
    """

    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        thresholds: Optional[dict] = None,
    ):
        self.metrics = metrics or get_metrics_collector()
        self.thresholds = thresholds or DEFAULT_HEALTH_THRESHOLDS

    def evaluate(self) -> str:
        """
        Return the most severe health signal.
        Priority order (most severe first):
          1. SANDBOX_UNSTABLE
          2. HIGH_NODE_FAILURE
          3. REPAIR_SYSTEM_STRESSED
          4. ROLLBACK_SPIKE
          5. HEALTHY
        """
        m = self.metrics.get_metrics()

        # 1. Sandbox instability (most critical — process isolation failing)
        if m["sandbox_kill_rate"] > self.thresholds.get("sandbox_kill_rate", 0.25):
            return SANDBOX_UNSTABLE

        # 2. High node failure rate
        if m["node_failure_rate"] > self.thresholds.get("node_failure_rate", 0.35):
            return HIGH_NODE_FAILURE

        # 3. Repair system stressed
        if m["total_repairs"] > 0:
            exhaustion_rate = m["total_repairs_exhausted"] / m["total_repairs"]
            if exhaustion_rate > self.thresholds.get("repair_exhaustion_rate", 0.20):
                return REPAIR_SYSTEM_STRESSED

        # 4. Rollback spike
        if m["total_rollbacks"] > self.thresholds.get("rollback_spike", 5):
            return ROLLBACK_SPIKE

        return HEALTHY


# ============================================
# MODULE SINGLETON
# ============================================

_default_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Returns the module-level singleton monitor."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = HealthMonitor()
    return _default_monitor
