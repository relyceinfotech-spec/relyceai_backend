"""
Phase 7: Anomaly Detector
Lightweight threshold-based anomaly detection over telemetry metrics.

Design:
  - No auto-shutdown — detection only
  - Emits ANOMALY_ALERT events to the logger
  - Decoupled from business logic
  - Called periodically or after significant state changes
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.observability.event_types import EventType
from app.observability.event_logger import ExecutionEventLogger
from app.observability.metrics_collector import MetricsCollector


# ============================================
# THRESHOLD CONFIGURATION
# ============================================

DEFAULT_THRESHOLDS = {
    "strategy_drift_rate": 0.2,       # >20% drift is anomalous
    "sandbox_kill_rate": 0.3,         # >30% sandbox kill rate
    "sandbox_kills_absolute": 10,     # >10 total sandbox kills
    "node_failure_rate": 0.25,        # >25% node failure rate
    "repair_exhaustion_rate": 0.5,    # >50% repairs hitting max attempts
    "rollback_spike": 5,             # >5 total rollbacks
}


class AnomalyDetector:
    """
    Evaluates current metrics against thresholds.
    Emits ANOMALY_ALERT events when thresholds are breached.

    Usage:
        detector = AnomalyDetector(metrics, logger)
        alerts = detector.evaluate()
    """

    def __init__(
        self,
        metrics: MetricsCollector,
        logger: ExecutionEventLogger,
        thresholds: Optional[Dict] = None,
    ):
        self.metrics = metrics
        self.logger = logger
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def evaluate(self) -> List[Dict]:
        """
        Run all anomaly checks against current metrics.
        Returns a list of triggered alerts.
        Does NOT modify execution state — detection only.
        """
        m = self.metrics.get_metrics()
        alerts: List[Dict] = []

        # 1. Strategy drift rate
        if m["strategy_drift_rate"] > self.thresholds.get("strategy_drift_rate", 0.2):
            alert = {
                "anomaly": "HIGH_STRATEGY_DRIFT",
                "value": m["strategy_drift_rate"],
                "threshold": self.thresholds["strategy_drift_rate"],
                "severity": "warning",
            }
            alerts.append(alert)
            self.logger.emit(EventType.ANOMALY_ALERT, alert)

        # 2. Sandbox kill rate
        if m["sandbox_kill_rate"] > self.thresholds.get("sandbox_kill_rate", 0.3):
            alert = {
                "anomaly": "HIGH_SANDBOX_KILL_RATE",
                "value": m["sandbox_kill_rate"],
                "threshold": self.thresholds["sandbox_kill_rate"],
                "severity": "critical",
            }
            alerts.append(alert)
            self.logger.emit(EventType.ANOMALY_ALERT, alert)

        # 3. Absolute sandbox kills
        if m["sandbox_kills"] > self.thresholds.get("sandbox_kills_absolute", 10):
            alert = {
                "anomaly": "EXCESSIVE_SANDBOX_KILLS",
                "value": m["sandbox_kills"],
                "threshold": self.thresholds["sandbox_kills_absolute"],
                "severity": "critical",
            }
            alerts.append(alert)
            self.logger.emit(EventType.ANOMALY_ALERT, alert)

        # 4. Node failure rate
        if m["node_failure_rate"] > self.thresholds.get("node_failure_rate", 0.25):
            alert = {
                "anomaly": "HIGH_NODE_FAILURE_RATE",
                "value": m["node_failure_rate"],
                "threshold": self.thresholds["node_failure_rate"],
                "severity": "warning",
            }
            alerts.append(alert)
            self.logger.emit(EventType.ANOMALY_ALERT, alert)

        # 5. Repair exhaustion rate
        if m["total_repairs"] > 0:
            exhaustion_rate = m["total_repairs_exhausted"] / m["total_repairs"]
            if exhaustion_rate > self.thresholds.get("repair_exhaustion_rate", 0.5):
                alert = {
                    "anomaly": "HIGH_REPAIR_EXHAUSTION",
                    "value": round(exhaustion_rate, 4),
                    "threshold": self.thresholds["repair_exhaustion_rate"],
                    "severity": "warning",
                }
                alerts.append(alert)
                self.logger.emit(EventType.ANOMALY_ALERT, alert)

        # 6. Rollback spike
        if m["total_rollbacks"] > self.thresholds.get("rollback_spike", 5):
            alert = {
                "anomaly": "ROLLBACK_SPIKE",
                "value": m["total_rollbacks"],
                "threshold": self.thresholds["rollback_spike"],
                "severity": "warning",
            }
            alerts.append(alert)
            self.logger.emit(EventType.ANOMALY_ALERT, alert)

        return alerts
