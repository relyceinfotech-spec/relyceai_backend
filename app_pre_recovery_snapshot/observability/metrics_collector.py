"""
Phase 7: Metrics Collector
Rolling counters and aggregation for execution kernel telemetry.

Strict separation:
  - EventLogger stores raw events
  - MetricsCollector aggregates operational counts
  - These never cross-reference at write time (decoupled)
"""
from __future__ import annotations

import threading
from typing import Dict, Any, Optional


class MetricsCollector:
    """
    Thread-safe rolling counter system for execution metrics.

    Usage:
        metrics = MetricsCollector()
        metrics.record_node_execution(latency_ms=42.5)
        metrics.record_repair()
        snapshot = metrics.get_metrics()
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Task-level counters
        self.total_tasks: int = 0
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0

        # Node-level counters
        self.total_nodes_executed: int = 0
        self.total_node_failures: int = 0
        self.total_node_latency_ms: float = 0.0

        # Repair counters
        self.total_repairs: int = 0
        self.total_repairs_exhausted: int = 0

        # Rollback counters
        self.total_rollbacks: int = 0

        # Sandbox counters
        self.sandbox_executions: int = 0
        self.sandbox_kills: int = 0
        self.sandbox_crashes: int = 0

        # Strategy counters
        self.strategy_drift_count: int = 0

        # Research counters
        self.research_calls: int = 0
        self.research_cache_hits: int = 0

    # ---- Recording Methods ----

    def record_task_started(self) -> None:
        with self._lock:
            self.total_tasks += 1

    def record_task_completed(self) -> None:
        with self._lock:
            self.completed_tasks += 1

    def record_task_failed(self) -> None:
        with self._lock:
            self.failed_tasks += 1

    def record_node_execution(self, latency_ms: float) -> None:
        with self._lock:
            self.total_nodes_executed += 1
            self.total_node_latency_ms += latency_ms

    def record_node_failure(self) -> None:
        with self._lock:
            self.total_node_failures += 1

    def record_repair(self) -> None:
        with self._lock:
            self.total_repairs += 1

    def record_repair_exhausted(self) -> None:
        with self._lock:
            self.total_repairs_exhausted += 1

    def record_rollback(self) -> None:
        with self._lock:
            self.total_rollbacks += 1

    def record_sandbox_execution(self) -> None:
        with self._lock:
            self.sandbox_executions += 1

    def record_sandbox_kill(self) -> None:
        with self._lock:
            self.sandbox_kills += 1

    def record_sandbox_crash(self) -> None:
        with self._lock:
            self.sandbox_crashes += 1

    def record_strategy_drift(self) -> None:
        with self._lock:
            self.strategy_drift_count += 1

    def record_research_call(self) -> None:
        with self._lock:
            self.research_calls += 1

    def record_research_cache_hit(self) -> None:
        with self._lock:
            self.research_cache_hits += 1

    # ---- Aggregation ----

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of all aggregated metrics."""
        with self._lock:
            avg_latency = (
                self.total_node_latency_ms / self.total_nodes_executed
                if self.total_nodes_executed > 0 else 0.0
            )

            node_failure_rate = (
                self.total_node_failures / self.total_nodes_executed
                if self.total_nodes_executed > 0 else 0.0
            )

            strategy_drift_rate = (
                self.strategy_drift_count / self.total_nodes_executed
                if self.total_nodes_executed > 0 else 0.0
            )

            sandbox_kill_rate = (
                self.sandbox_kills / self.sandbox_executions
                if self.sandbox_executions > 0 else 0.0
            )

            research_cache_rate = (
                self.research_cache_hits / self.research_calls
                if self.research_calls > 0 else 0.0
            )

            return {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
                "total_nodes_executed": self.total_nodes_executed,
                "total_node_failures": self.total_node_failures,
                "node_failure_rate": round(node_failure_rate, 4),
                "avg_node_latency_ms": round(avg_latency, 2),
                "total_repairs": self.total_repairs,
                "total_repairs_exhausted": self.total_repairs_exhausted,
                "total_rollbacks": self.total_rollbacks,
                "sandbox_executions": self.sandbox_executions,
                "sandbox_kills": self.sandbox_kills,
                "sandbox_crashes": self.sandbox_crashes,
                "sandbox_kill_rate": round(sandbox_kill_rate, 4),
                "strategy_drift_rate": round(strategy_drift_rate, 4),
                "research_calls": self.research_calls,
                "research_cache_hits": self.research_cache_hits,
                "research_cache_rate": round(research_cache_rate, 4),
            }

    def reset(self) -> None:
        """Reset all counters (for testing)."""
        with self._lock:
            self.total_tasks = 0
            self.completed_tasks = 0
            self.failed_tasks = 0
            self.total_nodes_executed = 0
            self.total_node_failures = 0
            self.total_node_latency_ms = 0.0
            self.total_repairs = 0
            self.total_repairs_exhausted = 0
            self.total_rollbacks = 0
            self.sandbox_executions = 0
            self.sandbox_kills = 0
            self.sandbox_crashes = 0
            self.strategy_drift_count = 0
            self.research_calls = 0
            self.research_cache_hits = 0


# ============================================
# MODULE SINGLETON
# ============================================

_default_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Returns the module-level singleton collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector
