"""
System Health Dashboard — Unified API endpoint for all operational metrics.
Aggregates: queue, streaming, cost, memory, and pipeline health.

Endpoint: GET /api/admin/health-dashboard
"""
from typing import Dict


def get_system_dashboard() -> Dict:
    """
    Assemble complete system health dashboard payload.
    All data from in-memory singletons — zero database queries.
    """
    result = {
        "status": "healthy",
        "queue": _safe_get_queue(),
        "streaming": _safe_get_streaming(),
        "cost": _safe_get_cost(),
        "pipeline": _safe_get_pipeline(),
    }

    # Determine overall health status
    queue = result["queue"]
    if queue.get("active_requests", 0) >= queue.get("max_active", 20) * 0.9:
        result["status"] = "degraded"
    if queue.get("total_rejected", 0) > 0:
        result["status"] = "degraded"

    cost = result["cost"]
    if cost.get("cost_efficiency_score", 0) > 5.0:
        result["status"] = "warning"

    return result


def _safe_get_queue() -> dict:
    try:
        from app.gateway.request_queue import get_queue_stats
        return get_queue_stats()
    except Exception:
        return {"error": "unavailable"}


def _safe_get_streaming() -> dict:
    try:
        from app.gateway.stream_metrics import get_stream_metrics
        return get_stream_metrics().get_metrics()
    except Exception:
        return {"error": "unavailable"}


def _safe_get_cost() -> dict:
    try:
        from app.gateway.cost_tracker import get_cost_tracker
        return get_cost_tracker().get_metrics()
    except Exception:
        return {"error": "unavailable"}


def _safe_get_pipeline() -> dict:
    """Pipeline-level metrics from existing observability."""
    try:
        from app.observability.metrics_collector import get_metrics_collector
        m = get_metrics_collector().get_metrics()
        return {
            "total_tasks": m.get("total_tasks", 0),
            "completed_tasks": m.get("completed_tasks", 0),
            "failed_tasks": m.get("failed_tasks", 0),
            "node_failure_rate": m.get("node_failure_rate", 0.0),
            "avg_node_latency_ms": m.get("avg_node_latency_ms", 0.0),
            "sandbox_kill_rate": m.get("sandbox_kill_rate", 0.0),
        }
    except Exception:
        return {"error": "unavailable"}
