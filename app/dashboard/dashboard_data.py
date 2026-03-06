"""
Advanced Hardening: UX Product Layer — Dashboard Data
Assembles dashboard payloads purely from in-memory metrics and events.
No database queries — fast, cheap, and real-time.

Features:
  - Task overview (counts, averages)
  - Token cost accounting (model-aware pricing)
  - Health & Governance status
  - Error breakdown (top failing tools)
"""
from __future__ import annotations

from typing import Dict, Optional, List
from collections import Counter

from app.observability.metrics_collector import get_metrics_collector
from app.observability.event_logger import get_event_logger
from app.observability.event_types import EventType
from app.health.health_monitor import get_health_monitor
from app.health.circuit_breaker import get_circuit_breaker
from app.health.mitigation_policies import get_mitigation_policies
from app.governance.token_tracker import get_token_tracker
from app.governance.spend_guard import get_spend_guard


# ============================================
# COST CONFIGURATION (Model-Aware)
# ============================================
# Price per 1K tokens in USD (example rates)

MODEL_PRICING = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4-turbo": {"prompt": 0.010, "completion": 0.030},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
}

DEFAULT_MODEL = "gpt-4o"


class DashboardAggregator:
    """
    Computes real-time dashboard payloads from in-memory telemetry.
    """

    @staticmethod
    def get_task_overview() -> Dict:
        """Overview of task execution and performance."""
        m = get_metrics_collector().get_metrics()
        
        # We track node latency, not full task duration in metrics right now.
        # Average node duration in seconds:
        avg_node_duration = round(m.get("avg_node_latency_ms", 0.0) / 1000.0, 2)

        return {
            "total_tasks_started": m.get("total_tasks", 0),
            "total_tasks_completed": m.get("completed_tasks", 0),
            "total_tasks_failed": m.get("failed_tasks", 0),
            "total_nodes_executed": m.get("total_nodes_executed", 0),
            "average_node_duration_seconds": avg_node_duration,
            "system_uptime_hours": 0.0,  # Placeholder, uptime not in metrics currently
        }

    @staticmethod
    def get_token_dashboard(model_name: str = DEFAULT_MODEL) -> Dict:
        """Token usage and model-aware cost estimation."""
        tracker = get_token_tracker()
        totals = tracker.get_totals()
        
        prompt_tokens = totals["total_prompt_tokens"]
        comp_tokens = totals["total_completion_tokens"]
        
        pricing = MODEL_PRICING.get(model_name, MODEL_PRICING[DEFAULT_MODEL])
        
        # Calculate cost based on per-1k pricing
        prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
        comp_cost = (comp_tokens / 1000) * pricing["completion"]
        total_cost = prompt_cost + comp_cost

        spend_guard = get_spend_guard().get_status()

        return {
            "model_used_for_estimate": model_name,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": comp_tokens,
                "total": totals["total_tokens"],
                "total_llm_calls": totals["total_calls"],
            },
            "estimated_cost_usd": {
                "prompt_cost": round(prompt_cost, 4),
                "completion_cost": round(comp_cost, 4),
                "total_cost": round(total_cost, 4),
            },
            "system_budget": {
                "budget_exhausted": spend_guard["budget_exhausted"],
                "utilization_pct": spend_guard["utilization_pct"],
                "remaining_tokens": spend_guard["remaining"],
            }
        }

    @staticmethod
    def get_system_health() -> Dict:
        """System health, circuit breaker, and active mitigations."""
        monitor = get_health_monitor()
        breaker = get_circuit_breaker()
        mitigations = get_mitigation_policies()
        
        signal = monitor.evaluate()
        
        return {
            "health_signal": signal,
            "circuit_state": breaker.get_state()["state"],
            "circuit_open_count": breaker.get_state()["open_count"],
            "active_mitigations": mitigations.get_active(),
            "sandbox_kill_rate": get_metrics_collector().get_metrics().get("sandbox_kill_rate", 0.0),
        }

    @staticmethod
    def get_error_breakdown() -> Dict:
        """Error classification and top failing tools."""
        m = get_metrics_collector().get_metrics()
        logger = get_event_logger()
        
        # Find top failing tools from recent NODE_FAILED events
        failed_events = logger.get_by_type(EventType.NODE_FAILED)
        tool_counter = Counter()
        for e in failed_events:
            # Safely extract node_id (which usually contains the tool name)
            payload = e.get("payload", {})
            node_id = payload.get("node_id", "unknown_tool")
            tool_counter[node_id] += 1
            
        top_failing = [{"tool": k, "failures": v} for k, v in tool_counter.most_common(5)]

        return {
            "error_metrics": {
                "total_node_failures": m.get("total_node_failures", 0),
                "total_repairs": m.get("total_repairs", 0),
                "repairs_exhausted": m.get("total_repairs_exhausted", 0),
                "sandbox_crashes": m.get("total_sandbox_crashes", 0),
                "sandbox_kills": m.get("total_sandbox_kills", 0),
            },
            "rates": {
                "node_failure_rate": m.get("node_failure_rate", 0.0),
                "sandbox_kill_rate": m.get("sandbox_kill_rate", 0.0),
                "repair_exhaustion_rate": round(
                    m.get("total_repairs_exhausted", 0) / max(1, m.get("total_repairs", 1)), 2
                ),
            },
            "top_failing_tools": top_failing,
        }

    @classmethod
    def get_full_overview(cls, model_name: str = DEFAULT_MODEL) -> Dict:
        """Assembles the complete dashboard payload."""
        return {
            "task_overview": cls.get_task_overview(),
            "token_dashboard": cls.get_token_dashboard(model_name),
            "system_health": cls.get_system_health(),
            "error_breakdown": cls.get_error_breakdown(),
        }


# ============================================
# MODULE INSTANCE
# ============================================

def get_dashboard_aggregator() -> DashboardAggregator:
    return DashboardAggregator()
