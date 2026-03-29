"""
Advanced Hardening: Dashboard Data Tests
Verifies in-memory aggregation of task overview, token costs,
system health, and error breakdown.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.dashboard.dashboard_data import DashboardAggregator, get_dashboard_aggregator, MODEL_PRICING
from app.observability.metrics_collector import get_metrics_collector
from app.observability.event_logger import get_event_logger
from app.observability.event_types import EventType
from app.governance.token_tracker import get_token_tracker
from app.health.health_monitor import HEALTHY, get_health_monitor


# Helper to clean up singletons before each test
@pytest.fixture(autouse=True)
def reset_globals():
    get_metrics_collector().reset()
    get_event_logger().clear()
    get_token_tracker().reset()


# ============================================
# TASK OVERVIEW TESTS
# ============================================

def test_task_overview():
    """Verify task counts and average execution duration compute correctly."""
    metrics = get_metrics_collector()
    
    # Simulate some tasks
    metrics.record_task_started()
    metrics.record_task_started()
    metrics.record_task_completed()
    
    # Simulate node execution time (in ms)
    metrics.record_node_execution(latency_ms=2000.0)
    metrics.record_node_execution(latency_ms=4000.0)

    aggregator = get_dashboard_aggregator()
    data = aggregator.get_task_overview()

    assert data["total_tasks_started"] == 2
    assert data["total_tasks_completed"] == 1
    assert data["total_nodes_executed"] == 2
    assert data["average_node_duration_seconds"] == 3.0  # 3000ms / 1000


# ============================================
# TOKEN COST ESTIMATE TESTS
# ============================================

def test_token_dashboard_cost_calculation():
    """Verify costs are computed correctly using model-aware pricing."""
    tracker = get_token_tracker()
    
    # Record exactly 1000 prompt and 2000 completion tokens
    tracker.record("u1", prompt_tokens=1000, completion_tokens=2000)

    aggregator = get_dashboard_aggregator()
    data = aggregator.get_token_dashboard(model_name="gpt-4o")

    # Expected gpt-4o: $0.005 / 1k prompt, $0.015 / 1k comp
    expected_prompt_cost = 0.005 * 1
    expected_comp_cost = 0.015 * 2
    expected_total = expected_prompt_cost + expected_comp_cost

    assert data["tokens"]["prompt"] == 1000
    assert data["tokens"]["completion"] == 2000
    assert data["estimated_cost_usd"]["prompt_cost"] == round(expected_prompt_cost, 4)
    assert data["estimated_cost_usd"]["completion_cost"] == round(expected_comp_cost, 4)
    assert data["estimated_cost_usd"]["total_cost"] == round(expected_total, 4)


def test_token_dashboard_different_models():
    """Different models should produce different cost estimates."""
    tracker = get_token_tracker()
    tracker.record("u1", prompt_tokens=1_000_000, completion_tokens=1_000_000)

    aggregator = get_dashboard_aggregator()

    data_opus = aggregator.get_token_dashboard("claude-3-opus")
    data_haiku = aggregator.get_token_dashboard("claude-3-haiku")

    assert data_opus["estimated_cost_usd"]["total_cost"] > data_haiku["estimated_cost_usd"]["total_cost"]
    assert data_opus["model_used_for_estimate"] == "claude-3-opus"


# ============================================
# ERROR BREAKDOWN TESTS
# ============================================

def test_error_breakdown_top_tools():
    """Top failing tools should be extracted from EventLogger."""
    logger = get_event_logger()
    metrics = get_metrics_collector()

    # Simulate 3 failures for ToolA
    for _ in range(3):
        metrics.record_node_failure()
        logger.emit(EventType.NODE_FAILED, {"node_id": "ToolA", "reason": "timeout"})

    # Simulate 1 failure for ToolB
    metrics.record_node_failure()
    logger.emit(EventType.NODE_FAILED, {"node_id": "ToolB", "reason": "crash"})

    # Simulate unrelated events
    logger.emit(EventType.NODE_STARTED, {"node_id": "ToolC"})

    aggregator = get_dashboard_aggregator()
    data = aggregator.get_error_breakdown()

    assert data["error_metrics"]["total_node_failures"] == 4
    
    top_tools = data["top_failing_tools"]
    assert len(top_tools) == 2
    assert top_tools[0] == {"tool": "ToolA", "failures": 3}
    assert top_tools[1] == {"tool": "ToolB", "failures": 1}


# ============================================
# FULL PAYLOAD INTEGRATION TEST
# ============================================

def test_full_overview():
    """Full overview brings everything together cleanly."""
    aggregator = get_dashboard_aggregator()
    data = aggregator.get_full_overview()

    assert "task_overview" in data
    assert "token_dashboard" in data
    assert "system_health" in data
    assert "error_breakdown" in data
