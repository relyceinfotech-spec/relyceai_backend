from app.agent.graph_scheduler import (
    _build_parallel_merge_snapshot,
    _dedupe_result_items,
    _dedupe_tool_result_data,
)
from app.agent.task_manager import TaskManager
from app.agent.tool_executor import ToolResult
from app.state.plan_graph import PlanGraph, PlanNode


def test_dedupe_result_items_removes_duplicate_urls():
    raw = [
        {"title": "Alpha", "url": "https://example.com/a"},
        {"title": "Alpha Duplicate", "url": "https://example.com/a"},
        {"title": "Beta", "url": "https://example.com/b"},
    ]
    out = _dedupe_result_items(raw)
    assert len(out) == 2
    assert out[0]["url"] == "https://example.com/a"
    assert out[1]["url"] == "https://example.com/b"


def test_dedupe_tool_result_data_handles_common_result_keys():
    data = {
        "results": [
            {"title": "One", "link": "https://a.com"},
            {"title": "One duplicate", "link": "https://a.com"},
            {"title": "Two", "link": "https://b.com"},
        ]
    }
    out = _dedupe_tool_result_data(data)
    assert isinstance(out, dict)
    assert len(out["results"]) == 2


def test_parallel_merge_snapshot_builds_unique_sources():
    node1 = PlanNode(node_id="P1", action_type="TOOL_CALL", payload={"tool": "search_web"})
    node2 = PlanNode(node_id="P2", action_type="TOOL_CALL", payload={"tool": "search_news"})

    r1 = ToolResult(
        tool_name="search_web",
        success=True,
        data={
            "results": [
                {"title": "Shared Source", "link": "https://shared.com"},
                {"title": "Unique One", "link": "https://one.com"},
            ]
        },
    )
    r2 = ToolResult(
        tool_name="search_news",
        success=True,
        data={
            "results": [
                {"title": "Shared Source Again", "link": "https://shared.com"},
                {"title": "Unique Two", "link": "https://two.com"},
            ]
        },
    )

    snap = _build_parallel_merge_snapshot(
        [
            (node1, "search_web", r1),
            (node2, "search_news", r2),
        ]
    )
    assert snap["tool_runs"] == 2
    assert snap["unique_sources"] == 3
    assert len(snap["sources"]) == 3


def test_collect_role_telemetry_includes_parallel_exception_metrics():
    graph = PlanGraph(graph_id="g-par", session_id="s-par")
    graph.metadata["parallel_batches_total"] = 1
    graph.metadata["parallel_tools_total"] = 3
    graph.metadata["parallel_success_total"] = 2
    graph.metadata["parallel_exception_count_total"] = 1
    graph.metadata["role_flow"] = [
        {
            "node_id": "P1",
            "action_type": "TOOL_CALL",
            "role": "researcher",
            "status": "COMPLETED",
            "duration_ms": 90,
            "retries": 0,
            "tool_calls": 1,
            "parallel_exception_count": 1,
            "role_fallback_applied": False,
            "role_resolution_source": "action_mapping",
        }
    ]

    out = TaskManager._collect_role_telemetry_from_graph(graph=graph, mode_name="research_pro")
    assert out["role_summary"]["parallel_exception_count_total"] == 1
    assert out["role_summary"]["parallel_exception_rate"] == 1.0
    assert out["role_metrics"]["researcher"]["parallel_exception_count"] == 1
