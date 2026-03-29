from app.agent.plan_validator import validate_plan
from app.agent.role_cognition import resolve_plan_node_role
from app.agent.task_manager import TaskManager
from app.state.plan_graph import PlanGraph, PlanNode


def test_simple_role_mapping_final_is_synthesizer():
    out = resolve_plan_node_role(node_id="FINAL", action_type="TOOL_CALL", declared_role="researcher")
    assert out.role == "synthesizer"
    assert out.role_resolution_source == "final_lock"
    assert out.role_fallback_applied is False


def test_simple_role_mapping_action_types():
    assert resolve_plan_node_role(node_id="P1", action_type="TOOL_CALL").role == "researcher"
    assert resolve_plan_node_role(node_id="P2", action_type="REASONING").role == "planner"
    assert resolve_plan_node_role(node_id="P3", action_type="VALIDATION").role == "critic"
    assert resolve_plan_node_role(node_id="P4", action_type="REPAIR").role == "critic"
    assert resolve_plan_node_role(node_id="P5", action_type="EXECUTION").role == "executor"


def test_simple_unknown_declared_role_fallback():
    out = resolve_plan_node_role(node_id="P1", action_type="REASONING", declared_role="unknown_role")
    assert out.role == "executor"
    assert out.role_fallback_applied is True


def test_simple_validator_rejects_non_final_synthesizer():
    graph = PlanGraph(graph_id="g1", session_id="s1")
    graph.add_node(
        PlanNode(
            node_id="P1",
            action_type="REASONING",
            role="synthesizer",
            payload={"instruction": "bad"},
        )
    )
    ok, errors = validate_plan(graph, allowed_tools=["search_web"])
    assert ok is False
    assert any("cannot use synthesizer role" in str(err) for err in errors)


def test_simple_hard_constraint_guard_research_pro_stays_strict():
    base = TaskManager._resolve_mode_policy("research_pro")
    role_applied, _ = TaskManager._apply_role_policy(base, "planner")
    guarded = TaskManager._apply_hard_constraint_guard(mode_policy=base, candidate_policy=role_applied)

    assert guarded.tool_force_for_non_smalltalk is True
    assert guarded.strict_evidence is True
    assert guarded.require_multi_source is True
    assert guarded.require_trusted_sources_only is True
    assert int(guarded.min_reliability_budget_ms) >= int(base.min_reliability_budget_ms)
    assert int(guarded.max_reliability_retries) >= int(base.max_reliability_retries)


def test_simple_telemetry_collector_reads_role_flow_from_graph_metadata():
    graph = PlanGraph(graph_id="g2", session_id="s2")
    graph.metadata["role_flow"] = [
        {
            "node_id": "P1",
            "action_type": "TOOL_CALL",
            "role": "researcher",
            "status": "COMPLETED",
            "duration_ms": 120,
            "retries": 0,
            "tool_calls": 1,
            "role_fallback_applied": False,
            "role_resolution_source": "action_mapping",
        },
        {
            "node_id": "FINAL",
            "action_type": "REASONING",
            "role": "synthesizer",
            "status": "COMPLETED",
            "duration_ms": 30,
            "retries": 0,
            "tool_calls": 0,
            "role_fallback_applied": False,
            "role_resolution_source": "final_lock",
        },
    ]
    out = TaskManager._collect_role_telemetry_from_graph(graph=graph, mode_name="research_pro")
    assert len(out["role_flow"]) == 2
    assert out["role_summary"]["final_synthesizer_ok"] is True
    assert out["role_metrics"]["researcher"]["runs"] == 1
    assert out["role_metrics"]["synthesizer"]["runs"] == 1
