from app.agent.plan_validator import validate_plan
from app.agent.role_cognition import resolve_plan_node_role
from app.agent.task_manager import TaskManager
from app.state.plan_graph import PlanGraph, PlanNode


def test_role_resolver_final_lock_to_synthesizer():
    resolved = resolve_plan_node_role(node_id="FINAL", action_type="TOOL_CALL", declared_role="researcher")
    assert resolved.role == "synthesizer"
    assert resolved.role_fallback_applied is False
    assert resolved.role_resolution_source == "final_lock"


def test_role_resolver_unknown_role_falls_back_to_executor():
    resolved = resolve_plan_node_role(node_id="P1", action_type="REASONING", declared_role="mystery_role")
    assert resolved.role == "executor"
    assert resolved.role_fallback_applied is True
    assert resolved.role_resolution_source == "unknown_declared_role_fallback"


def test_validator_rejects_non_final_synthesizer():
    graph = PlanGraph(graph_id="g-role", session_id="s-role")
    graph.add_node(
        PlanNode(
            node_id="P1",
            action_type="REASONING",
            role="synthesizer",
            payload={"instruction": "bad role"},
        )
    )
    ok, errors = validate_plan(graph, allowed_tools=["search_web"])
    assert ok is False
    assert any("cannot use synthesizer role" in err for err in errors)


def test_hard_constraint_guard_keeps_research_pro_strictness():
    base = TaskManager._resolve_mode_policy("research_pro")
    # Planner role attempts to soften behavior; hard guard must preserve strict mode constraints.
    softened, _ = TaskManager._apply_role_policy(base, "planner")
    guarded = TaskManager._apply_hard_constraint_guard(mode_policy=base, candidate_policy=softened)

    assert guarded.tool_force_for_non_smalltalk is True
    assert guarded.strict_evidence is True
    assert guarded.require_multi_source is True
    assert guarded.require_trusted_sources_only is True
    assert int(guarded.min_reliability_budget_ms) >= int(base.min_reliability_budget_ms)
    assert int(guarded.max_reliability_retries) >= int(base.max_reliability_retries)
