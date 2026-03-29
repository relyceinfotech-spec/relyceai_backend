"""
Plan Validator
Verifies the integrity and safety of a PlanGraph before execution.
"""
from typing import List, Tuple
from app.state.plan_graph import PlanGraph
from app.agent.role_cognition import ROLE_POLICIES

def validate_plan(graph: PlanGraph, allowed_tools: List[str]) -> Tuple[bool, List[str]]:
    """
    Validates a PlanGraph for:
    1. Unique node IDs (implicitly handled by PlanGraph.add_node).
    2. Valid tool names.
    3. Valid dependencies.
    4. Cycle-free DAG.
    """
    errors = []

    # 1. Tool Validation
    for node_id, node in graph.nodes.items():
        if node.action_type == "TOOL_CALL":
            tool_name = node.payload.get("tool")
            if not tool_name:
                errors.append(f"Node {node_id} is a TOOL_CALL but missing 'tool' in payload.")
            elif tool_name not in allowed_tools:
                errors.append(f"Node {node_id} uses disallowed tool: {tool_name}")

    # 2. Dependency Validation
    for node_id, node in graph.nodes.items():
        for dep_id in node.dependencies:
            if dep_id not in graph.nodes:
                errors.append(f"Node {node_id} depends on non-existent node: {dep_id}")

    # 3. Role Validation
    for node_id, node in graph.nodes.items():
        role = str(getattr(node, "role", "") or "").strip().lower()
        if role and role not in ROLE_POLICIES:
            errors.append(f"Node {node_id} has unsupported role: {role}")
        if role == "synthesizer" and str(node_id).upper() != "FINAL":
            errors.append(f"Node {node_id} cannot use synthesizer role (reserved for FINAL node)")
        if str(node_id).upper() == "FINAL" and role and role != "synthesizer":
            errors.append("FINAL node role must be synthesizer")

    # 4. Cycle Validation
    if not graph.validate_cycle_free():
        errors.append("PlanGraph contains circular dependencies (not a DAG).")

    is_valid = len(errors) == 0
    return is_valid, errors
