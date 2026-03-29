from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set


ROLE_PLANNER = "planner"
ROLE_RESEARCHER = "researcher"
ROLE_EXECUTOR = "executor"
ROLE_CRITIC = "critic"
ROLE_SYNTHESIZER = "synthesizer"

SUPPORTED_ROLES: Set[str] = {
    ROLE_PLANNER,
    ROLE_RESEARCHER,
    ROLE_EXECUTOR,
    ROLE_CRITIC,
    ROLE_SYNTHESIZER,
}


ROLE_POLICIES: Dict[str, Dict[str, Any]] = {
    ROLE_PLANNER: {
        "reasoning_hints_enabled": True,
        "tool_force_for_non_smalltalk": False,
    },
    ROLE_RESEARCHER: {
        "memory_read_enabled": True,
        "tool_force_for_non_smalltalk": True,
        "strict_evidence": True,
        "require_multi_source": True,
    },
    ROLE_EXECUTOR: {
        "memory_read_enabled": True,
    },
    ROLE_CRITIC: {
        "critic_repair_requires_missing_sources": True,
        "strict_evidence": True,
        "require_multi_source": True,
    },
    ROLE_SYNTHESIZER: {
        "reasoning_hints_enabled": False,
    },
}


@dataclass(frozen=True)
class RoleResolution:
    role: str
    role_fallback_applied: bool
    role_resolution_source: str
    warning: str = ""


def map_role_from_node(node_id: str, action_type: str) -> str:
    node_key = str(node_id or "").strip().upper()
    action_key = str(action_type or "").strip().upper()
    if node_key == "FINAL":
        return ROLE_SYNTHESIZER
    if action_key in {"VALIDATION", "REPAIR"}:
        return ROLE_CRITIC
    if action_key == "TOOL_CALL":
        return ROLE_RESEARCHER
    if action_key == "REASONING":
        return ROLE_PLANNER
    return ROLE_EXECUTOR


def resolve_plan_node_role(
    *,
    node_id: str,
    action_type: str,
    declared_role: Optional[str] = None,
) -> RoleResolution:
    node_key = str(node_id or "").strip().upper()

    # Hard lock: FINAL node is always synthesizer, regardless of declared role.
    if node_key == "FINAL":
        return RoleResolution(
            role=ROLE_SYNTHESIZER,
            role_fallback_applied=False,
            role_resolution_source="final_lock",
        )

    raw_declared = str(declared_role or "").strip().lower()
    if raw_declared:
        if raw_declared in SUPPORTED_ROLES:
            return RoleResolution(
                role=raw_declared,
                role_fallback_applied=False,
                role_resolution_source="declared_role",
            )
        return RoleResolution(
            role=ROLE_EXECUTOR,
            role_fallback_applied=True,
            role_resolution_source="unknown_declared_role_fallback",
            warning=f"unknown role '{raw_declared}' -> executor fallback",
        )

    mapped = map_role_from_node(node_id=node_id, action_type=action_type)
    if mapped not in ROLE_POLICIES:
        return RoleResolution(
            role=ROLE_EXECUTOR,
            role_fallback_applied=True,
            role_resolution_source="unknown_mapped_role_fallback",
            warning=f"mapped role '{mapped}' is unsupported -> executor fallback",
        )
    return RoleResolution(
        role=mapped,
        role_fallback_applied=False,
        role_resolution_source="action_mapping",
    )
