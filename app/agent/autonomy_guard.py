"""
Relyce AI - Autonomy Guard
Layers 12–13: Autonomy Boundaries + Adaptive Autonomy.

Classifies every action before execution:
  Low risk    → suggest / execute
  Medium risk → prepare
  High risk   → confirm

With guardrails:
  - Reversibility check: irreversible actions → always confirm
  - Goal-aware downgrade: outcome-oriented tasks → prepare, not execute
  - Adaptive mode from user profile: guide / assistant / executor
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.agent.action_classifier import ActionDecision, is_reversible


# ============================================
# DATA STRUCTURE
# ============================================

@dataclass
class AutonomyDecision:
    """Result of autonomy evaluation."""
    risk_tier: str = "low"          # "low" | "medium" | "high"
    reversible: bool = True         # can this action be undone?
    autonomy_mode: str = "assistant"  # "guide" | "assistant" | "executor"
    action: str = "suggest"         # "suggest" | "prepare" | "confirm" | "execute"
    reason: str = ""


# ============================================
# MODE THRESHOLDS
# ============================================

# Minimum interactions before promoting to higher autonomy
_EXECUTOR_THRESHOLD = 50    # interactions before executor mode
_ASSISTANT_THRESHOLD = 10   # interactions before assistant mode


# ============================================
# CORE FUNCTION
# ============================================

def evaluate_autonomy(
    action_decision: ActionDecision,
    user_profile: Optional[object] = None,
) -> AutonomyDecision:
    """
    Evaluate autonomy for an action based on risk, reversibility, goal context,
    and user profile.

    Args:
        action_decision: From action_classifier
        user_profile: UserProfile from user_profiler (optional)

    Returns:
        AutonomyDecision with action recommendation
    """
    result = AutonomyDecision()
    query_text = action_decision.goal.goal or ""

    # --- Step 1: Determine autonomy mode from user profile ---
    result.autonomy_mode = _determine_mode(user_profile)

    # --- Step 2: Copy risk from action decision ---
    result.risk_tier = action_decision.risk_level

    # --- Step 3: Check reversibility ---
    result.reversible = is_reversible(query_text)

    # --- Step 4: Base action from risk tier ---
    if result.risk_tier == "high":
        result.action = "confirm"
        result.reason = "high-risk action"
    elif result.risk_tier == "medium":
        result.action = "prepare"
        result.reason = "medium-risk action"
    else:
        result.action = "execute"
        result.reason = "low-risk action"

    # --- Step 5: Reversibility guardrail ---
    # Irreversible actions at medium+ risk → always confirm
    if not result.reversible and result.risk_tier in ("medium", "high"):
        result.action = "confirm"
        result.reason = f"irreversible {result.risk_tier}-risk action"

    # --- Step 6: Goal-aware guardrail ---
    # Outcome-oriented tasks should prepare, not execute blindly
    if (action_decision.goal.is_outcome_oriented
            and result.action == "execute"):
        result.action = "prepare"
        result.reason = "outcome-oriented task requires collaboration"

    # --- Step 7: Apply autonomy mode overrides ---
    if result.autonomy_mode == "guide":
        # Guide mode: always suggest, never execute directly
        if result.action == "execute":
            result.action = "suggest"
            result.reason = "guide mode — suggesting instead of executing"
    elif result.autonomy_mode == "executor":
        # Executor mode: can upgrade prepare → execute for reversible low-risk
        if (result.action == "prepare"
                and result.risk_tier == "low"
                and result.reversible
                and not action_decision.goal.is_outcome_oriented):
            result.action = "execute"
            result.reason = "executor mode — low-risk reversible action"
        # But NEVER auto-execute confirm-level actions
        # confirm stays confirm even in executor mode

    # --- Step 8: Questions always get "act" (respond) ---
    if action_decision.action_type == "QUESTION":
        result.action = "execute"
        result.reason = "question — responding directly"
        result.risk_tier = "low"
        result.reversible = True

    return result


def _determine_mode(user_profile: Optional[object] = None) -> str:
    """
    Determine autonomy mode based on user profile.

    Modes:
      guide    — new users (<10 interactions), always explain before acting
      assistant — regular users (10–50), act on clear tasks, ask on ambiguous
      executor — power users (>50), act autonomously, confirm only high-risk
    """
    if user_profile is None:
        return "assistant"  # safe default

    # Try to get interaction count from profile
    total = getattr(user_profile, "total_interactions", 0)
    if total is None:
        total = 0

    if total >= _EXECUTOR_THRESHOLD:
        return "executor"
    elif total >= _ASSISTANT_THRESHOLD:
        return "assistant"
    else:
        return "guide"
