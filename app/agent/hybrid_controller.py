"""
Relyce AI - Hybrid Controller (Advisory Only)
Returns HybridAdvice — NEVER executes anything.

Golden Rule:
  Hybrid = THINK
  Processor = EXECUTE

This module analyzes the query and produces strategic advice
that the orchestrator passes to the processor for execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from app.agent.strategy_engine import (
    detect_intent,
    select_planning_mode,
    build_parallel_reasoning,
    detect_knowledge_gap,
    decide_research,
    IntentSignals,
)
from app.memory.strategy_memory import get_memory


# ============================================
# ADVICE DATA STRUCTURE
# ============================================

@dataclass
class HybridAdvice:
    """
    Advisory output — consumed by orchestrator + processor.
    Contains NO execution logic, NO tool calls, NO side effects.
    """
    planning_mode: str = "SEQUENTIAL_PLAN"
    reasoning_context: Dict[str, str] = field(default_factory=dict)
    validation_required: bool = False
    repair_policy: Dict = field(default_factory=lambda: {
        "enabled": False,
        "max_attempts": 0,
    })
    research_needed: Optional[Dict] = None
    strategy_confidence: float = 1.0
    research_confidence: float = 0.0
    intent_signals: Optional[IntentSignals] = None


# ============================================
# ADVISORY ENTRY POINT
# ============================================

def generate_strategy_advice(
    query: str,
    context: Optional[Dict] = None,
    session_id: Optional[str] = None,
) -> HybridAdvice:
    """
    Generate strategic advice for query processing.

    This function ONLY thinks — it:
      1. Detects intent signals
      2. Selects planning mode
      3. Builds reasoning context (for PARALLEL_REASONING)
      4. Sets validation/repair policies (for ADAPTIVE_CODE_PLAN)

    It NEVER:
      - Executes tools
      - Calls LLMs
      - Modifies state
      - Runs repair cycles
      - Performs validation
    """
    intent = detect_intent(query)
    planning_mode = select_planning_mode(intent)

    reasoning_context: Dict[str, str] = {}
    validation_required = False
    repair_policy = {"enabled": False, "max_attempts": 0}

    # PARALLEL_REASONING: compose multi-track reasoning prompts
    if planning_mode == "PARALLEL_REASONING":
        reasoning_context = build_parallel_reasoning(query)

    # ADAPTIVE_CODE_PLAN: flag for validation + repair (executed by processor)
    if planning_mode == "ADAPTIVE_CODE_PLAN":
        validation_required = True
        repair_policy = {
            "enabled": True,
            "max_attempts": 2,
        }

    # KNOWLEDGE GAP & RESEARCH DECISION
    gap = detect_knowledge_gap(context)
    research_needed = decide_research(gap)

    # CONFIDENCE SCORING & ADAPTIVE RISK SCALING
    strategy_confidence = intent.strategy_confidence
    research_confidence = intent.research_confidence
    
    # ---------------------------------------------------------
    # MEMORY INFLUENCE (PHASE 4B)
    # ---------------------------------------------------------
    if session_id:
        mem = get_memory(session_id)
        # Penalize repeated misclassifications
        if mem["strategy_misclassifications"].get(planning_mode, 0) > 3:
            strategy_confidence = max(0.4, strategy_confidence - 0.1)
            
        # Penalize research overuse
        if mem["unnecessary_research_count"] > 5:
            research_confidence = max(0.4, research_confidence - 0.15)
            
        # Tweak repair policy aggressiveness
        if planning_mode == "ADAPTIVE_CODE_PLAN" and mem["repair_failures"].get("FIX_CODE", 0) > 3:
            # Force shorter loop, fallback to rewrite early
            repair_policy["max_attempts"] = 1

    # PHASE 4C: Strict Probability Clamping Guard
    strategy_confidence = max(0.0, min(1.0, strategy_confidence))
    research_confidence = max(0.0, min(1.0, research_confidence))

    if strategy_confidence < 0.5:
        # High uncertainty -> disable aggressive hooks
        research_needed = None
        validation_required = False
        repair_policy["enabled"] = False

    return HybridAdvice(
        planning_mode=planning_mode,
        reasoning_context=reasoning_context,
        validation_required=validation_required,
        repair_policy=repair_policy,
        research_needed=research_needed,
        strategy_confidence=strategy_confidence,
        research_confidence=research_confidence,
        intent_signals=intent,
    )
