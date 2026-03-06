"""
Relyce AI - Repair Engine
Failure classification and bounded repair cycle.

Called ONLY from processor.py — never from hybrid controller.
Handles recovery when code execution fails, with a hard limit
of MAX_REPAIR_ATTEMPTS to prevent infinite loops.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ============================================
# CONSTANTS
# ============================================

MAX_REPAIR_ATTEMPTS = 2  # Hard safety limit — never override


# ============================================
# FAILURE TYPES
# ============================================

class FailureType:
    FIX_CODE = "FIX_CODE"
    INSTALL_OR_REPLACE = "INSTALL_OR_REPLACE"
    REWRITE_FUNCTION = "REWRITE_FUNCTION"
    ADAPT_LIBRARY = "ADAPT_LIBRARY"
    UNKNOWN = "UNKNOWN"


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class RepairResult:
    """Result of a single repair attempt."""
    success: bool = False
    strategy_used: str = ""
    error_before: str = ""
    error_after: Optional[str] = None
    attempt: int = 0


@dataclass
class RepairCycleResult:
    """Result of the full bounded repair cycle."""
    status: str = "no_repair_needed"  # "repair_success" | "repair_failed" | "no_repair_needed"
    attempts: int = 0
    repairs: List[RepairResult] = field(default_factory=list)
    final_failure_type: Optional[str] = None


# ============================================
# FAILURE CLASSIFICATION PATTERNS
# ============================================

_SYNTAX_PATTERNS = [
    r"SyntaxError",
    r"IndentationError",
    r"TabError",
    r"unexpected EOF",
    r"invalid syntax",
    r"unexpected token",
    r"Unexpected token",
]

_DEPENDENCY_PATTERNS = [
    r"ModuleNotFoundError",
    r"ImportError",
    r"No module named",
    r"Cannot find module",
    r"Package .* not found",
    r"npm ERR!",
    r"pip .* not found",
]

_LOGIC_PATTERNS = [
    r"TypeError",
    r"ValueError",
    r"AttributeError",
    r"KeyError",
    r"IndexError",
    r"ZeroDivisionError",
    r"RecursionError",
    r"AssertionError",
    r"NameError",
    r"UnboundLocalError",
]

_VERSION_PATTERNS = [
    r"version .* not compatible",
    r"deprecated",
    r"DeprecationWarning",
    r"requires .* version",
    r"incompatible",
    r"breaking change",
]


# ============================================
# FAILURE CLASSIFICATION
# ============================================

def classify_failure(error: str) -> str:
    """
    Classify an error into a failure type for repair routing.

    Priority order:
      1. Syntax errors → FIX_CODE (simple patch)
      2. Dependency errors → INSTALL_OR_REPLACE
      3. Logic errors → REWRITE_FUNCTION
      4. Version conflicts → ADAPT_LIBRARY
      5. Unknown → UNKNOWN (fallback)
    """
    if not error:
        return FailureType.UNKNOWN

    error_str = str(error)

    if any(re.search(p, error_str, re.IGNORECASE) for p in _SYNTAX_PATTERNS):
        return FailureType.FIX_CODE

    if any(re.search(p, error_str, re.IGNORECASE) for p in _DEPENDENCY_PATTERNS):
        return FailureType.INSTALL_OR_REPLACE

    if any(re.search(p, error_str, re.IGNORECASE) for p in _LOGIC_PATTERNS):
        return FailureType.REWRITE_FUNCTION

    if any(re.search(p, error_str, re.IGNORECASE) for p in _VERSION_PATTERNS):
        return FailureType.ADAPT_LIBRARY

    return FailureType.UNKNOWN


# ============================================
# REPAIR STRATEGY GENERATION
# ============================================

_PIVOT_PROMPT_PREFIX = (
    "WARNING: You attempted to fix this exact type of error in the previous step, "
    "but the execution failed again. Your previous approach did not work. "
    "You MUST pivot and try a completely different strategy. Do NOT repeat the previous fix.\n\n"
)

_REPAIR_PROMPTS = {
    FailureType.FIX_CODE: (
        "The code has a syntax error. Fix ONLY the syntax issue. "
        "Do not change logic, do not refactor, do not add features. "
        "Return the corrected code only."
    ),
    FailureType.INSTALL_OR_REPLACE: (
        "A required dependency is missing or cannot be found. "
        "Either add the correct import/install command, or replace the "
        "dependency with an equivalent that is available. "
        "Return the corrected code only."
    ),
    FailureType.REWRITE_FUNCTION: (
        "The code has a logic error (wrong type, missing key, bad index, etc.). "
        "Rewrite the failing function to fix the logic issue. "
        "Preserve the function's interface (name, parameters, return type). "
        "Return the corrected code only."
    ),
    FailureType.ADAPT_LIBRARY: (
        "A library version conflict or deprecated API is causing the error. "
        "Update the code to use the current API. Do not downgrade — adapt forward. "
        "Return the corrected code only."
    ),
    FailureType.UNKNOWN: (
        "An unknown error occurred. Analyze the error message, identify the root cause, "
        "and apply the minimal fix needed. Return the corrected code only."
    ),
}


from app.memory.strategy_memory import get_memory

def generate_repair_strategy(error: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Generate a repair strategy for the given error.

    Returns a dict containing:
      - failure_type: classified failure category
      - repair_prompt: instruction for LLM to apply repair
      - original_error: the error string
    """
    failure_type = classify_failure(error)
    repair_prompt = _REPAIR_PROMPTS.get(failure_type, _REPAIR_PROMPTS[FailureType.UNKNOWN])

    # Diversification: Check if this exact failure type just happened in current history
    if context and "repair_history" in context:
        history = context["repair_history"]
        if history and history.count(failure_type) >= 1:
            # Check if memory says we've failed this a lot over the session
            session_id = context.get("session_id")
            if session_id:
                mem = get_memory(session_id)
                if mem["repair_failures"].get(failure_type, 0) > 3.0:
                    repair_prompt = (
                        "You have repeatedly failed to fix this issue with small patches. "
                        "You must pivot your strategy completely. Rewrite the failing component "
                        "from scratch using a different approach. Do not attempt a minor fix."
                    )
                else:
                    repair_prompt = _PIVOT_PROMPT_PREFIX + repair_prompt
            else:
                repair_prompt = _PIVOT_PROMPT_PREFIX + repair_prompt

    return {
        "failure_type": failure_type,
        "repair_prompt": repair_prompt,
        "original_error": str(error),
    }


# ============================================
# BOUNDED REPAIR CYCLE
# ============================================

def repair_cycle(
    failure: str,
    context: Optional[Dict] = None,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> RepairCycleResult:
    """
    Run bounded repair cycle for a failed execution.

    Called ONLY from processor.py.

    This function classifies the failure, generates a repair strategy,
    and returns the strategy for the processor to apply. It does NOT
    execute the repair itself — the processor handles LLM calls.

    Args:
        failure: Error string from failed execution
        context: Execution context dict
        max_attempts: Max repair attempts (capped at MAX_REPAIR_ATTEMPTS)

    Returns:
        RepairCycleResult with status and repair strategies
    """
    if context is None:
        context = {}

    # Hard cap — never exceed MAX_REPAIR_ATTEMPTS regardless of input
    max_attempts = min(max_attempts, MAX_REPAIR_ATTEMPTS)

    result = RepairCycleResult()

    if not failure:
        result.status = "no_repair_needed"
        return result

    failure_type = classify_failure(failure)
    result.final_failure_type = failure_type

    # Initialize or append to repair history
    if "repair_history" not in context:
        context["repair_history"] = []
    
    strategy = generate_repair_strategy(failure, context)

    # Record this failure type AFTER generating strategy to evaluate sequence
    context["repair_history"].append(failure_type)

    # Store strategy in context so processor can use it
    context["repair_strategy"] = strategy
    context["repair_max_attempts"] = max_attempts

    # Mark as needing repair — processor will execute
    result.status = "repair_needed"
    result.attempts = 0  # processor tracks actual attempts

    return result


def build_repair_prompt(
    original_code: str,
    error: str,
    repair_strategy: Dict,
) -> str:
    """
    Build a complete repair prompt for the LLM.

    Called by processor.py to generate the retry message.
    """
    failure_type = repair_strategy.get("failure_type", "UNKNOWN")
    repair_instruction = repair_strategy.get("repair_prompt", "Fix the error.")

    prompt = (
        f"**ERROR TYPE:** {failure_type}\n\n"
        f"**ERROR:**\n```\n{error}\n```\n\n"
        f"**ORIGINAL CODE:**\n```\n{original_code}\n```\n\n"
        f"**REPAIR INSTRUCTION:**\n{repair_instruction}\n\n"
        f"Return ONLY the corrected code. No explanations."
    )

    return prompt
