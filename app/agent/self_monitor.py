"""
Relyce AI - Self Monitor
Layer 14: Passive post-response observer.

CRITICAL: Record only — NEVER intervene in live behavior.
           Feeds data into feedback_engine for long-term learning.
           No loops, no live adjustments.

Evaluates:
  - Goal match: did the response address the user's intent?
  - Constraint adherence: did STRICT limits get respected?
  - Tool success: did tool calls return useful results?
  - Response clarity: is the response well-structured?
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================
# DATA STRUCTURE
# ============================================

@dataclass
class MonitorReport:
    """Passive evaluation report. Logged, never applied live."""
    goal_match: float = 1.0         # 0.0–1.0 — did response match intent?
    constraint_adherence: bool = True  # did STRICT rules get respected?
    tool_success: bool = True       # did tool calls return useful results?
    clarity_score: float = 1.0      # 0.0–1.0 — response quality
    adjustments: List[str] = field(default_factory=list)  # logged, NOT applied


# ============================================
# CONFIGURATION
# ============================================

# Minimum response length for meaningful evaluation
_MIN_RESPONSE_LENGTH = 10

# Intent keywords for goal matching
_INTENT_KEYWORDS = {
    "explain": ["because", "means", "refers to", "defined as", "works by", "is a"],
    "summarize": ["in summary", "key points", "overview", "brief", "tldr"],
    "list": ["1.", "2.", "- ", "* ", "•"],
    "compare": ["vs", "versus", "whereas", "while", "compared to", "difference"],
    "generate": ["```", "here is", "here's", "created", "generated"],
    "transform": ["converted", "translated", "rewritten", "rephrased"],
}


# ============================================
# MAIN FUNCTION
# ============================================

def evaluate_response(
    query: str,
    response: str,
    constraint_analysis: Optional[object] = None,
    tool_results: Optional[List[Dict]] = None,
    intent: str = "explain",
) -> MonitorReport:
    """
    Passively evaluate a response. Does NOT change any live behavior.

    Args:
        query: Original user query
        response: Generated response text
        constraint_analysis: ConstraintAnalysis from safe_agent (optional)
        tool_results: List of tool call results (optional)
        intent: Detected intent from router

    Returns:
        MonitorReport — logged to feedback engine, never applied live
    """
    report = MonitorReport()

    if not response or len(response.strip()) < _MIN_RESPONSE_LENGTH:
        report.goal_match = 0.2
        report.clarity_score = 0.2
        report.adjustments.append("response_too_short")
        return report

    # --- Goal match: keyword overlap ---
    report.goal_match = _evaluate_goal_match(query, response, intent)

    # --- Constraint adherence ---
    report.constraint_adherence = _check_constraints(response, constraint_analysis)

    # --- Tool success ---
    report.tool_success = _check_tool_success(tool_results)

    # --- Clarity score ---
    report.clarity_score = _evaluate_clarity(response)

    # --- Collect adjustments (passive observations) ---
    if report.goal_match < 0.4:
        report.adjustments.append("low_goal_match")
    if not report.constraint_adherence:
        report.adjustments.append("constraint_violated")
    if not report.tool_success:
        report.adjustments.append("tool_returned_empty")
    if report.clarity_score < 0.4:
        report.adjustments.append("low_clarity")

    return report


# ============================================
# INTERNAL EVALUATORS
# ============================================

def _evaluate_goal_match(query: str, response: str, intent: str) -> float:
    """
    Check if response addresses the query's intent.
    Simple keyword-overlap heuristic — not LLM-based.
    """
    score = 0.5  # neutral baseline
    r_lower = response.lower()

    # Check intent-specific markers
    markers = _INTENT_KEYWORDS.get(intent, [])
    if markers:
        hits = sum(1 for m in markers if m in r_lower)
        if hits >= 2:
            score += 0.3
        elif hits >= 1:
            score += 0.15

    # Check query keyword presence in response
    query_words = set(re.findall(r"\b\w{4,}\b", query.lower()))
    response_words = set(re.findall(r"\b\w{4,}\b", r_lower))
    if query_words:
        overlap = len(query_words & response_words) / len(query_words)
        score += overlap * 0.2

    return min(1.0, max(0.0, score))


def _check_constraints(
    response: str,
    constraint_analysis: Optional[object] = None,
) -> bool:
    """
    Check if STRICT constraints were respected.
    Uses the constraint_analysis from safe_agent if available.
    """
    if constraint_analysis is None:
        return True  # no constraints to check

    # Check if constraint_analysis has active_rules
    active_rules = getattr(constraint_analysis, "active_rules", {})
    if not active_rules:
        return True

    # Check for STRICT rules
    strict_rules = {k: v for k, v in active_rules.items()
                    if hasattr(v, "is_strict") and v.is_strict()}

    if not strict_rules:
        return True

    # Simple count check
    for key, rule in strict_rules.items():
        if key == "count_limit" and isinstance(rule.value, int):
            # Count bullet/numbered items
            items = re.findall(r"(?:^|\n)\s*(?:\d+\.|[-*•])\s+", response)
            if len(items) > rule.value:
                return False

        elif key == "length_limit":
            lines = [l for l in response.split("\n") if l.strip()]
            if rule.value == "1_line" and len(lines) > 1:
                return False
            elif rule.value == "2_lines" and len(lines) > 2:
                return False
            elif isinstance(rule.value, int) and len(lines) > rule.value:
                return False

    return True


def _check_tool_success(tool_results: Optional[List[Dict]] = None) -> bool:
    """Check if tool calls returned any useful results."""
    if tool_results is None:
        return True  # no tool calls — not a failure

    if not tool_results:
        return True  # empty list — no tools were called

    # Check if any tool returned empty or error
    for result in tool_results:
        if isinstance(result, dict):
            content = result.get("content") or result.get("result") or ""
            if not content or content.strip() in ("", "null", "none", "error"):
                return False

    return True


def _evaluate_clarity(response: str) -> float:
    """
    Evaluate response clarity/structure.
    Higher scores for well-structured responses.
    """
    score = 0.5  # baseline

    # Has structure (headings, lists, code blocks)
    if re.search(r"^#{1,3}\s+", response, re.MULTILINE):
        score += 0.15  # headings
    if re.search(r"(?:^|\n)\s*(?:\d+\.|[-*])\s+", response):
        score += 0.1  # lists
    if "```" in response:
        score += 0.1  # code blocks

    # Reasonable length (not too terse, not too verbose)
    word_count = len(response.split())
    if 20 <= word_count <= 500:
        score += 0.1
    elif word_count < 10:
        score -= 0.2

    # Has paragraphs (not a wall of text)
    paragraphs = response.split("\n\n")
    if len(paragraphs) >= 2:
        score += 0.05

    return min(1.0, max(0.0, score))
