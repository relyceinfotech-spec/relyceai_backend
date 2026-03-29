"""
Relyce AI - Output Validator
Post-generation validation for the Safe Chat Agent.

Only runs when STRICT limits exist. Zero overhead otherwise.
Handles: count validation, length validation, structure validation,
structural-safe trimming, retry control, clarity notes.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from app.chat.session_rules import Rule, Severity, DEBUG_SAFE_AGENT
from app.llm.safe_agent import ImpossibilityTier, check_semantic_loss

# ============================================
# CONSTANTS
# ============================================

MAX_RETRIES = 2

# Filler patterns to trim (structural-safe: preserves headers/labels/axes)
FILLER_INTRO_PATTERNS = [
    re.compile(r"^(?:Sure!?|Of course!?|Absolutely!?|Great question!?|Here(?:'s| is| are))[^\n]*\n*", re.IGNORECASE),
    re.compile(r"^(?:Let me |I'd be happy to |I'll )[^\n]*\n*", re.IGNORECASE),
    re.compile(r"^(?:Certainly!?|No problem!?|Happy to help!?)[^\n]*\n*", re.IGNORECASE),
]

FILLER_OUTRO_PATTERNS = [
    re.compile(r"\n*(?:I hope this helps|Hope that helps|Let me know if)[^\n]*$", re.IGNORECASE),
    re.compile(r"\n*(?:Feel free to ask|Don't hesitate|If you (?:need|have|want))[^\n]*$", re.IGNORECASE),
    re.compile(r"\n*(?:Is there anything else|Would you like)[^\n]*$", re.IGNORECASE),
]

# Structural anchors to preserve during trimming
STRUCTURAL_ANCHOR_PATTERNS = [
    re.compile(r"^#{1,3}\s+.+", re.MULTILINE),          # Markdown headings
    re.compile(r"^\d+\.\s+.+", re.MULTILINE),            # Numbered items
    re.compile(r"^[-*]\s+.+", re.MULTILINE),              # Bullet items
    re.compile(r"^\|.+\|$", re.MULTILINE),                # Table rows
    re.compile(r"^(?:\w+)\s*(?:vs\.?|versus)\s*\w+", re.MULTILINE | re.IGNORECASE),  # Comparison axes
]

# Trim revert threshold (char-based)
TRIM_REVERT_THRESHOLD = 0.30  # Revert if >30% content removed


# ============================================
# VALIDATION RESULT
# ============================================

@dataclass
class ValidationResult:
    """Result of output validation."""
    passed: bool = True
    violations: List[str] = None
    trimmed_response: str = ""
    clarity_note: str = ""
    retries_used: int = 0
    was_trimmed: bool = False
    was_reverted: bool = False

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


# ============================================
# VALIDATORS
# ============================================

def _count_bullets(text: str) -> int:
    """Count bullet/numbered items in response."""
    bullet_lines = re.findall(r"^[\s]*[-*•]\s+.+", text, re.MULTILINE)
    numbered_lines = re.findall(r"^[\s]*\d+[.)]\s+.+", text, re.MULTILINE)
    return len(bullet_lines) + len(numbered_lines)


def _count_lines(text: str) -> int:
    """Count non-empty lines."""
    return len([l for l in text.strip().split("\n") if l.strip()])


def _count_paragraphs(text: str) -> int:
    """Count paragraphs (separated by blank lines)."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return len([p for p in paragraphs if p.strip()])


def count_validator(response: str, strict_rules: Dict[str, Rule]) -> List[str]:
    """Check bullet/item count against strict limit."""
    violations = []

    count_rule = strict_rules.get("count_limit")
    if count_rule and isinstance(count_rule.value, int):
        target = count_rule.value
        actual = _count_bullets(response)

        # Only flag if bullets exist and exceed limit
        if actual > 0 and actual > target:
            violations.append(
                f"count_exceeded: expected {target} items, got {actual}"
            )

    return violations


def length_validator(response: str, strict_rules: Dict[str, Rule]) -> List[str]:
    """Check response length against strict limit."""
    violations = []

    length_rule = strict_rules.get("length_limit")
    if not length_rule:
        return violations

    val = length_rule.value

    if val == "1_line":
        lines = _count_lines(response)
        if lines > 1:
            violations.append(f"length_exceeded: expected 1 line, got {lines}")

    elif val == "2_lines":
        lines = _count_lines(response)
        if lines > 2:
            violations.append(f"length_exceeded: expected 2 lines, got {lines}")

    elif val == "1_word":
        words = len(response.strip().split())
        if words > 3:  # Allow small margin for punctuation
            violations.append(f"length_exceeded: expected ~1 word, got {words}")

    elif isinstance(val, int):
        lines = _count_lines(response)
        if lines > val + 1:  # Allow 1 line margin
            violations.append(f"length_exceeded: expected {val} lines, got {lines}")

    return violations


def structure_validator(response: str, strict_rules: Dict[str, Rule]) -> List[str]:
    """Check format compliance."""
    violations = []

    format_rule = strict_rules.get("format_preference")
    if not format_rule or not format_rule.is_strict():
        return violations

    fmt = format_rule.value

    if fmt == "bullet":
        bullet_count = _count_bullets(response)
        if bullet_count == 0 and len(response.strip()) > 50:
            violations.append("format_mismatch: expected bullet format, got prose")

    elif fmt == "table":
        if "|" not in response:
            violations.append("format_mismatch: expected table format, no table found")

    elif fmt == "numbered":
        numbered = re.findall(r"^\s*\d+[.)]\s+", response, re.MULTILINE)
        if not numbered and len(response.strip()) > 50:
            violations.append("format_mismatch: expected numbered list, got prose")

    return violations


# ============================================
# STRUCTURAL-SAFE TRIMMER
# ============================================

def trim_response(response: str, strict_rules: Dict[str, Rule]) -> Tuple[str, bool]:
    """
    Remove filler while preserving structural anchors.
    Returns (trimmed_text, was_trimmed).
    Reverts if >30% content removed (char-based).
    """
    original = response
    original_len = len(original)
    result = response

    # Remove intro filler
    for pattern in FILLER_INTRO_PATTERNS:
        result = pattern.sub("", result, count=1)

    # Remove outro filler
    for pattern in FILLER_OUTRO_PATTERNS:
        result = pattern.sub("", result, count=1)

    result = result.strip()

    if not result:
        return original.strip(), False

    # Check revert threshold (char-based)
    removed_chars = original_len - len(result)
    if original_len > 0:
        removed_ratio = removed_chars / original_len
        if removed_ratio > TRIM_REVERT_THRESHOLD:
            if DEBUG_SAFE_AGENT:
                print(f"[Validator] trim reverted: {removed_ratio:.0%} removed > {TRIM_REVERT_THRESHOLD:.0%} threshold")
            return original.strip(), False

    was_trimmed = len(result) < original_len
    return result, was_trimmed


def _trim_excess_items(response: str, max_count: int) -> str:
    """Trim excess bullet/numbered items to match count limit."""
    lines = response.split("\n")
    result_lines = []
    item_count = 0

    for line in lines:
        is_item = bool(re.match(r"^\s*[-*•]\s+", line) or re.match(r"^\s*\d+[.)]\s+", line))

        if is_item:
            item_count += 1
            if item_count <= max_count:
                result_lines.append(line)
            # Skip excess items
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


# ============================================
# RETRY PROMPT BUILDER
# ============================================

def build_retry_prompt(
    original_query: str,
    previous_response: str,
    violations: List[str],
    strict_rules: Dict[str, Rule],
) -> str:
    """Build a tightened prompt for retry."""
    constraint_lines = []

    for key, rule in strict_rules.items():
        if key == "count_limit" and isinstance(rule.value, int):
            constraint_lines.append(f"EXACTLY {rule.value} items. No more, no less.")
        elif key == "length_limit":
            val = rule.value
            if val == "1_line":
                constraint_lines.append("EXACTLY 1 line. Nothing more.")
            elif val == "2_lines":
                constraint_lines.append("EXACTLY 2 lines. Nothing more.")
            elif val == "1_word":
                constraint_lines.append("Respond with EXACTLY 1 word.")
            elif isinstance(val, int):
                constraint_lines.append(f"EXACTLY {val} lines. Nothing more.")
        elif key == "format_preference":
            constraint_lines.append(f"Use {rule.value} format.")

    constraints_str = "\n".join(f"- {c}" for c in constraint_lines)

    return (
        f"The previous response violated the user's constraints.\n"
        f"Violations: {', '.join(violations)}\n\n"
        f"STRICT REQUIREMENTS:\n{constraints_str}\n\n"
        f"Rewrite your response to the following query, strictly obeying the constraints above.\n"
        f"Do NOT add introductions, conclusions, or filler.\n\n"
        f"Query: {original_query}"
    )


# ============================================
# MAIN VALIDATION ENTRY POINT
# ============================================

def validate(
    response: str,
    active_rules: Dict[str, Rule],
    intent: str = "explain",
    query: str = "",
) -> ValidationResult:
    """
    Run all validators on the response.
    Only processes STRICT rules. Returns ValidationResult.
    """
    strict_rules = {k: v for k, v in active_rules.items() if v.is_strict()}

    # Fast path: no strict rules → skip entirely
    if not strict_rules:
        return ValidationResult(passed=True, trimmed_response=response)

    result = ValidationResult()

    # Run validators
    violations = []
    violations.extend(count_validator(response, strict_rules))
    violations.extend(length_validator(response, strict_rules))
    violations.extend(structure_validator(response, strict_rules))

    result.violations = violations
    result.passed = len(violations) == 0

    # Apply structural-safe trim
    trimmed, was_trimmed = trim_response(response, strict_rules)
    result.was_trimmed = was_trimmed

    # If count exceeded, try trimming excess items
    count_rule = strict_rules.get("count_limit")
    if count_rule and isinstance(count_rule.value, int):
        bullet_count = _count_bullets(trimmed)
        if bullet_count > count_rule.value:
            trimmed = _trim_excess_items(trimmed, count_rule.value)
            result.was_trimmed = True

            # Re-validate after trim
            new_violations = []
            new_violations.extend(count_validator(trimmed, strict_rules))
            new_violations.extend(length_validator(trimmed, strict_rules))
            if not new_violations:
                result.passed = True
                result.violations = []

    result.trimmed_response = trimmed

    # Clarity notes (intent-gated)
    if strict_rules and result.passed:
        has_loss = check_semantic_loss(query, intent)
        if has_loss:
            result.clarity_note = "\n\n*I can expand this if you'd like.*"
        elif was_trimmed:
            result.clarity_note = "\n\n*(Kept within your requested limit)*"

    if DEBUG_SAFE_AGENT:
        print(
            f"[Validator] validated={result.passed} "
            f"violations={len(result.violations)} "
            f"trimmed={result.was_trimmed} "
            f"clarity={'yes' if result.clarity_note else 'no'}"
        )

    return result


def should_retry(
    validation_result: ValidationResult,
    impossibility_tier: str,
    retry_count: int,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """
    Decide whether to retry generation.
    IMPOSSIBLE → never retry.
    Check budget.
    """
    if validation_result.passed:
        return False

    if impossibility_tier == ImpossibilityTier.IMPOSSIBLE:
        return False

    if retry_count >= max_retries:
        return False

    return True


def get_retry_mode(retry_count: int) -> str:
    """
    Retry 1 → single-shot (tightened).
    Retry 2 → streaming + trim (fallback).
    """
    if retry_count < 1:
        return "single-shot"
    else:
        return "streaming-trim"
