"""
Relyce AI - Safe Chat Agent
Core constraint detection, task simplification, and prompt injection engine.

15-step pipeline:
 1. Resolve references       8.  Check compressibility
 2. Detect explicit limits   9.  Impossibility gradient
 3. Infer soft preferences   10. Build constraint prompt
 4. Classify intent          11. Decide streaming mode
 5. Detect override scope    12. Generate (caller)
 6. Confirmation reset       13. Validate (output_validator)
 7. Apply rule decay         14. Trim (output_validator)
                             15. Clarity notes (output_validator)
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from app.chat.session_rules import (
    Rule, Severity, OverrideScope, SessionState,
    get_session, get_active_rules, get_last_assistant_output,
    update_rules, confirmation_reset, apply_decay, apply_override,
    increment_turn, DEBUG_SAFE_AGENT,
)

# ============================================
# OPERATIONAL SAFETY
# ============================================

# Max time for constraint pipeline before skipping (seconds)
CONSTRAINT_TIMEOUT_S = float(os.getenv("CONSTRAINT_TIMEOUT_MS", "20")) / 1000.0


class _StrictRateTracker:
    """Tracks STRICT trigger rate over a rolling window for observability."""
    def __init__(self, window: int = 100):
        self._window = window
        self._total = 0
        self._strict_count = 0
        self._log_interval = 50  # Log every N requests

    def record(self, had_strict: bool) -> None:
        self._total += 1
        if had_strict:
            self._strict_count += 1
        if DEBUG_SAFE_AGENT and self._total % self._log_interval == 0:
            rate = (self._strict_count / self._total * 100) if self._total else 0
            print(f"[SafeAgent] strict_rate={rate:.0f}% ({self._strict_count}/{self._total})")

    @property
    def rate(self) -> float:
        return (self._strict_count / self._total * 100) if self._total else 0.0


strict_rate_tracker = _StrictRateTracker()


# --- Step 2: Explicit limit patterns (→ STRICT) ---
COUNT_PATTERNS = [
    # "3 points", "exactly 5 items", "only 2 examples", "no more than 4 tips"
    (r"(?:exactly\s+|only\s+|no more than\s+)?(\d{1,2})\s*(?:points?|items?|tips?|reasons?|benefits?|steps?|examples?|things?|ways?|ideas?|facts?|lines?|sentences?|paragraphs?|bullets?|headings?|topics?|sections?)", "count_limit"),
    # "in 2 lines", "within 3 paragraphs"
    (r"(?:in|within|under)\s+(\d{1,2})\s*(?:lines?|sentences?|paragraphs?|words?|bullets?)", "length_limit"),
]

FORMAT_PATTERNS = [
    (r"\b(?:bullet\s*(?:points?|list)?|bulleted)\b", "format_preference", "bullet"),
    (r"\b(?:numbered\s*(?:list)?|steps?)\b", "format_preference", "numbered"),
    (r"\b(?:table|tabular)\b", "format_preference", "table"),
    (r"\b(?:paragraph|prose)\b", "format_preference", "paragraph"),
]

LENGTH_KEYWORDS_STRICT = [
    (r"\b(?:one[\- ]?liner|single\s+line|1\s+line)\b", "length_limit", "1_line"),
    (r"\b(?:two[\- ]?liner|2\s+lines?)\b", "length_limit", "2_lines"),
]

# Assertive qualifiers that reinforce STRICT
STRICT_QUALIFIERS = re.compile(r"\b(?:exactly|only|must|strictly|no more than|at most|maximum|max)\b", re.IGNORECASE)

# --- Step 3: Soft inference (→ SOFT) ---
SOFT_INFERENCE_MAP = {
    # vague phrase → (rule_key, value)
    r"\b(?:few|a couple|couple of)\b": ("count_limit", "~3-5"),
    r"\b(?:quick(?:ly)?|brief(?:ly)?|short(?:ly)?|concise(?:ly)?|compact)\b": ("length_limit", "short"),
    r"\b(?:overview|high[\- ]?level|bird'?s?[\- ]?eye)\b": ("depth_limit", "shallow"),
    r"\b(?:key\s+(?:ideas?|points?|takeaways?)|main\s+(?:points?|ideas?))\b": ("format_preference", "list"),
    r"\b(?:one[\- ]?word|single[\- ]?word)\b": ("length_limit", "1_word"),
}

# Hedging words → force SOFT even if pattern matches
HEDGING_PATTERNS = re.compile(
    r"\b(?:try\s+to|preferably|if\s+possible|maybe|perhaps|ideally|roughly|about|around)\b",
    re.IGNORECASE,
)

# --- Step 4: Intent classification ---
INTENT_PATTERNS = {
    "explain": re.compile(r"\b(?:explain|describe|tell\s+me\s+about|what\s+is|what\s+are|define)\b", re.IGNORECASE),
    "summarize": re.compile(r"\b(?:summarize|summary|recap|tldr|tl;?dr|gist)\b", re.IGNORECASE),
    "list": re.compile(r"\b(?:list|enumerate|give\s+me|name\s+(?:some|the)|what\s+are\s+(?:some|the))\b", re.IGNORECASE),
    "compare": re.compile(r"\b(?:compare|vs\.?|versus|difference|differ|contrast|pros?\s+(?:and|&)\s+cons?)\b", re.IGNORECASE),
    "generate": re.compile(r"\b(?:write|create|generate|draft|compose|build|make)\b", re.IGNORECASE),
    "transform": re.compile(r"\b(?:convert|translate|transform|rewrite|rephrase|turn\s+into)\b", re.IGNORECASE),
}

# --- Step 5: Override detection ---
OVERRIDE_CONTENT = re.compile(
    r"\b(?:expand\s+(?:this|that|it|on\s+this)|more\s+detail|elaborate|go\s+deeper|dig\s+deeper)\b",
    re.IGNORECASE,
)
OVERRIDE_FORMAT = re.compile(
    r"\b(?:use\s+(?:paragraphs?|bullets?|table|numbered)|switch\s+(?:to|format))\b",
    re.IGNORECASE,
)
OVERRIDE_GLOBAL = re.compile(
    r"\b(?:give\s+me\s+everything|no\s+limits?|full\s+(?:detail|explanation)|completely|everything\s+about)\b",
    re.IGNORECASE,
)

# --- Step 8: Compressibility ---
# Concepts with established single-noun descriptors
COMPRESSIBLE_CONCEPTS = {
    "ai": "intelligence", "machine learning": "pattern-recognition",
    "blockchain": "ledger", "internet": "network", "gravity": "attraction",
    "photosynthesis": "energy-conversion", "evolution": "adaptation",
    "electricity": "charge-flow", "dna": "genetic-code",
}

# --- Step 15: Semantic loss signals (intent-gated) ---
LOSS_SIGNALS = [
    "why", "how", "compare", "relationship between",
    "process", "history of", "explain how", "difference between",
    "cause", "effect", "impact", "consequence",
]

# --- SOFT tone anti-patterns ---
SOFT_ANTI_PATTERNS = (
    "USER PREFERENCES (follow when possible):\n"
    "- Be direct, avoid lengthy preambles\n"
    "- Avoid anecdotes, long transitions, storytelling\n"
    "- Do not use filler qualifiers like \"Some people believe…\", \"It's worth noting…\", \"In many cases…\"\n"
    "- Do not use closers like \"In summary\", \"Overall\", \"To conclude\"\n"
    "- Avoid unnecessary qualifiers and hedging language\n"
)


# ============================================
# RESULT DATA STRUCTURES
# ============================================

class ImpossibilityTier:
    FITS = "FITS"
    TIGHT = "TIGHT"
    IMPOSSIBLE = "IMPOSSIBLE"


@dataclass
class ConstraintAnalysis:
    """Complete result of the 15-step pipeline (pre-generation steps 1-11)."""
    # Step 1
    resolved_query: str = ""
    reference_target: str = ""  # "assistant" | "user" | "topic" | ""

    # Steps 2-3
    strict_rules: Dict[str, Rule] = field(default_factory=dict)
    soft_rules: Dict[str, Rule] = field(default_factory=dict)

    # Step 4
    intent: str = "explain"

    # Step 5
    override_scope: Optional[OverrideScope] = None

    # Steps 8-9
    compressibility: str = ""  # "compressible" | "uncertain" | "incompressible"
    impossibility_tier: str = ImpossibilityTier.FITS

    # Step 10
    constraint_prompt: str = ""

    # Step 11
    use_streaming: bool = True  # False = single-shot

    # All active rules after merging
    active_rules: Dict[str, Rule] = field(default_factory=dict)

    def has_strict(self) -> bool:
        return any(r.is_strict() for r in self.active_rules.values())

    def log_summary(self) -> str:
        strict_count = sum(1 for r in self.active_rules.values() if r.is_strict())
        return (
            f"strict={strict_count > 0} mode={'single-shot' if not self.use_streaming else 'stream'} "
            f"tier={self.impossibility_tier} intent={self.intent}"
        )


# ============================================
# PIPELINE STEPS
# ============================================

def _step1_resolve_references(
    query: str,
    last_assistant_output: str,
    last_user_query: str,
) -> Tuple[str, str]:
    """
    Resolve 'this', 'that', 'it' to their referent.
    Priority: last assistant output > last user query.
    Freshness: last turn only.
    Returns (resolved_query, reference_target).
    """
    reference_words = re.compile(r"\b(?:this|that|it)\b", re.IGNORECASE)

    if not reference_words.search(query):
        return query, ""

    # Check if reference has a clear local subject (e.g., "explain this code")
    # If surrounding words provide context, don't flag as ambiguous
    has_local_subject = re.search(
        r"\b(?:this|that|it)\s+\w+", query, re.IGNORECASE
    )
    if has_local_subject:
        return query, ""

    if last_assistant_output:
        return query, "assistant"
    elif last_user_query:
        return query, "user"

    return query, ""


def _step2_detect_explicit_limits(query: str) -> Dict[str, Rule]:
    """Regex-based limit detection → STRICT."""
    rules = {}
    q_lower = query.lower()
    has_assertive = bool(STRICT_QUALIFIERS.search(query))

    # Count & length patterns
    for pattern, key in COUNT_PATTERNS:
        match = re.search(pattern, q_lower)
        if match:
            value = int(match.group(1))
            rules[key] = Rule(key=key, value=value, severity=Severity.STRICT)

    # Format patterns
    for pattern, key, value in FORMAT_PATTERNS:
        if re.search(pattern, q_lower):
            # Format is STRICT only with assertive qualifier
            sev = Severity.STRICT if has_assertive else Severity.SOFT
            rules[key] = Rule(key=key, value=value, severity=sev)

    # Explicit length keywords
    for pattern, key, value in LENGTH_KEYWORDS_STRICT:
        if re.search(pattern, q_lower):
            rules[key] = Rule(key=key, value=value, severity=Severity.STRICT)

    return rules


def _step3_infer_soft_preferences(query: str, explicit_keys: set) -> Dict[str, Rule]:
    """Inference layer → ALWAYS SOFT. Skips keys already detected explicitly."""
    rules = {}
    q_lower = query.lower()

    for pattern, (key, value) in SOFT_INFERENCE_MAP.items():
        if key in explicit_keys:
            continue
        if re.search(pattern, q_lower):
            rules[key] = Rule(key=key, value=value, severity=Severity.SOFT)

    return rules


def _step4_classify_intent(query: str) -> str:
    """Lightweight intent classifier, separate from constraints."""
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(query):
            return intent
    return "explain"  # Default


def _step5_detect_override(query: str) -> Optional[OverrideScope]:
    """Detect scoped override keywords."""
    if OVERRIDE_GLOBAL.search(query):
        return OverrideScope.GLOBAL
    if OVERRIDE_CONTENT.search(query):
        return OverrideScope.CONTENT
    if OVERRIDE_FORMAT.search(query):
        return OverrideScope.FORMAT
    return None


def _step6_confirmation_reset(
    user_id: str, chat_id: str, new_rules: Dict[str, Rule]
) -> bool:
    """Check if user re-affirmed any existing rules. Returns True if skippable."""
    active = get_active_rules(user_id, chat_id)
    reaffirmed = [k for k in new_rules if k in active]
    if reaffirmed:
        return confirmation_reset(user_id, chat_id, reaffirmed)
    return False


def _step8_check_compressibility(query: str, strict_rules: Dict[str, Rule]) -> str:
    """
    Domain-aware compressibility check.
    Uncertain → default TIGHT (fail toward usefulness).
    """
    # Only relevant for very tight constraints
    has_tight_length = False
    for rule in strict_rules.values():
        if rule.key in ("length_limit", "count_limit"):
            val = rule.value
            if isinstance(val, int) and val <= 2:
                has_tight_length = True
            elif isinstance(val, str) and val in ("1_line", "2_lines", "1_word"):
                has_tight_length = True

    if not has_tight_length:
        return "compressible"

    # Check known compressible concepts
    q_lower = query.lower()
    for concept in COMPRESSIBLE_CONCEPTS:
        if concept in q_lower:
            return "compressible"

    # Uncertain → TIGHT (not IMPOSSIBLE)
    return "uncertain"


def _step9_impossibility_gradient(
    compressibility: str, strict_rules: Dict[str, Rule]
) -> str:
    """
    FITS / TIGHT / IMPOSSIBLE.
    Uncertain compressibility → TIGHT.
    """
    if not strict_rules:
        return ImpossibilityTier.FITS

    has_extreme = False
    for rule in strict_rules.values():
        if rule.key == "length_limit" and rule.value == "1_word":
            has_extreme = True
        elif rule.key == "count_limit" and isinstance(rule.value, int) and rule.value <= 1:
            has_extreme = True

    if has_extreme and compressibility == "incompressible":
        return ImpossibilityTier.IMPOSSIBLE
    elif has_extreme or compressibility == "uncertain":
        return ImpossibilityTier.TIGHT
    else:
        return ImpossibilityTier.FITS


def _step10_build_constraint_prompt(
    active_rules: Dict[str, Rule],
    intent: str,
) -> str:
    """Build the constraint block for system prompt injection."""
    strict = {k: v for k, v in active_rules.items() if v.is_strict()}
    soft = {k: v for k, v in active_rules.items() if v.is_soft()}

    prompt_parts = []

    if strict:
        lines = ["STRICT USER CONSTRAINTS (OBEY EXACTLY):"]
        for key, rule in strict.items():
            if key == "count_limit":
                lines.append(f"- Maximum {rule.value} items/points")
            elif key == "length_limit":
                val = rule.value
                if isinstance(val, int):
                    lines.append(f"- Maximum {val} lines")
                elif val == "1_line":
                    lines.append("- Respond in exactly 1 line")
                elif val == "2_lines":
                    lines.append("- Respond in exactly 2 lines")
                elif val == "1_word":
                    lines.append("- Respond in exactly 1 word")
                else:
                    lines.append(f"- Keep response {val}")
            elif key == "format_preference":
                lines.append(f"- Use {rule.value} format")
            elif key == "depth_limit":
                lines.append(f"- Keep depth: {rule.value}")

        lines.append("- Do not expand beyond the requested scope")
        lines.append("- Do not add introductions or conclusions unless part of the structure")
        prompt_parts.append("\n".join(lines))

    if soft:
        prompt_parts.append(SOFT_ANTI_PATTERNS)
    elif not strict:
        # No constraints at all — return empty
        return ""

    # Always add the core safety footer
    prompt_parts.append(
        "Follow user limits strictly.\n"
        "Do not expand scope.\n"
        "Do not assume completeness."
    )

    return "\n\n".join(prompt_parts)


def _step11_decide_streaming(has_strict: bool) -> bool:
    """STRICT exists → False (single-shot). Otherwise → True (stream)."""
    return not has_strict


# ============================================
# MAIN PIPELINE ENTRY POINT
# ============================================

async def analyze_constraints(
    user_query: str,
    user_id: str,
    chat_id: str,
    context_messages: Optional[List[Dict]] = None,
) -> ConstraintAnalysis:
    """
    Run steps 1–11 of the Safe Chat Agent pipeline.
    Returns ConstraintAnalysis for the caller (processor.py).
    """
    result = ConstraintAnalysis()
    session = get_session(user_id, chat_id)

    # --- Step 1: Resolve references ---
    last_assistant = get_last_assistant_output(user_id, chat_id)
    last_user = session.last_query
    result.resolved_query, result.reference_target = _step1_resolve_references(
        user_query, last_assistant, last_user
    )

    # --- Step 2: Detect explicit limits → STRICT ---
    explicit_rules = _step2_detect_explicit_limits(user_query)

    # Check hedging: downgrade to SOFT if hedging present
    if HEDGING_PATTERNS.search(user_query):
        for key, rule in explicit_rules.items():
            rule.severity = Severity.SOFT

    result.strict_rules = {k: v for k, v in explicit_rules.items() if v.is_strict()}

    # --- Step 3: Infer soft preferences ---
    explicit_keys = set(explicit_rules.keys())
    soft_rules = _step3_infer_soft_preferences(user_query, explicit_keys)
    result.soft_rules = soft_rules

    # Merge all detected rules
    all_new_rules = {**explicit_rules, **soft_rules}

    # --- Step 4: Classify intent (SEPARATE from constraints) ---
    result.intent = _step4_classify_intent(user_query)

    # --- Step 5: Detect override scope ---
    result.override_scope = _step5_detect_override(user_query)
    if result.override_scope:
        apply_override(user_id, chat_id, result.override_scope)

    # --- Step 6: Confirmation reset ---
    was_confirmed = _step6_confirmation_reset(user_id, chat_id, all_new_rules)

    # --- Step 7: Apply rule decay ---
    query_embedding = None
    if session.has_any_rules() and not was_confirmed:
        # Only compute embedding if we have active rules and no confirmation
        try:
            from app.llm.embeddings import generate_embedding
            query_embedding = await generate_embedding(user_query)
        except Exception:
            pass

    apply_decay(user_id, chat_id, query_embedding)

    # --- Merge new rules into session ---
    if all_new_rules:
        update_rules(user_id, chat_id, all_new_rules)

    # Get final active rules
    result.active_rules = get_active_rules(user_id, chat_id)

    # --- Step 8: Check compressibility ---
    strict_active = {k: v for k, v in result.active_rules.items() if v.is_strict()}
    result.compressibility = _step8_check_compressibility(user_query, strict_active)

    # --- Step 9: Impossibility gradient ---
    result.impossibility_tier = _step9_impossibility_gradient(
        result.compressibility, strict_active
    )

    # --- Step 10: Build constraint prompt ---
    result.constraint_prompt = _step10_build_constraint_prompt(
        result.active_rules, result.intent
    )

    # --- Step 11: Decide streaming ---
    result.use_streaming = _step11_decide_streaming(result.has_strict())

    # --- Track STRICT rate ---
    strict_rate_tracker.record(result.has_strict())

    # --- Debug log ---
    if DEBUG_SAFE_AGENT:
        print(f"[SafeAgent] {result.log_summary()}")

    return result


def check_semantic_loss(query: str, intent: str) -> bool:
    """
    Intent-gated semantic loss check.
    Only returns True for explain/compare + loss signal words.
    """
    if intent not in ("explain", "compare"):
        return False
    q_lower = query.lower()
    return any(s in q_lower for s in LOSS_SIGNALS)
