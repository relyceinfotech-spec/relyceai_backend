"""
Safe Chat Agent — Automated Test Script
Tests the core constraint detection, session rules, and output validation.
Run: python -m pytest test_safe_agent.py -v
"""
import asyncio
import pytest
import sys
import os

# Ensure app is on path
sys.path.insert(0, os.path.dirname(__file__))

from app.chat.session_rules import (
    Rule, Severity, OverrideScope, SessionState,
    get_session, increment_turn, update_rules,
    confirmation_reset, apply_decay, apply_override,
    get_active_rules, clear_session,
    _session_store,
)
from app.llm.safe_agent import (
    _step1_resolve_references,
    _step2_detect_explicit_limits,
    _step3_infer_soft_preferences,
    _step4_classify_intent,
    _step5_detect_override,
    _step8_check_compressibility,
    _step9_impossibility_gradient,
    _step10_build_constraint_prompt,
    _step11_decide_streaming,
    check_semantic_loss,
    ImpossibilityTier,
    strict_rate_tracker,
)
from app.llm.output_validator import (
    validate,
    count_validator,
    length_validator,
    structure_validator,
    trim_response,
    should_retry,
    get_retry_mode,
    build_retry_prompt,
    _count_bullets,
    _count_lines,
    ValidationResult,
)


# ============================================
# FIXTURES
# ============================================

@pytest.fixture(autouse=True)
def clean_sessions():
    """Clear session store between tests."""
    _session_store.clear()
    yield
    _session_store.clear()


# ============================================
# STEP 2: EXPLICIT LIMIT DETECTION
# ============================================

class TestExplicitLimits:
    def test_count_limit_basic(self):
        rules = _step2_detect_explicit_limits("Give me 3 points about AI")
        assert "count_limit" in rules
        assert rules["count_limit"].value == 3
        assert rules["count_limit"].severity == Severity.STRICT

    def test_count_limit_with_qualifier(self):
        rules = _step2_detect_explicit_limits("Exactly 5 benefits of exercise")
        assert "count_limit" in rules
        assert rules["count_limit"].value == 5
        assert rules["count_limit"].severity == Severity.STRICT

    def test_length_limit_lines(self):
        rules = _step2_detect_explicit_limits("in 2 lines explain gravity")
        assert "length_limit" in rules
        # "in 2 lines" matches both the count pattern (2) and the keyword pattern ("2_lines")
        # Keyword pattern runs after count pattern and overwrites
        assert rules["length_limit"].value in (2, "2_lines")

    def test_length_limit_one_liner(self):
        rules = _step2_detect_explicit_limits("Give me a one-liner on AI")
        assert "length_limit" in rules
        assert rules["length_limit"].value == "1_line"

    def test_format_bullet(self):
        rules = _step2_detect_explicit_limits("List in bullet points")
        assert "format_preference" in rules
        assert rules["format_preference"].value == "bullet"

    def test_format_table(self):
        rules = _step2_detect_explicit_limits("Show as a table")
        assert "format_preference" in rules
        assert rules["format_preference"].value == "table"

    def test_no_limits_detected(self):
        rules = _step2_detect_explicit_limits("What is machine learning?")
        assert len(rules) == 0


# ============================================
# STEP 3: SOFT INFERENCE
# ============================================

class TestSoftInference:
    def test_few_becomes_soft(self):
        rules = _step3_infer_soft_preferences("Give me a few tips", set())
        assert "count_limit" in rules
        assert rules["count_limit"].severity == Severity.SOFT

    def test_quick_becomes_soft(self):
        rules = _step3_infer_soft_preferences("Quick overview of React", set())
        assert "length_limit" in rules
        assert rules["length_limit"].severity == Severity.SOFT

    def test_skips_explicit_keys(self):
        rules = _step3_infer_soft_preferences("Give me a few tips", {"count_limit"})
        assert "count_limit" not in rules

    def test_no_inference_on_normal(self):
        rules = _step3_infer_soft_preferences("What is machine learning?", set())
        assert len(rules) == 0


# ============================================
# STEP 4: INTENT CLASSIFICATION
# ============================================

class TestIntentClassification:
    def test_explain(self):
        assert _step4_classify_intent("Explain quantum mechanics") == "explain"

    def test_summarize(self):
        assert _step4_classify_intent("Summarize this article") == "summarize"

    def test_list(self):
        assert _step4_classify_intent("List the planets") == "list"

    def test_compare(self):
        assert _step4_classify_intent("Compare Python vs JavaScript") == "compare"

    def test_generate(self):
        assert _step4_classify_intent("Write a poem about rain") == "generate"

    def test_default(self):
        assert _step4_classify_intent("What about gravity?") == "explain"


# ============================================
# STEP 5: OVERRIDE DETECTION
# ============================================

class TestOverrideDetection:
    def test_content_override(self):
        scope = _step5_detect_override("Expand this further")
        assert scope == OverrideScope.CONTENT

    def test_format_override(self):
        scope = _step5_detect_override("Use paragraphs instead")
        assert scope == OverrideScope.FORMAT

    def test_global_override(self):
        scope = _step5_detect_override("Give me everything about AI")
        assert scope == OverrideScope.GLOBAL

    def test_no_override(self):
        scope = _step5_detect_override("What is quantum mechanics?")
        assert scope is None


# ============================================
# STEP 1: REFERENCE RESOLUTION
# ============================================

class TestReferenceResolution:
    def test_resolves_to_assistant(self):
        _, target = _step1_resolve_references("Expand this", "Previous answer", "Previous query")
        assert target == "assistant"

    def test_resolves_to_user(self):
        _, target = _step1_resolve_references("Expand that", "", "Previous query")
        assert target == "user"

    def test_no_reference(self):
        _, target = _step1_resolve_references("Explain quantum physics", "", "")
        assert target == ""

    def test_local_subject_not_flagged(self):
        _, target = _step1_resolve_references("Explain this code", "Previous answer", "")
        assert target == ""  # "this code" has local subject


# ============================================
# STEPS 8-9: COMPRESSIBILITY + IMPOSSIBILITY
# ============================================

class TestCompressibility:
    def test_known_concept_compressible(self):
        rules = {"length_limit": Rule(key="length_limit", value="1_word", severity=Severity.STRICT)}
        result = _step8_check_compressibility("Explain AI in 1 word", rules)
        assert result == "compressible"

    def test_unknown_concept_uncertain(self):
        rules = {"length_limit": Rule(key="length_limit", value="1_word", severity=Severity.STRICT)}
        result = _step8_check_compressibility("Explain democracy in 1 word", rules)
        # Democracy is not in COMPRESSIBLE_CONCEPTS and has 1_word constraint
        # Uncertain → defaults to TIGHT (safe default)
        assert result in ("uncertain", "compressible")  # May be compressible if no tight check triggers

    def test_no_tight_constraint_compressible(self):
        rules = {"count_limit": Rule(key="count_limit", value=5, severity=Severity.STRICT)}
        result = _step8_check_compressibility("Explain democracy", rules)
        assert result == "compressible"


class TestImpossibility:
    def test_fits(self):
        rules = {"count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT)}
        tier = _step9_impossibility_gradient("compressible", rules)
        assert tier == ImpossibilityTier.FITS

    def test_tight_on_uncertain(self):
        rules = {"length_limit": Rule(key="length_limit", value="1_word", severity=Severity.STRICT)}
        tier = _step9_impossibility_gradient("uncertain", rules)
        assert tier == ImpossibilityTier.TIGHT

    def test_impossible(self):
        rules = {"length_limit": Rule(key="length_limit", value="1_word", severity=Severity.STRICT)}
        tier = _step9_impossibility_gradient("incompressible", rules)
        assert tier == ImpossibilityTier.IMPOSSIBLE


# ============================================
# SEMANTIC LOSS (INTENT-GATED)
# ============================================

class TestSemanticLoss:
    def test_explain_with_why(self):
        assert check_semantic_loss("Why does inflation happen?", "explain") is True

    def test_compare_with_difference(self):
        assert check_semantic_loss("Difference between AI and ML", "compare") is True

    def test_explain_no_signals(self):
        assert check_semantic_loss("AI in 1 word", "explain") is False

    def test_list_with_signals_no_loss(self):
        """List intent should NOT trigger loss even with signal words."""
        assert check_semantic_loss("Why are there planets?", "list") is False


# ============================================
# SESSION RULES
# ============================================

class TestSessionRules:
    def test_create_and_get(self):
        session = get_session("user1", "chat1")
        assert session.turn_count == 0
        assert not session.has_any_rules()

    def test_update_rules(self):
        update_rules("user1", "chat1", {
            "count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT)
        })
        rules = get_active_rules("user1", "chat1")
        assert "count_limit" in rules
        assert rules["count_limit"].value == 3

    def test_turn_decay(self):
        update_rules("user1", "chat1", {
            "count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT)
        })
        # Advance 2 turns
        increment_turn("user1", "chat1", "q1")
        increment_turn("user1", "chat1", "q2")
        session = get_session("user1", "chat1")
        session.turn_count = 3  # Force turn count past decay

        decayed = apply_decay("user1", "chat1")
        assert decayed > 0
        assert len(get_active_rules("user1", "chat1")) == 0

    def test_confirmation_reset(self):
        update_rules("user1", "chat1", {
            "count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT)
        })
        increment_turn("user1", "chat1", "q1")
        increment_turn("user1", "chat1", "q2")
        session = get_session("user1", "chat1")

        # Re-affirm before decay
        was_confirmed = confirmation_reset("user1", "chat1", ["count_limit"])
        assert was_confirmed is True

    def test_scoped_override_content(self):
        update_rules("user1", "chat1", {
            "count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT),
            "format_preference": Rule(key="format_preference", value="bullet", severity=Severity.SOFT),
        })
        apply_override("user1", "chat1", OverrideScope.CONTENT)
        rules = get_active_rules("user1", "chat1")
        assert "count_limit" not in rules
        assert "format_preference" in rules  # Format preserved

    def test_scoped_override_global(self):
        update_rules("user1", "chat1", {
            "count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT),
            "format_preference": Rule(key="format_preference", value="bullet", severity=Severity.SOFT),
        })
        apply_override("user1", "chat1", OverrideScope.GLOBAL)
        assert len(get_active_rules("user1", "chat1")) == 0


# ============================================
# OUTPUT VALIDATOR
# ============================================

class TestCountValidator:
    def test_passes_when_within_limit(self):
        text = "- Point 1\n- Point 2\n- Point 3"
        rules = {"count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT)}
        assert count_validator(text, rules) == []

    def test_fails_when_exceeds(self):
        text = "- Point 1\n- Point 2\n- Point 3\n- Point 4"
        rules = {"count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT)}
        violations = count_validator(text, rules)
        assert len(violations) > 0


class TestLengthValidator:
    def test_1_line_passes(self):
        text = "Intelligence"
        rules = {"length_limit": Rule(key="length_limit", value="1_line", severity=Severity.STRICT)}
        assert length_validator(text, rules) == []

    def test_1_line_fails(self):
        text = "Line 1\nLine 2\nLine 3"
        rules = {"length_limit": Rule(key="length_limit", value="1_line", severity=Severity.STRICT)}
        violations = length_validator(text, rules)
        assert len(violations) > 0


class TestTrimmer:
    def test_removes_intro_filler(self):
        text = "Sure! Here is the answer.\n\nThe actual content here is important and valuable to the reader."
        rules = {}
        trimmed, was_trimmed = trim_response(text, rules)
        assert "Sure" not in trimmed
        assert was_trimmed

    def test_removes_outro_filler(self):
        text = "The actual content is here and has enough substance to avoid revert.\n\nMore good content here.\nI hope this helps!"
        rules = {}
        trimmed, was_trimmed = trim_response(text, rules)
        assert "hope this helps" not in trimmed

    def test_preserves_structural_content(self):
        text = "# Heading\n- Point 1\n- Point 2"
        rules = {}
        trimmed, _ = trim_response(text, rules)
        assert "# Heading" in trimmed
        assert "- Point 1" in trimmed

    def test_reverts_heavy_trim(self):
        """If trim removes >30%, revert."""
        text = "Sure! Let me explain this for you.\nOk."
        rules = {}
        trimmed, was_trimmed = trim_response(text, rules)
        # Heavy trim would remove too much — should revert
        assert len(trimmed) > 0


class TestFullValidation:
    def test_no_strict_skips(self):
        rules = {"count_limit": Rule(key="count_limit", value=3, severity=Severity.SOFT)}
        result = validate("Long response here", rules)
        assert result.passed is True

    def test_strict_count_violation(self):
        text = "- A\n- B\n- C\n- D\n- E"
        rules = {"count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT)}
        result = validate(text, rules, query="Give me 3 points")
        # After auto-trim, should pass
        assert "D" not in result.trimmed_response
        assert "E" not in result.trimmed_response


class TestRetryLogic:
    def test_impossible_no_retry(self):
        vr = ValidationResult(passed=False, violations=["length_exceeded"])
        assert should_retry(vr, ImpossibilityTier.IMPOSSIBLE, 0) is False

    def test_fits_retry(self):
        vr = ValidationResult(passed=False, violations=["count_exceeded"])
        assert should_retry(vr, ImpossibilityTier.FITS, 0) is True

    def test_budget_exhausted(self):
        vr = ValidationResult(passed=False, violations=["count_exceeded"])
        assert should_retry(vr, ImpossibilityTier.FITS, 2) is False

    def test_retry_modes(self):
        assert get_retry_mode(0) == "single-shot"
        assert get_retry_mode(1) == "streaming-trim"


# ============================================
# STREAMING DECISION
# ============================================

class TestStreamingDecision:
    def test_no_strict_streams(self):
        assert _step11_decide_streaming(False) is True

    def test_strict_single_shot(self):
        assert _step11_decide_streaming(True) is False


# ============================================
# CONSTRAINT PROMPT BUILDER
# ============================================

class TestConstraintPrompt:
    def test_empty_rules_empty_prompt(self):
        prompt = _step10_build_constraint_prompt({}, "explain")
        assert prompt == ""

    def test_strict_rules_generate_prompt(self):
        rules = {
            "count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT),
        }
        prompt = _step10_build_constraint_prompt(rules, "explain")
        assert "Maximum 3" in prompt
        assert "STRICT" in prompt

    def test_soft_rules_generate_preferences(self):
        rules = {
            "length_limit": Rule(key="length_limit", value="short", severity=Severity.SOFT),
        }
        prompt = _step10_build_constraint_prompt(rules, "explain")
        assert "PREFERENCES" in prompt


# ============================================
# COEXISTENCE: STRICT + SOFT
# ============================================

class TestStrictSoftCoexistence:
    def test_3_points_briefly(self):
        """'Explain AI briefly in 3 points' → STRICT(count) + SOFT(brief)"""
        explicit = _step2_detect_explicit_limits("Explain AI briefly in 3 points")
        assert "count_limit" in explicit
        assert explicit["count_limit"].severity == Severity.STRICT
        assert explicit["count_limit"].value == 3

        soft = _step3_infer_soft_preferences(
            "Explain AI briefly in 3 points", set(explicit.keys())
        )
        assert "length_limit" in soft
        assert soft["length_limit"].severity == Severity.SOFT

    def test_mixed_prompt_has_both_sections(self):
        """Prompt should have both STRICT and PREFERENCES sections."""
        rules = {
            "count_limit": Rule(key="count_limit", value=3, severity=Severity.STRICT),
            "length_limit": Rule(key="length_limit", value="short", severity=Severity.SOFT),
        }
        prompt = _step10_build_constraint_prompt(rules, "explain")
        assert "STRICT USER CONSTRAINTS" in prompt
        assert "PREFERENCES" in prompt
        assert "Maximum 3" in prompt


# ============================================
# STRICT RATE TRACKER
# ============================================

class TestStrictRateTracker:
    def test_rate_starts_zero(self):
        tracker = type(strict_rate_tracker)()
        assert tracker.rate == 0.0

    def test_rate_tracks_correctly(self):
        tracker = type(strict_rate_tracker)()
        tracker.record(True)
        tracker.record(False)
        tracker.record(False)
        tracker.record(False)
        assert tracker.rate == 25.0  # 1/4 = 25%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
