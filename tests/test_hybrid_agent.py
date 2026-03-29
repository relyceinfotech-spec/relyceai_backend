"""
Hybrid Strategic Agent — Automated Test Script
Tests strategy engine, hybrid controller (advisory), static validator, and repair engine.
Run: python -m pytest test_hybrid_agent.py -v
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.agent.strategy_engine import (
    detect_intent,
    select_planning_mode,
    build_parallel_reasoning,
    merge_reasoning_tracks,
    get_adaptive_code_steps,
    get_sequential_steps,
    IntentSignals,
)
from app.agent.hybrid_controller import (
    generate_strategy_advice,
    HybridAdvice,
)
from app.agent.static_validator import (
    static_validation,
    contains_dangerous_ops,
    get_dangerous_ops,
)
from app.agent.repair_engine import (
    classify_failure,
    generate_repair_strategy,
    repair_cycle,
    build_repair_prompt,
    FailureType,
    MAX_REPAIR_ATTEMPTS,
)


# ============================================
# INTENT DETECTION
# ============================================

class TestIntentDetection:
    def test_code_task(self):
        intent = detect_intent("Write a Python function to sort a list")
        assert intent.is_code_generation is True

    def test_code_task_with_patterns(self):
        intent = detect_intent("Fix this: def calculate(x):")
        assert intent.is_code_generation is True

    def test_comparison(self):
        intent = detect_intent("Compare React vs Vue")
        assert intent.is_comparison is True

    def test_tool_needed(self):
        intent = detect_intent("Search for the latest news on AI")
        assert intent.requires_tools is True

    def test_generation(self):
        intent = detect_intent("Write a poem about the ocean")
        assert intent.is_generation is True

    def test_exploratory(self):
        intent = detect_intent("Explain in detail how neural networks work")
        assert intent.is_exploratory is True

    def test_simple_question(self):
        intent = detect_intent("What is 2 + 2?")
        assert intent.is_code_generation is False
        assert intent.is_comparison is False
        assert intent.requires_tools is False

    def test_mixed_signals(self):
        intent = detect_intent("Compare Python vs JavaScript for building APIs")
        assert intent.is_comparison is True
        assert intent.is_code_generation is False  # "Compare" is not a code generation command
        assert intent.has_tech_domain is True      # But it does flag the tech domain


# ============================================
# PLANNING MODE SELECTION
# ============================================

class TestPlanningModeSelection:
    def test_code_task_gets_adaptive(self):
        intent = IntentSignals(is_code_generation=True)
        assert select_planning_mode(intent) == "ADAPTIVE_CODE_PLAN"

    def test_tools_get_sequential(self):
        intent = IntentSignals(requires_tools=True)
        assert select_planning_mode(intent) == "SEQUENTIAL_PLAN"

    def test_comparison_gets_parallel(self):
        intent = IntentSignals(is_comparison=True)
        assert select_planning_mode(intent) == "PARALLEL_REASONING"

    def test_generation_gets_parallel(self):
        intent = IntentSignals(is_generation=True)
        assert select_planning_mode(intent) == "PARALLEL_REASONING"

    def test_exploratory_gets_parallel(self):
        intent = IntentSignals(is_exploratory=True)
        assert select_planning_mode(intent) == "PARALLEL_REASONING"

    def test_default_is_sequential(self):
        intent = IntentSignals()
        assert select_planning_mode(intent) == "SEQUENTIAL_PLAN"

    def test_code_overrides_comparison(self):
        """Code task takes priority even if comparison is also detected."""
        intent = IntentSignals(is_code_generation=True, is_comparison=True)
        assert select_planning_mode(intent) == "ADAPTIVE_CODE_PLAN"

    def test_tools_override_generation(self):
        """Tool need takes priority over generation."""
        intent = IntentSignals(requires_tools=True, is_generation=True)
        assert select_planning_mode(intent) == "SEQUENTIAL_PLAN"


# ============================================
# PARALLEL REASONING
# ============================================

class TestParallelReasoning:
    def test_always_includes_structure_and_alternatives(self):
        tracks = build_parallel_reasoning("Compare apples and oranges")
        assert "structure" in tracks
        assert "alternatives" in tracks

    def test_comparison_gets_all_tracks(self):
        tracks = build_parallel_reasoning("Compare React vs Vue")
        assert "performance" in tracks
        assert "structure" in tracks
        assert "risk" in tracks
        assert "alternatives" in tracks

    def test_technical_gets_performance(self):
        tracks = build_parallel_reasoning("How to optimize database performance")
        assert "performance" in tracks

    def test_merge_produces_content(self):
        tracks = build_parallel_reasoning("Compare React vs Vue")
        merged = merge_reasoning_tracks(tracks)
        assert "MULTI-DIMENSIONAL REASONING CONTEXT" in merged
        assert len(merged) > 100

    def test_empty_merge(self):
        assert merge_reasoning_tracks({}) == ""


# ============================================
# HYBRID CONTROLLER (ADVISORY ONLY)
# ============================================

class TestHybridAdvice:
    def test_comparison_produces_parallel(self):
        advice = generate_strategy_advice("Compare electric cars vs hybrid cars", {})
        assert advice.planning_mode == "PARALLEL_REASONING"
        assert advice.validation_required is False
        assert advice.repair_policy["enabled"] is False

    def test_code_produces_adaptive(self):
        advice = generate_strategy_advice("Write a Python function to sort", {})
        assert advice.planning_mode == "ADAPTIVE_CODE_PLAN"
        assert advice.validation_required is True
        assert advice.repair_policy["enabled"] is True
        assert advice.repair_policy["max_attempts"] == 2

    def test_tool_query_produces_sequential(self):
        advice = generate_strategy_advice("Search for today's weather", {})
        assert advice.planning_mode == "SEQUENTIAL_PLAN"

    def test_simple_query_produces_sequential(self):
        advice = generate_strategy_advice("What is 2+2?", {})
        assert advice.planning_mode == "SEQUENTIAL_PLAN"

    def test_intent_signals_attached(self):
        advice = generate_strategy_advice("Compare React vs Vue", {})
        assert advice.intent_signals is not None
        assert advice.intent_signals.is_comparison is True

    def test_parallel_has_reasoning_context(self):
        advice = generate_strategy_advice("Compare electric cars vs diesel for production", {})
        assert len(advice.reasoning_context) > 0

    def test_sequential_has_no_reasoning_context(self):
        advice = generate_strategy_advice("What time is it?", {})
        assert len(advice.reasoning_context) == 0


# ============================================
# STATIC VALIDATOR
# ============================================

class TestStaticValidator:
    def test_safe_code_passes(self):
        ctx = {"generated_code": "def hello():\n    return 'world'"}
        assert static_validation(ctx) == "validation_passed"

    def test_eval_detected(self):
        ctx = {"generated_code": "result = eval(user_input)"}
        assert static_validation(ctx) == "unsafe_code"
        assert ctx["risk_flag"] is True

    def test_exec_detected(self):
        ctx = {"generated_code": "exec(code_string)"}
        assert static_validation(ctx) == "unsafe_code"

    def test_os_system_detected(self):
        ctx = {"generated_code": "import os\nos.system('rm -rf /')"}
        assert static_validation(ctx) == "unsafe_code"

    def test_subprocess_detected(self):
        ctx = {"generated_code": "subprocess.run(['ls'])"}
        assert static_validation(ctx) == "unsafe_code"

    def test_dynamic_import_detected(self):
        ctx = {"generated_code": "__import__('os')"}
        assert static_validation(ctx) == "unsafe_code"

    def test_filesystem_delete_detected(self):
        ctx = {"generated_code": "shutil.rmtree('/tmp/data')"}
        assert static_validation(ctx) == "unsafe_code"

    def test_empty_code_fails(self):
        ctx = {"generated_code": ""}
        assert static_validation(ctx) == "validation_failed"

    def test_no_code_fails(self):
        ctx = {}
        assert static_validation(ctx) == "validation_failed"

    def test_dangerous_ops_detail(self):
        code = "eval(x)\nos.system('cmd')"
        ops = get_dangerous_ops(code)
        names = [o["name"] for o in ops]
        assert "call_eval" in names
        assert "call_method_system" in names

    def test_contains_dangerous_empty(self):
        assert contains_dangerous_ops("") is False
        assert contains_dangerous_ops("   ") is False


# ============================================
# FAILURE CLASSIFICATION
# ============================================

class TestFailureClassification:
    def test_syntax_error(self):
        assert classify_failure("SyntaxError: invalid syntax") == FailureType.FIX_CODE

    def test_indentation_error(self):
        assert classify_failure("IndentationError: unexpected indent") == FailureType.FIX_CODE

    def test_module_not_found(self):
        assert classify_failure("ModuleNotFoundError: No module named 'flask'") == FailureType.INSTALL_OR_REPLACE

    def test_import_error(self):
        assert classify_failure("ImportError: cannot import name 'foo'") == FailureType.INSTALL_OR_REPLACE

    def test_type_error(self):
        assert classify_failure("TypeError: unsupported operand type") == FailureType.REWRITE_FUNCTION

    def test_key_error(self):
        assert classify_failure("KeyError: 'missing_key'") == FailureType.REWRITE_FUNCTION

    def test_version_conflict(self):
        assert classify_failure("DeprecationWarning: function is deprecated") == FailureType.ADAPT_LIBRARY

    def test_unknown_error(self):
        assert classify_failure("Something weird happened") == FailureType.UNKNOWN

    def test_empty_error(self):
        assert classify_failure("") == FailureType.UNKNOWN


# ============================================
# REPAIR CYCLE
# ============================================

class TestRepairCycle:
    def test_no_failure_no_repair(self):
        result = repair_cycle(failure="", context={})
        assert result.status == "no_repair_needed"

    def test_syntax_failure_needs_repair(self):
        result = repair_cycle(
            failure="SyntaxError: invalid syntax",
            context={},
            max_attempts=2,
        )
        assert result.status == "repair_needed"
        assert result.final_failure_type == FailureType.FIX_CODE

    def test_max_attempts_capped(self):
        """Even if caller passes high max_attempts, it's capped at MAX_REPAIR_ATTEMPTS."""
        result = repair_cycle(
            failure="TypeError: bad type",
            context={},
            max_attempts=100,
        )
        assert result.status == "repair_needed"
        # Check context was set correctly
        ctx = {}
        repair_cycle(failure="TypeError: bad type", context=ctx, max_attempts=100)
        assert ctx["repair_max_attempts"] == MAX_REPAIR_ATTEMPTS

    def test_repair_strategy_generated(self):
        strat = generate_repair_strategy("SyntaxError: invalid syntax")
        assert strat["failure_type"] == FailureType.FIX_CODE
        assert "syntax" in strat["repair_prompt"].lower()
        assert strat["original_error"] == "SyntaxError: invalid syntax"

    def test_repair_prompt_built(self):
        strategy = {
            "failure_type": "FIX_CODE",
            "repair_prompt": "Fix the syntax.",
        }
        prompt = build_repair_prompt(
            original_code="def foo(:\n    pass",
            error="SyntaxError: invalid syntax",
            repair_strategy=strategy,
        )
        assert "ERROR TYPE" in prompt
        assert "FIX_CODE" in prompt
        assert "SyntaxError" in prompt
        assert "def foo" in prompt


# ============================================
# PLAN STEPS
# ============================================

class TestPlanSteps:
    def test_adaptive_steps(self):
        steps = get_adaptive_code_steps()
        assert steps[0] == "understand_requirements"
        assert "static_validation" in steps
        assert "repair_cycle" in steps
        assert steps[-1] == "final_verification"
        assert len(steps) == 8

    def test_sequential_steps(self):
        steps = get_sequential_steps()
        assert steps[0] == "analyze_request"
        assert steps[-1] == "verify_result"
        assert len(steps) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
