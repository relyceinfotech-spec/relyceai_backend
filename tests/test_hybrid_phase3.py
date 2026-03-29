import pytest
from app.agent.strategy_engine import detect_intent, select_planning_mode, IntentSignals
from app.agent.static_validator import static_validation
from app.agent.repair_engine import repair_cycle

def test_priority_reordering_comparison_vs_code():
    # "Compare React vs Vue" - React/Vue are tech entities, but 'compare' makes it comparison
    intent = detect_intent("compare React vs Vue")
    assert intent.is_comparison is True
    assert intent.has_tech_domain is True
    assert intent.is_code_generation is False
    
    mode = select_planning_mode(intent)
    assert mode == "PARALLEL_REASONING"  # Comparison beats tech entity

def test_priority_reordering_explicit_code():
    # "Write a React component"
    intent = detect_intent("Write a React component")
    assert intent.is_code_generation is True
    
    mode = select_planning_mode(intent)
    assert mode == "ADAPTIVE_CODE_PLAN"

def test_ambiguous_mixed_intent_with_confidence():
    # "Explain how React hooks work and write an example"
    intent = detect_intent("Explain how React hooks work and write an example")
    # explain -> exploratory
    assert intent.is_exploratory is True
    # React -> tech domain
    assert intent.has_tech_domain is True
    # write -> code generation
    assert intent.is_code_generation is True
    
    # Confidence should drop because multiple intents match
    assert intent.strategy_confidence < 0.9
    
    mode = select_planning_mode(intent)
    assert mode == "ADAPTIVE_CODE_PLAN" # Code takes absolute top priority

def test_ast_validation_bypass_prevention():
    # Regex bypass attempt: using getattr to call os.system dynamically
    malicious_code = '''
import os
getattr(os, "sy" + "stem")("echo hacked")
'''
    # The AST validator flags 'import os'
    ctx = {"generated_code": malicious_code}
    res = static_validation(ctx)
    assert res == "unsafe_code"
    
    ops = [op["name"] for op in ctx["dangerous_ops"]]
    assert "import_os" in ops

def test_ast_validation_file_write_out_of_bounds():
    # Using open in python
    malicious_code = '''
with open("/etc/passwd", "w") as f:
    f.write("hacked")
'''
    ctx = {"generated_code": malicious_code}
    res = static_validation(ctx)
    assert res == "unsafe_code"
    
    ops = [op["name"] for op in ctx["dangerous_ops"]]
    assert "call_open" in ops

def test_repair_loop_diversification():
    # Simulate a syntax error failure twice
    failure = "SyntaxError: invalid syntax"
    ctx = {}
    
    # First attempt
    res1 = repair_cycle(failure, ctx)
    assert res1.final_failure_type == "FIX_CODE"
    assert "WARNING: You attempted" not in ctx["repair_strategy"]["repair_prompt"]
    
    # Second attempt (same failure)
    res2 = repair_cycle(failure, ctx)
    assert res2.final_failure_type == "FIX_CODE"
    assert "WARNING: You attempted" in ctx["repair_strategy"]["repair_prompt"]
