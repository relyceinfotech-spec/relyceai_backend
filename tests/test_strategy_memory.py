import pytest

from app.memory.strategy_memory import (
    get_memory,
    decay_memory,
    _calculate_token_overlap,
    update_strategy_memory,
    clear_session_memory
)
from app.agent.hybrid_controller import generate_strategy_advice

# ============================================
# MEMORY ISOLATION & BASIC TRACKING
# ============================================

def test_memory_isolation():
    clear_session_memory("session_A")
    clear_session_memory("session_B")
    
    mem_a = get_memory("session_A")
    mem_b = get_memory("session_B")
    
    mem_a["strategy_misclassifications"]["ADAPTIVE_CODE_PLAN"] = 5.0
    
    # Assert session B is entirely unaffected
    assert "ADAPTIVE_CODE_PLAN" not in mem_b["strategy_misclassifications"]
    assert mem_a["strategy_misclassifications"]["ADAPTIVE_CODE_PLAN"] == 5.0

def test_token_overlap_detection():
    # Exact match
    assert _calculate_token_overlap("hello world", "hello world output") > 0.5
    
    # Tiny overlap (< 5%)
    # 20 words in research, 1 word overlap
    research = "a b c d e f g h i j k l m n o p q r s t"
    output = "a completely unrelated output text without those letters"
    
    ratio = _calculate_token_overlap(research, output)
    assert ratio < 0.06

# ============================================
# OUTCOME HOOK & CONFIDENCE ADJUSTMENTS
# ============================================

def test_update_hook_and_confidence_floor():
    session = "test_confidence_floor"
    clear_session_memory(session)
    
    # 1. Simulate repeated confidence drifts
    for _ in range(5):
        update_strategy_memory(session, {
            "strategy_mode": "ADAPTIVE_CODE_PLAN",
            "confidence_drifted": True,
            "research_used": False,
        })
        
    mem = get_memory(session)
    assert mem["strategy_misclassifications"]["ADAPTIVE_CODE_PLAN"] == 5.0
    
    # 2. Check if hybrid controller drops confidence 
    # (Base is usually 1.0 for highly specific code commands)
    advice = generate_strategy_advice("write a python script", {}, session_id=session)
    
    # Base confidence for 'write' is 0.7. Misclassification drops it by 0.1. Thus 0.6.
    assert round(advice.strategy_confidence, 2) == 0.6
    
    # 3. Force memory to extremely high drift to test the floor (0.4)
    mem["strategy_misclassifications"]["ADAPTIVE_CODE_PLAN"] = 50.0
    mem["unnecessary_research_count"] = 50.0
    
    advice2 = generate_strategy_advice("write a python script", {}, session_id=session)
    assert advice2.strategy_confidence >= 0.4  # Core invariant: floor must protect from dropping to 0

def test_research_overuse_penalty():
    session = "test_research"
    clear_session_memory(session)
    
    # Send 6 completely ignored research payloads
    for _ in range(6):
        update_strategy_memory(session, {
            "strategy_mode": "RESEARCH_PLAN",
            "research_used": True,
            "research_text": " ".join(f"token_{i}" for i in range(100)),
            "output_text": "I am just going to write the code from memory using zero context whatsoever"
        })
        
    mem = get_memory(session)
    assert mem["unnecessary_research_count"] >= 6.0
    
    advice = generate_strategy_advice("how to use fastAPI", {}, session_id=session)
    
    # Research overused > 5 times, research confidence should drop by 0.15
    assert advice.research_confidence < 1.0
    

# ============================================
# CONTEXTUAL REPAIR & DECAY
# ============================================

def test_repair_pivot_and_reset():
    session = "test_repair"
    clear_session_memory(session)
    
    # Record 4 FAILURES
    for _ in range(4):
        update_strategy_memory(session, {
            "repair_attempts": [{"type": "FIX_CODE", "success": False}]
        })
        
    mem = get_memory(session)
    assert mem["repair_failures"]["FIX_CODE"] == 4.0
    
    # Let's verify the repair generation pivot
    from app.agent.repair_engine import generate_repair_strategy
    context = {"repair_history": ["FIX_CODE"], "session_id": session}
    
    strategy = generate_repair_strategy("SyntaxError", context)
    assert "pivot your strategy completely" in strategy["repair_prompt"]
    
    # Record a SUCCESS (Contextual resetting logic)
    update_strategy_memory(session, {
        "repair_attempts": [{"type": "FIX_CODE", "success": True}]
    })
    
    # 4.0 * 0.8 = 3.2
    assert mem["repair_failures"]["FIX_CODE"] == 3.2

def test_scheduled_decay():
    session = "test_decay"
    clear_session_memory(session)
    
    mem = get_memory(session)
    mem["strategy_misclassifications"]["TEST"] = 10.0
    mem["unnecessary_research_count"] = 10.0
    mem["repair_failures"]["BUG"] = 10.0
    mem["total_updates"] = 19
    
    # The 20th update should fire decay logic
    update_strategy_memory(session, {})
    
    assert mem["strategy_misclassifications"]["TEST"] == 9.5
    assert mem["unnecessary_research_count"] == 9.5
    assert mem["repair_failures"]["BUG"] == 9.5
