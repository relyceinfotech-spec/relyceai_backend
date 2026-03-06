"""
Adaptive Strategy Memory Layer.
Tracks behavioral patterns across requests per session.
"""
from typing import Dict, Any, Optional
import json

# Session-scoped in-memory store. NO GLOBAL MEMORY.
_MEMORY_STORE: Dict[str, Dict[str, Any]] = {}

def _init_session_memory() -> Dict[str, Any]:
    """Creates a hard-structured dictionary for a new session."""
    return {
        "strategy_misclassifications": {}, # Dict[str, float]
        "unnecessary_research_count": 0.0, # float
        "repair_failures": {},             # Dict[str, float]
        "total_updates": 0                 # int
    }

def get_memory(session_id: str) -> Dict[str, Any]:
    """Retrieve or initialize memory for a given session."""
    if not session_id:
        return _init_session_memory() # Return ephemeral if no session
        
    if session_id not in _MEMORY_STORE:
        _MEMORY_STORE[session_id] = _init_session_memory()
    return _MEMORY_STORE[session_id]

def decay_memory(session_id: str) -> None:
    """Decays float counts by 0.95. Called only every N cycles."""
    mem = get_memory(session_id)
    
    # Decay misclassifications
    for k in mem["strategy_misclassifications"]:
        mem["strategy_misclassifications"][k] *= 0.95
        
    # Decay unnecessary research
    mem["unnecessary_research_count"] *= 0.95
    
    # Decay repair failures
    for k in mem["repair_failures"]:
        mem["repair_failures"][k] *= 0.95
        
    print(f"[Agent Memory] Decayed memory for session {session_id}")

def _calculate_token_overlap(research_text: str, output_text: str) -> float:
    """Calculates Jaccard similarity / token intersection ratio."""
    if not research_text or not output_text:
        return 0.0
        
    r_tokens = set(research_text.lower().split())
    o_tokens = set(output_text.lower().split())
    
    if not r_tokens:
        return 0.0
        
    return len(r_tokens.intersection(o_tokens)) / len(r_tokens)

def update_strategy_memory(session_id: str, outcome: Dict[str, Any]) -> None:
    """
    Called at the end of execution to update behavioral memory.
    outcome: {
        "strategy_mode": str,
        "confidence_drifted": bool,
        "research_used": bool,
        "research_text": str,
        "output_text": str,
        "repair_attempts": list[dict], # [{"type": str, "success": bool}]
    }
    """
    if not session_id:
        return
        
    from app.state.transaction_manager import is_memory_suppressed
    if is_memory_suppressed(session_id):
        print(f"[Agent Memory] Ignoring update for session {session_id} due to active rollback.")
        return
        
    mem = get_memory(session_id)
    mem["total_updates"] += 1
    
    # 1. Strategy Misclassifications
    if outcome.get("confidence_drifted"):
        mode = outcome.get("strategy_mode")
        if mode:
            mem["strategy_misclassifications"][mode] = mem["strategy_misclassifications"].get(mode, 0.0) + 1.0

    # 2. Unnecessary Research (Signal Check via Token Overlap)
    if outcome.get("research_used") and outcome.get("research_text"):
        r_text = outcome["research_text"]
        o_text = outcome.get("output_text", "")
        overlap_ratio = _calculate_token_overlap(r_text, o_text)
        
        r_tokens_len = len(set(r_text.lower().split()))
        
        # Jaccard Bias Bypass: Skip overlap penalty if research result is very short
        if r_tokens_len >= 50:
            # If less than 5% of research tokens made it into output, penalty.
            if overlap_ratio < 0.05:
                mem["unnecessary_research_count"] += 1.0

    # 3. Repair Patterns
    for repair in outcome.get("repair_attempts", []):
        r_type = repair.get("type", "UNKNOWN")
        if repair.get("success"):
            # Contextual Reset: Slowly forgive past failures if a rewrite succeeded
            if r_type in mem["repair_failures"]:
                mem["repair_failures"][r_type] *= 0.8
        else:
            mem["repair_failures"][r_type] = mem["repair_failures"].get(r_type, 0.0) + 1.0
            
    # Trace for debugging
    print(f'[INFO] {json.dumps({"memory_update": mem})}')

    # Decay scheduling (every 20 execution cycles per session)
    if mem["total_updates"] % 20 == 0:
        decay_memory(session_id)

def clear_session_memory(session_id: str) -> None:
    """Clears memory for testing purposes."""
    if session_id in _MEMORY_STORE:
        del _MEMORY_STORE[session_id]
