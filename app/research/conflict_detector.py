"""
Conflict Detection for Research Results.
Scans chunks for contradicting concepts.
"""

def detect_conflicts(summaries: list[str]) -> bool:
    """
    Basic version extracts contradictions like stable vs beta,
    deprecated vs recommended, etc.
    """
    if not summaries:
        return False
        
    text_corpus = " ".join(summaries).lower()
    
    # Conflict patterns
    has_positive = "stable" in text_corpus or "recommended" in text_corpus or "production" in text_corpus
    has_beta = "beta" in text_corpus or "experimental" in text_corpus or "deprecated" in text_corpus
    
    # This is a basic semantic clash detector.
    # E.g. one source says API is stable, another says it's deprecated.
    if has_positive and has_beta:
        # To avoid false positives (like "this beta is stable"), 
        # we check if they appear in different summaries.
        stable_count = sum(1 for s in summaries if "stable" in s.lower() or "production" in s.lower())
        beta_count = sum(1 for s in summaries if "beta" in s.lower() or "deprecated" in s.lower())
        
        if stable_count > 0 and beta_count > 0 and stable_count != len(summaries):
            return True
            
    return False

def adjust_confidence_for_conflicts(confidence: float, has_conflict: bool) -> float:
    """Lower the agent's research confidence if contradictions exist."""
    if has_conflict:
        return max(0.0, confidence - 0.2)
    return confidence
