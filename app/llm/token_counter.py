"""
Simple token approximation utility.
Used to evaluate string length against context thresholds without heavy tokenizer loading.
"""

def estimate_tokens(messages: list[dict], summary: str = "") -> int:
    """
    Produce a rough token estimate based on message and summary content.
    Generally, 1 token ≈ 4 English characters.
    """
    total = len(summary or "") // 4
    
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
            
    # Add minimal prompt overhead (system prompt + spacing)
    total += 150 
    
    return total
