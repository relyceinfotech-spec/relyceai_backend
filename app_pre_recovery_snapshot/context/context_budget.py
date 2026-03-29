"""
Adaptive Context Budget — Dynamic prompt token budgeting.
Drops lower-priority layers when context exceeds limits.

Priority order (highest first):
  1. System prompt (never dropped)
  2. Recent messages (never dropped)
  3. Knowledge graph
  4. Vector memories
  5. Web content
  6. Session summary
"""


MAX_CONTEXT_TOKENS = 6000  # Total budget for injected context (excl. system prompt)

# Approximate char-to-token ratio
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def enforce_budget(
    system_prompt: str,
    web_content: str = "",
    graph_block: str = "",
    memory_block: str = "",
    summary_block: str = "",
) -> dict:
    """
    Enforce token budget on context layers.
    Returns dict of layers that fit within budget.
    Drops lowest-priority layers first.
    """
    budget = MAX_CONTEXT_TOKENS

    # Priority-ordered layers (highest first = kept first)
    # Order: knowledge_graph > vector_memory > web_content > session_summary
    layers = [
        ("knowledge_graph", graph_block),
        ("vector_memory", memory_block),
        ("web_content", web_content),
        ("session_summary", summary_block),
    ]

    result = {}
    used = 0

    for name, content in layers:
        if not content:
            continue

        cost = _estimate_tokens(content)
        if used + cost <= budget:
            result[name] = content
            used += cost
        else:
            # Try truncated version (half)
            if cost > 100:
                half_content = content[:len(content) // 2] + "\n[truncated]"
                half_cost = _estimate_tokens(half_content)
                if used + half_cost <= budget:
                    result[name] = half_content
                    used += half_cost
                    print(f"[ContextBudget] Truncated {name}: {cost} → {half_cost} tokens")
                    continue

            print(f"[ContextBudget] Dropped {name}: {cost} tokens (budget: {used}/{budget})")

    if used > 0:
        print(f"[ContextBudget] Total context: {used}/{budget} tokens")

    return result
