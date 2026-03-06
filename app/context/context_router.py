"""
Context Router — Decides which context layers are needed per query.
Rule-based router (no LLM call, zero latency).

Layers: user_profile, emotion, knowledge_graph, vector_memory, session_summary
"""
from typing import List, Set


# Keywords that trigger each context layer
_LAYER_RULES = {
    "user_profile": {
        "my", "me", "prefer", "profile", "preference", "style", "always",
        "usually", "personality", "personal",
    },
    "knowledge_graph": {
        "backend", "architecture", "stack", "system", "infrastructure",
        "framework", "design", "structure", "component", "service",
        "project", "build", "tool", "uses", "using",
    },
    "vector_memory": {
        "remember", "recall", "previously", "before", "earlier",
        "mentioned", "said", "told", "discussed", "last", "history",
        "backend", "architecture", "stack", "project", "research",
        "compare", "analyze", "evaluate",
    },
    "session_summary": {
        "compare", "analyze", "research", "evaluate", "review",
        "continue", "resume", "earlier", "context", "summary",
        "so far", "conversation",
    },
}

# Sub-intents that always include certain layers
_INTENT_OVERRIDES = {
    "debugging": {"vector_memory", "knowledge_graph"},
    "system_design": {"knowledge_graph", "vector_memory", "session_summary"},
    "analysis": {"vector_memory", "session_summary"},
    "research": {"vector_memory", "knowledge_graph", "session_summary"},
    "reasoning": {"knowledge_graph", "vector_memory"},
    "code_generation": {"knowledge_graph"},
    "code_review": {"knowledge_graph"},
}


def select_context_layers(
    query: str,
    sub_intent: str = "",
    has_profile: bool = False,
    has_emotion: bool = False,
    has_web_content: bool = False,
) -> Set[str]:
    """
    Select which context layers to load based on query content and intent.
    Returns set of layer names. 'recent_messages' is always included (not tracked here).
    """
    layers: Set[str] = set()
    query_lower = query.lower()
    query_words = set(query_lower.split())

    # If web content already fetched, reduce memory injection to avoid prompt bloat
    if has_web_content:
        return layers  # Web content is sufficient context

    # Keyword-based matching
    for layer, keywords in _LAYER_RULES.items():
        if query_words & keywords:  # Set intersection
            layers.add(layer)

    # Intent-based overrides
    if sub_intent in _INTENT_OVERRIDES:
        layers |= _INTENT_OVERRIDES[sub_intent]

    # Always include profile if user has one and query is personalized
    if has_profile and not layers:
        layers.add("user_profile")

    # For long queries (likely complex), include more context
    if len(query_words) > 15:
        layers.add("knowledge_graph")
        layers.add("vector_memory")

    return layers
