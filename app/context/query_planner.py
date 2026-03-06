"""
Semantic Query Planner — Zero-cost task classification using keyword heuristics.
Produces a structured plan that tells the pipeline which subsystems to activate.

No LLM call needed. Runs before everything else.

Output example:
    {
        "task_type": "research_analysis",
        "needs_web": True,
        "needs_memory": True,
        "needs_chunking": True,
        "reasoning_depth": "high",
        "model_tier": "reasoning"
    }
"""
import re
from typing import Dict
from dataclasses import dataclass, asdict


# ============================================
# TASK PROFILES
# ============================================

TASK_PROFILES = {
    "coding": {
        "needs_web": False,
        "needs_memory": True,
        "needs_chunking": False,
        "reasoning_depth": "medium",
        "model_tier": "reasoning",
    },
    "architecture": {
        "needs_web": False,
        "needs_memory": True,
        "needs_chunking": False,
        "reasoning_depth": "high",
        "model_tier": "reasoning",
    },
    "debugging": {
        "needs_web": False,
        "needs_memory": True,
        "needs_chunking": False,
        "reasoning_depth": "high",
        "model_tier": "reasoning",
    },
    "research": {
        "needs_web": True,
        "needs_memory": False,
        "needs_chunking": True,
        "reasoning_depth": "high",
        "model_tier": "reasoning",
    },
    "web_analysis": {
        "needs_web": True,
        "needs_memory": False,
        "needs_chunking": True,
        "reasoning_depth": "medium",
        "model_tier": "fast",
    },
    "document_analysis": {
        "needs_web": False,
        "needs_memory": False,
        "needs_chunking": True,
        "reasoning_depth": "high",
        "model_tier": "reasoning",
    },
    "general_chat": {
        "needs_web": False,
        "needs_memory": True,
        "needs_chunking": False,
        "reasoning_depth": "low",
        "model_tier": "fast",
    },
    "creative": {
        "needs_web": False,
        "needs_memory": False,
        "needs_chunking": False,
        "reasoning_depth": "medium",
        "model_tier": "fast",
    },
}

# ============================================
# KEYWORD CLASSIFIERS
# ============================================

_TASK_KEYWORDS = {
    "coding": [
        "write", "implement", "code", "function", "class", "script",
        "program", "algorithm", "snippet", "endpoint", "decorator",
        "import", "def ", "async def", "refactor", "optimize code",
    ],
    "architecture": [
        "design", "architecture", "scale", "system design", "microservice",
        "infrastructure", "deploy", "load balance", "distributed",
        "database design", "schema", "pattern", "trade-off",
    ],
    "debugging": [
        "debug", "error", "bug", "fix", "crash", "fail", "broken",
        "not working", "issue", "exception", "traceback", "slow",
        "memory leak", "race condition", "deadlock",
    ],
    "research": [
        "explain", "how does", "what is", "compare", "difference between",
        "research", "paper", "study", "theory", "concept", "overview",
        "transformer", "attention", "embedding", "model",
    ],
    "web_analysis": [
        "summarize this url", "summarize this link", "this page",
        "this article", "from this website", "fetch", "scrape",
    ],
    "document_analysis": [
        "analyze this text", "analyze the following", "summarize this document",
        "extract from this", "identify the main", "key points from",
    ],
    "creative": [
        "suggest", "brainstorm", "ideas", "creative", "propose",
        "come up with", "invent", "imagine",
    ],
}

# URL pattern
_URL_PATTERN = re.compile(r'https?://[^\s]+')


# ============================================
# PLANNER
# ============================================

def plan_query(
    query: str,
    sub_intent: str = "",
    has_urls: bool = False,
) -> Dict:
    """
    Classify query and produce a task plan.
    Zero-cost: uses keyword matching only, no LLM call.

    Returns task profile dict with all pipeline control flags.
    """
    query_lower = query.lower()
    plan = {"task_type": "general_chat"}

    # URL detection overrides
    if has_urls or _URL_PATTERN.search(query):
        plan["task_type"] = "web_analysis"
        result = {**TASK_PROFILES["web_analysis"], **plan}
        print(f"[QueryPlanner] Plan: {plan['task_type']} (URL detected)")
        return result

    # Sub-intent override (from existing classifier)
    intent_map = {
        "code_generation": "coding",
        "code_review": "coding",
        "debugging": "debugging",
        "system_design": "architecture",
        "analysis": "research",
        "reasoning": "research",
        "ui_generation": "coding",
    }
    if sub_intent in intent_map:
        plan["task_type"] = intent_map[sub_intent]
        result = {**TASK_PROFILES[plan["task_type"]], **plan}
        print(f"[QueryPlanner] Plan: {plan['task_type']} (intent: {sub_intent})")
        return result

    # Keyword scoring
    scores = {}
    for task_type, keywords in _TASK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[task_type] = score

    if scores:
        best = max(scores, key=scores.get)
        plan["task_type"] = best

    # Long text → document analysis
    if len(query) > 500 and plan["task_type"] == "general_chat":
        plan["task_type"] = "document_analysis"

    result = {**TASK_PROFILES.get(plan["task_type"], TASK_PROFILES["general_chat"]), **plan}
    print(f"[QueryPlanner] Plan: {plan['task_type']} (scores: {scores})")
    return result
