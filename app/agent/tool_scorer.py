"""
Tool Scorer
Scores candidate tools for a given query using keyword weights,
action type affinity, and cooldown penalties.

Used by agent_orchestrator.py to replace flat keyword matching with
a ranked selection approach.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple, Optional

# ============================================
# TOOL SCORE PROFILES
# Each tool has keyword boost terms and a base score.
# ============================================

_TOOL_PROFILES: Dict[str, Dict] = {
    "search_web": {
        "base": 0.50,
        "keywords": ["search", "find", "what is", "who is", "latest", "current", "today", "news", "how does"],
    },
    "search_news": {
        "base": 0.45,
        "keywords": ["news", "headline", "breaking", "latest", "update", "today", "current events"],
    },
    "search_images": {
        "base": 0.40,
        "keywords": ["image", "images", "photo", "picture", "logo", "diagram", "screenshot", "show me"],
    },
    "search_videos": {
        "base": 0.40,
        "keywords": ["video", "videos", "youtube", "watch", "clip", "tutorial", "show me how"],
    },
    "search_places": {
        "base": 0.40,
        "keywords": ["near me", "nearby", "restaurant", "hotel", "location", "address", "place"],
    },
    "search_maps": {
        "base": 0.40,
        "keywords": ["map", "maps", "directions", "route", "distance", "navigate"],
    },
    "search_reviews": {
        "base": 0.45,
        "keywords": ["review", "reviews", "rating", "ratings", "testimonial", "feedback"],
    },
    "search_shopping": {
        "base": 0.45,
        "keywords": ["buy", "price", "pricing", "cost", "deal", "discount", "shop", "amazon", "flipkart", "order"],
    },
    "search_scholar": {
        "base": 0.45,
        "keywords": ["paper", "study", "journal", "academic", "scholar", "research paper", "citation"],
    },
    "search_patents": {
        "base": 0.50,
        "keywords": ["patent", "patents", "ip", "intellectual property", "trademark"],
    },
    "search_weather": {
        "base": 0.60,
        "keywords": ["weather", "forecast", "temperature", "rain", "humidity", "climate"],
    },
    "search_currency": {
        "base": 0.60,
        "keywords": ["currency", "exchange rate", "fx", "usd", "eur", "inr", "gbp", "convert currency"],
    },
    "search_company": {
        "base": 0.55,
        "keywords": ["company", "ceo", "headquarters", "revenue", "funding", "employees", "about company"],
    },
    "search_legal": {
        "base": 0.55,
        "keywords": ["legal", "policy", "compliance", "regulation", "law", "terms", "privacy policy"],
    },
    "search_jobs": {
        "base": 0.50,
        "keywords": ["job", "jobs", "career", "hiring", "role", "opening", "vacancy"],
    },
    "search_tech_docs": {
        "base": 0.55,
        "keywords": ["documentation", "docs", "api reference", "sdk", "guide", "manual", "how to use"],
    },
    "compare_products": {
        "base": 0.55,
        "keywords": ["compare", "vs", "versus", "alternatives", "which is better", "difference between"],
    },
    "summarize_url": {
        "base": 0.55,
        "keywords": ["summarize", "summary", "tl;dr", "tldr", "overview of url"],
    },
    "extract_tables": {
        "base": 0.50,
        "keywords": ["table", "tables", "csv", "spreadsheet", "extract data"],
    },
    "search_products": {
        "base": 0.45,
        "keywords": ["product", "products", "buy", "shop", "price", "review"],
    },
    "search_competitors": {
        "base": 0.50,
        "keywords": ["competitor", "competition", "rivals", "alternatives to"],
    },
    "search_trends": {
        "base": 0.50,
        "keywords": ["trend", "trends", "market trend", "growth", "forecast"],
    },
    "sentiment_scan": {
        "base": 0.55,
        "keywords": ["sentiment", "tone", "positive or negative", "polarity", "feeling"],
    },
    "faq_builder": {
        "base": 0.50,
        "keywords": ["faq", "questions", "q&a", "common questions", "build faq"],
    },
    "document_compare": {
        "base": 0.55,
        "keywords": ["compare documents", "diff", "difference", "document changes"],
    },
    "data_cleaner": {
        "base": 0.45,
        "keywords": ["clean data", "dedupe", "normalize data", "cleanup"],
    },
    "unit_cost_calc": {
        "base": 0.50,
        "keywords": ["unit cost", "cost breakdown", "pricing model", "cost per unit"],
    },
    "pdf_maker": {
        "base": 0.60,
        "keywords": ["make pdf", "create pdf", "export pdf", "convert to pdf", "generate pdf"],
    },
    "extract_entities": {
        "base": 0.50,
        "keywords": ["extract", "entities", "emails", "urls", "phones", "names from text"],
    },
    "validate_code": {
        "base": 0.55,
        "keywords": ["validate code", "lint", "security scan", "code review", "check syntax"],
    },
    "generate_tests": {
        "base": 0.60,
        "keywords": ["generate tests", "unit test", "pytest", "jest", "write tests"],
    },
    "execute_code": {
        "base": 0.50,
        "keywords": ["run code", "execute code", "run python", "run script", "evaluate expression"],
    },
    "read_file": {
        "base": 0.55,
        "keywords": ["read file", "open file", "content of", ".txt", ".md", ".csv", ".json"],
    },
    "calculate": {
        "base": 0.70,
        "keywords": ["calculate", "compute", "math", "what is", "+ ", "- ", "* ", "/ "],
    },
    "retrieve_knowledge": {
        "base": 0.50,
        "keywords": ["retrieve", "knowledge", "topic", "from knowledge base"],
    },
    "web_fetch": {
        "base": 0.65,
        "keywords": ["https://", "http://", "fetch url", "read url", "summarize url"],
    },
    "get_current_time": {
        "base": 0.70,
        "keywords": ["what time", "current time", "time now", "what's the time"],
    },
    "search_documents": {
        "base": 0.55,
        "keywords": ["my document", "uploaded", "from my file", "in my data", "search documents"],
    },
}


# ============================================
# SCORING FUNCTION
# ============================================

def score_tools(
    query: str,
    candidate_tools: List[str],
    tool_last_called: Optional[Dict[str, float]] = None,
    cooldown_seconds: float = 30.0,
    cooldown_penalty: float = 0.3,
    threshold: float = 0.55,
) -> List[Tuple[str, float]]:
    """
    Score candidate tools against the query.

    Returns a list of (tool_name, score) tuples sorted descending,
    filtered to those >= threshold.

    Args:
        query: The user query.
        candidate_tools: Tools eligible to be selected.
        tool_last_called: Dict mapping tool_name -> last unix timestamp called.
        cooldown_seconds: How long before cooldown penalty expires.
        cooldown_penalty: Score reduction for recently-called tools.
        threshold: Minimum score to include in results.
    """
    q_lower = query.lower()
    now = time.time()
    results: List[Tuple[str, float]] = []

    for tool_name in candidate_tools:
        profile = _TOOL_PROFILES.get(tool_name)
        if not profile:
            # Unknown tool — include with a low base score
            results.append((tool_name, 0.40))
            continue

        score = profile["base"]

        # Boost score for each matching keyword
        for kw in profile["keywords"]:
            if kw in q_lower:
                score += 0.10
                break  # Only one keyword boost per tool to avoid stacking

        # Apply cooldown penalty if the tool was recently called
        if tool_last_called:
            last = tool_last_called.get(tool_name, 0)
            if (now - last) < cooldown_seconds:
                score -= cooldown_penalty

        # Clamp to [0, 1.0]
        score = max(0.0, min(1.0, score))

        if score >= threshold:
            results.append((tool_name, score))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def select_top_tools(
    query: str,
    candidate_tools: List[str],
    tool_last_called: Optional[Dict[str, float]] = None,
    top_k: int = 12,
    threshold: float = 0.55,
) -> List[str]:
    """
    Convenience wrapper: returns only the tool names of the top-k scoring tools.
    Always preserves 'get_current_time' if it's a candidate (essential baseline).
    """
    scored = score_tools(query, candidate_tools, tool_last_called, threshold=threshold)
    selected = [name for name, _ in scored[:top_k]]

    # Always ensure time tool is available when queried
    if "get_current_time" in candidate_tools and "get_current_time" not in selected:
        selected.insert(0, "get_current_time")

    return selected
