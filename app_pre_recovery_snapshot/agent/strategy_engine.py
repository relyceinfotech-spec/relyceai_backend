"""
Relyce AI - Strategy Engine
Intent Detection + Planning Mode Selection.

Maps user queries to execution strategies:
  SEQUENTIAL_PLAN     → step-by-step (tool-heavy / simple queries)
  PARALLEL_REASONING  → multi-track reasoning (comparison / generation / exploration)
  ADAPTIVE_CODE_PLAN  → 8-step code pipeline (understand → design → generate → validate → risk → execute → repair → verify)
  RESEARCH_PLAN       → knowledge-gap-driven research (detect gap → decide search → inject findings)

Golden Rule: Parallel Thinking, Sequential Execution.
This module only THINKS — never executes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class IntentSignals:
    """Detected intent signals from user query."""
    requires_tools: bool = False
    is_generation: bool = False
    is_comparison: bool = False
    is_exploratory: bool = False
    is_code_generation: bool = False
    has_tech_domain: bool = False
    is_research_task: bool = False
    strategy_confidence: float = 1.0
    research_confidence: float = 0.0


# ============================================
# INTENT DETECTION PATTERNS
# ============================================

_TOOL_KEYWORDS = [
    "search", "find", "look up", "fetch", "get me", "what time",
    "current", "today", "latest", "live", "real-time",
]

_GENERATION_KEYWORDS = [
    "write", "create", "generate", "compose", "draft", "build",
    "design", "make", "produce", "craft",
]

_COMPARISON_KEYWORDS = [
    "compare", "vs", "versus", "difference between", "which is better",
    "pros and cons", "trade-offs", "advantages", "disadvantages",
    "better than", "worse than", "or",
]

_EXPLORATORY_KEYWORDS = [
    "research", "explore", "investigate", "analyze", "deep dive",
    "understand", "explain", "comprehensive", "thorough",
    "tell me everything", "how does", "why does",
]

_RESEARCH_KEYWORDS = [
    "how to use", "documentation", "api for", "library for",
    "example of", "tutorial", "guide for", "best practice",
    "pattern for", "reference for", "usage of", "syntax for",
]

_CODE_COMMANDS = [
    "debug", "fix this", "refactor", "implement", "write a function", 
    "write code", "create a script", "build an api", "write a class"
]

_TECH_ENTITIES = [
    "code", "function", "class", "script", "program", "api",
    "algorithm", "data structure", "sql", "query",
    "html", "css", "javascript", "python", "java", "react", "vue",
    "node", "aws", "docker", "kubernetes", "component", "endpoint", 
    "backend", "frontend", "database",
]

_CODE_PATTERNS = [
    r"```",                          # code blocks
    r"def\s+\w+",                    # function definitions
    r"class\s+\w+",                  # class definitions
    r"import\s+\w+",                 # imports
    r"\w+\.\w+\(",                   # method calls
    r"(?:let|const|var)\s+\w+",      # JS declarations
]


# ============================================
# INTENT DETECTION
# ============================================

def _has_keyword(query: str, keywords: List[str]) -> bool:
    """Helper to cleanly check keywords using word boundaries where applicable."""
    for kw in keywords:
        # Use simple string match for special chars or multi-word,
        # but regex word boundary for safety on standard words
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', query.lower()):
            return True
    return False


def detect_intent(query: str) -> IntentSignals:
    """
    Analyze query to detect intent signals.
    Uses strict word-boundary matching and pattern detection.
    """
    signals = IntentSignals()
    q_lower = query.lower()

    # Tool need detection
    signals.requires_tools = _has_keyword(query, _TOOL_KEYWORDS)

    # Generation detection
    signals.is_generation = _has_keyword(query, _GENERATION_KEYWORDS)

    # Comparison detection
    signals.is_comparison = _has_keyword(query, _COMPARISON_KEYWORDS)

    # Exploratory detection
    signals.is_exploratory = _has_keyword(query, _EXPLORATORY_KEYWORDS)

    # Tech domain detection (frameworks, languages, concepts)
    signals.has_tech_domain = _has_keyword(query, _TECH_ENTITIES)

    # Explicit code generation detection (commands + code blocks + generation over tech domain)
    has_code_command = _has_keyword(query, _CODE_COMMANDS)
    has_code_pattern = any(re.search(p, query) for p in _CODE_PATTERNS)
    signals.is_code_generation = has_code_command or has_code_pattern or (signals.is_generation and signals.has_tech_domain)

    # Research task detection
    signals.is_research_task = _has_keyword(query, _RESEARCH_KEYWORDS)

    # Calculate confidences based on signal strength
    num_signals = sum([
        signals.requires_tools, signals.is_generation, signals.is_comparison,
        signals.is_exploratory, signals.is_code_generation, signals.is_research_task
    ])
    
    # High confidence if clear single intent, lower if ambiguous
    signals.strategy_confidence = 1.0 if num_signals <= 1 else max(0.4, 1.0 - (num_signals * 0.15))
    
    # Research confidence boosted by explicit keywords or exploratory+tech combo
    if signals.is_research_task:
        signals.research_confidence = 0.9
    elif signals.is_exploratory and signals.has_tech_domain:
        signals.research_confidence = 0.75
    else:
        signals.research_confidence = 0.2

    return signals


# ============================================
# PLANNING MODE SELECTION
# ============================================

PLANNING_MODES = {
    "SEQUENTIAL_PLAN": "SEQUENTIAL_PLAN",
    "PARALLEL_REASONING": "PARALLEL_REASONING",
    "ADAPTIVE_CODE_PLAN": "ADAPTIVE_CODE_PLAN",
    "RESEARCH_PLAN": "RESEARCH_PLAN",
}


def select_planning_mode(intent: IntentSignals) -> str:
    """
    Select the planning mode based on detected intent.

    Priority:
    Priority:
      1. Explicit code generation → ADAPTIVE_CODE_PLAN
      2. Comparison → PARALLEL_REASONING (even if tech entities mentioned)
      3. Research task (or Exploratory + Tech Domain) → RESEARCH_PLAN
      4. Tool-heavy → SEQUENTIAL_PLAN
      5. Generation / Exploratory → PARALLEL_REASONING
      6. Default → SEQUENTIAL_PLAN
    """
    if intent.is_code_generation:
        return PLANNING_MODES["ADAPTIVE_CODE_PLAN"]

    if intent.is_comparison:
        return PLANNING_MODES["PARALLEL_REASONING"]

    if intent.is_research_task or (intent.is_exploratory and intent.has_tech_domain):
        return PLANNING_MODES["RESEARCH_PLAN"]

    if intent.requires_tools:
        return PLANNING_MODES["SEQUENTIAL_PLAN"]

    if intent.is_generation or intent.is_exploratory:
        return PLANNING_MODES["PARALLEL_REASONING"]

    return PLANNING_MODES["SEQUENTIAL_PLAN"]


# ============================================
# PARALLEL REASONING (PROMPT COMPOSITION ONLY)
# ============================================

# Reasoning track templates — these are prompt sections, NOT executions
_REASONING_TRACKS = {
    "performance": (
        "**[Performance Analysis]**\n"
        "Evaluate efficiency, speed, resource usage, and scalability.\n"
        "Consider: time complexity, memory footprint, throughput, latency impact."
    ),
    "structure": (
        "**[Structural Analysis]**\n"
        "Evaluate architecture, organization, maintainability, and design patterns.\n"
        "Consider: modularity, separation of concerns, extensibility, readability."
    ),
    "risk": (
        "**[Risk Assessment]**\n"
        "Evaluate potential failures, edge cases, security concerns, and dependencies.\n"
        "Consider: error handling, data validation, compatibility, breaking changes."
    ),
    "alternatives": (
        "**[Alternative Approaches]**\n"
        "Identify at least 2 alternative solutions or perspectives.\n"
        "For each: state the approach, its trade-offs, and when to prefer it."
    ),
}


def build_parallel_reasoning(query: str) -> Dict[str, str]:
    """
    Build parallel reasoning tracks for the query.

    These are prompt sections composed in parallel (no LLM calls).
    They get merged into the system prompt context so the LLM
    reasons across all dimensions simultaneously.
    """
    tracks = {}
    q_lower = query.lower()

    # Always include structure and alternatives for parallel reasoning
    tracks["structure"] = _REASONING_TRACKS["structure"]
    tracks["alternatives"] = _REASONING_TRACKS["alternatives"]

    # Include performance track for technical queries
    tech_signals = ["performance", "speed", "fast", "efficient", "optimize",
                    "scale", "framework", "library", "tool", "stack"]
    if any(s in q_lower for s in tech_signals):
        tracks["performance"] = _REASONING_TRACKS["performance"]

    # Include risk track for decision-heavy queries
    risk_signals = ["production", "deploy", "security", "migration",
                    "breaking", "upgrade", "risk", "safe", "reliable"]
    if any(s in q_lower for s in risk_signals):
        tracks["risk"] = _REASONING_TRACKS["risk"]

    # For comparison queries, always include all tracks
    comparison_signals = ["compare", "vs", "versus", "difference", "better"]
    if any(s in q_lower for s in comparison_signals):
        tracks = dict(_REASONING_TRACKS)  # all tracks

    return tracks


def merge_reasoning_tracks(tracks: Dict[str, str]) -> str:
    """Merge reasoning tracks into a single context string."""
    if not tracks:
        return ""

    sections = [
        "**[MULTI-DIMENSIONAL REASONING CONTEXT]**",
        "Analyze the following query across these dimensions simultaneously:",
        "",
    ]
    for name, content in tracks.items():
        sections.append(content)
        sections.append("")

    sections.append(
        "Synthesize insights from ALL dimensions into a cohesive, "
        "well-structured response. Do not answer each dimension separately."
    )

    return "\n".join(sections)


# ============================================
# ADAPTIVE CODE PLAN STEPS
# ============================================

ADAPTIVE_CODE_STEPS = [
    "understand_requirements",
    "design_code_structure",
    "generate_initial_code",
    "static_validation",
    "risk_assessment",
    "execution_attempt",
    "repair_cycle",
    "final_verification",
]


def get_adaptive_code_steps() -> List[str]:
    """Return the ordered steps for the adaptive code plan."""
    return list(ADAPTIVE_CODE_STEPS)


# ============================================
# SEQUENTIAL PLAN STEPS
# ============================================

SEQUENTIAL_STEPS = [
    "analyze_request",
    "prepare_execution",
    "execute_stepwise",
    "verify_result",
]


def get_sequential_steps() -> List[str]:
    """Return the ordered steps for the sequential plan."""
    return list(SEQUENTIAL_STEPS)


# ============================================
# KNOWLEDGE GAP DETECTION
# ============================================

# Gap type patterns — detected from query + context
_GAP_PATTERNS = {
    "API_DOC": [
        r"\bapi\b", r"endpoint", r"rest\b", r"graphql",
        r"how to call", r"request to", r"response from",
        r"documentation for", r"api for", r"api reference",
    ],
    "LIBRARY_USAGE": [
        r"how to use", r"example of", r"usage of",
        r"library for", r"package for", r"module for",
        r"import\s+\w+", r"install\s+\w+", r"npm\s+",
        r"pip\s+install", r"how does .* internally",
    ],
    "UI_REFERENCE": [
        r"component", r"layout", r"styling", r"css for",
        r"ui for", r"design pattern", r"responsive",
        r"animation", r"tailwind", r"bootstrap",
    ],
    "ERROR_FIX": [
        r"error", r"exception", r"traceback", r"stack trace",
        r"not working", r"broken", r"fix this", r"bug",
        r"fails with", r"crash", r"issue with",
    ],
}


def detect_knowledge_gap(context: Optional[Dict] = None) -> Optional[str]:
    """
    Detect what knowledge the agent is missing to fulfill the request.

    Analyzes the execution context (query, generated code, errors)
    to identify specific gaps:
      - API_DOC:       needs API documentation
      - LIBRARY_USAGE: needs library examples/patterns
      - UI_REFERENCE:  needs UI component references
      - ERROR_FIX:     needs error resolution guidance
      - None:          no gap detected

    Advisory only — processor decides whether to research.
    """
    if not context:
        return None

    # Combine all text signals from context
    signals = ""
    if isinstance(context, dict):
        for key in ["query", "user_query", "generated_code", "error", "last_error"]:
            val = context.get(key, "")
            if val:
                signals += f" {val}"

    if not signals.strip():
        return None

    signals_lower = signals.lower()

    # Check each gap type in priority order
    for gap_type, patterns in _GAP_PATTERNS.items():
        if any(re.search(p, signals_lower) for p in patterns):
            return gap_type

    # Fallback to general domains based on IntentSignals if passed in context
    if context.get("intent_signals"):
        signals_obj = context["intent_signals"]
        if signals_obj.is_exploratory and signals_obj.has_tech_domain:
            return "LIBRARY_USAGE"

    return None


# ============================================
# RESEARCH DECISION
# ============================================

# Maps gap types to search strategies
_RESEARCH_MODES = {
    "API_DOC": {
        "tool": "search_web",
        "mode": "DOC_SEARCH",
        "query_prefix": "API documentation for",
    },
    "LIBRARY_USAGE": {
        "tool": "search_web",
        "mode": "CODE_EXAMPLE",
        "query_prefix": "code example usage of",
    },
    "UI_REFERENCE": {
        "tool": "search_web",
        "mode": "UI_SEARCH",
        "query_prefix": "UI component reference for",
    },
    "ERROR_FIX": {
        "tool": "search_web",
        "mode": "ERROR_SEARCH",
        "query_prefix": "fix error",
    },
}


def decide_research(gap: Optional[str]) -> Optional[Dict]:
    """
    Decide what research to perform based on the detected knowledge gap.

    Returns:
      - None: no research needed
      - Dict with tool, mode, and query_prefix for the processor to execute

    Advisory only — the processor calls the actual tool.
    """
    if not gap:
        return None

    return _RESEARCH_MODES.get(gap)
