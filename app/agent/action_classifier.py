"""
Relyce AI - Action Classifier
Layers 4+5: Task Detection, Goal Mode, Ask vs Act Intelligence.

Classifies user input into:
  QUESTION -> respond
  TASK     -> plan + execute
  ACTION   -> execute immediately

Then evaluates:
  Clarity  -> is the request unambiguous?
  Risk     -> is the action reversible / safe?
  Decision -> ask | act | confirm

Also extracts GoalContext for outcome-oriented tasks and computes
a complexity_score used by the Orchestrator delegation gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class GoalContext:
    """Derived goal for TASK-type queries."""
    goal: str = ""
    is_outcome_oriented: bool = False
    completion_criteria: Optional[str] = None


@dataclass
class ActionDecision:
    """Complete classification result."""
    action_type: str = "QUESTION"
    clarity_score: float = 1.0
    risk_level: str = "low"
    decision: str = "act"
    missing_info: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    goal: GoalContext = field(default_factory=GoalContext)
    complexity_score: float = 0.0
    requires_comparison: bool = False
    requires_research: bool = False
    intent_signature: str = ""
    flush_memory: bool = False


# ============================================
# PATTERNS
# ============================================

_QUESTION_PATTERNS = re.compile(
    r"\b(?:what\s+is|what\s+are|who\s+is|when\s+did|where\s+is|"
    r"how\s+does|how\s+do|why\s+does|why\s+do|can\s+you\s+explain|"
    r"explain|describe|define|tell\s+me\s+about|meaning\s+of|"
    r"difference\s+between|is\s+it\s+true|does\s+it|do\s+they)\b",
    re.IGNORECASE,
)

_TASK_PATTERNS = re.compile(
    r"\b(?:build|create|make|design|implement|develop|set\s*up|"
    r"write\s+(?:a|an|the|me)|draft|compose|generate|plan|"
    r"refactor|rewrite|restructure|migrate|convert|transform|"
    r"fix|debug|resolve|solve|optimize|improve)\b",
    re.IGNORECASE,
)

_ACTION_PATTERNS = re.compile(
    r"\b(?:search\s+for|look\s+up|find|fetch|get\s+me|"
    r"send|post|submit|delete|remove|run|execute|deploy|"
    r"upload|download|install|update|push|pull)\b",
    re.IGNORECASE,
)

_COMPARISON_PATTERNS = re.compile(
    r"\b(?:compare|vs\.?|versus|difference\s+between|"
    r"pros?\s+(?:and|&)\s+cons?|which\s+is\s+better|"
    r"advantages?\s+(?:and|&|over)\s+disadvantages?|trade.?offs?)\b",
    re.IGNORECASE,
)

_RESEARCH_PATTERNS = re.compile(
    r"\b(?:analyze|evaluate|assess|review|audit|investigate|"
    r"research|study|examine|deep\s*dive|break\s*down|"
    r"comprehensive|thorough|detailed\s+analysis)\b",
    re.IGNORECASE,
)

_HIGH_RISK_ACTIONS = re.compile(
    r"\b(?:delete|remove|drop|destroy|wipe|erase|purge|"
    r"terminate|shutdown|shut\s*down|reset|overwrite|truncate)\b",
    re.IGNORECASE,
)

_MEDIUM_RISK_ACTIONS = re.compile(
    r"\b(?:send|post|submit|publish|deploy|push|execute|run|"
    r"update|modify|change|install|uninstall|migrate)\b",
    re.IGNORECASE,
)

_IRREVERSIBLE_ACTIONS = re.compile(
    r"\b(?:delete|remove|drop|destroy|wipe|erase|purge|"
    r"send|email|post|publish|deploy|push|truncate)\b",
    re.IGNORECASE,
)

_AMBIGUITY_SIGNALS = [
    re.compile(r"^(?:do\s+)?(?:it|that|this)\s*$", re.IGNORECASE),
    re.compile(r"\b(?:something|somehow|some\s+way)\b", re.IGNORECASE),
    re.compile(r"\b(?:the\s+(?:thing|stuff|one))\b", re.IGNORECASE),
]

_GOAL_PATTERNS = re.compile(
    r"\b(?:i\s+want\s+to|i\s+need\s+to|i(?:'| a)m\s+trying\s+to|"
    r"my\s+goal\s+is|the\s+goal\s+is|i(?:'| a)m\s+building|"
    r"i(?:'| a)m\s+working\s+on|help\s+me\s+(?:build|create|make|design)|"
    r"can\s+you\s+(?:build|create|make|design))\b",
    re.IGNORECASE,
)

_CRITERIA_PATTERNS = re.compile(
    r"\b(?:should\s+(?:be\s+able\s+to|have|include|support)|"
    r"must\s+(?:have|include|support|work)|"
    r"needs?\s+to\s+(?:have|include|support|work)|"
    r"with\s+(?:support\s+for|ability\s+to))\b",
    re.IGNORECASE,
)


# ============================================
# CORE CLASSIFICATION
# ============================================

def classify_action(
    query: str,
    context_messages: Optional[List[Dict]] = None,
    intent: str = "",
    sub_intent: str = "",
) -> ActionDecision:
    """
    Main entry point: classify query into QUESTION/TASK/ACTION,
    extract goal, assess clarity and risk, make ask/act/confirm decision.
    """
    result = ActionDecision()
    result.requires_comparison = bool(_COMPARISON_PATTERNS.search(query))
    result.requires_research = bool(_RESEARCH_PATTERNS.search(query))

    # Comparison/research override: always TASK (activates Goal Mode)
    if result.requires_comparison or result.requires_research:
        result.action_type = "TASK"
    else:
        result.action_type = _classify_type(query, intent, sub_intent)

    result.goal = _extract_goal(query, result.action_type)
    result.clarity_score, result.missing_info = _assess_clarity(query, context_messages)
    result.risk_level = _assess_risk(query, sub_intent)
    if result.action_type == "TASK":
        result.subtasks = _extract_subtasks(query, sub_intent)
    result.complexity_score = _compute_complexity(
        query, sub_intent, result.subtasks,
        result.requires_comparison, result.requires_research,
    )
    result.decision = _decide(
        result.action_type, result.clarity_score,
        result.risk_level, result.missing_info, query,
    )
    
    import hashlib
    goal_hash = hashlib.md5(result.goal.goal.encode()).hexdigest()[:8] if result.goal.goal else "none"
    result.intent_signature = f"{result.action_type}:{goal_hash}"
    
    # Stateless check for intent shift using context_messages
    if context_messages:
        for msg in reversed(context_messages):
            if msg.get("role") == "user":
                prev_query = msg.get("content", "")
                prev_type = _classify_type(prev_query, intent, sub_intent)
                prev_goal = _extract_goal(prev_query, prev_type)
                prev_hash = hashlib.md5(prev_goal.goal.encode()).hexdigest()[:8] if prev_goal.goal else "none"
                prev_signature = f"{prev_type}:{prev_hash}"
                
                if prev_signature and prev_signature != "none" and prev_signature != result.intent_signature:
                    if not (prev_type == "QUESTION" and prev_hash == "none"):
                        result.flush_memory = True
                break
    
    return result


# ============================================
# INTERNAL HELPERS
# ============================================

def _classify_type(query: str, intent: str, sub_intent: str) -> str:
    q = query.strip()
    
    MULTI_INTENT_VERBS = ["search", "calculate", "compare", "find", "summarize", "analyze"]
    if " and " in q.lower():
        if any(v in q.lower() for v in MULTI_INTENT_VERBS):
            return "TASK"
            
    if q.endswith("?") and len(q.split()) <= 15:
        return "QUESTION"
    has_action = bool(_ACTION_PATTERNS.search(q))
    has_task = bool(_TASK_PATTERNS.search(q))
    has_question = bool(_QUESTION_PATTERNS.search(q))
    if has_action and not has_task:
        return "ACTION"
    if has_task:
        return "TASK"
    if has_question:
        return "QUESTION"
    if intent in ("explain", "summarize"):
        return "QUESTION"
    if intent in ("generate", "transform"):
        return "TASK"
    return "QUESTION"


def _extract_goal(query: str, action_type: str) -> GoalContext:
    goal = GoalContext()
    if action_type == "QUESTION":
        return goal
    goal_match = _GOAL_PATTERNS.search(query)
    if goal_match:
        start = goal_match.end()
        goal.goal = query[start:].strip().rstrip("?.!")
        goal.is_outcome_oriented = True
    elif action_type == "TASK":
        cleaned = re.sub(
            r"^(?:please\s+|can\s+you\s+|could\s+you\s+|i\s+want\s+you\s+to\s+)",
            "", query, flags=re.IGNORECASE
        ).strip()
        goal.goal = cleaned
        goal.is_outcome_oriented = True
    elif action_type == "ACTION":
        goal.goal = query.strip()
        goal.is_outcome_oriented = False
    criteria_match = _CRITERIA_PATTERNS.search(query)
    if criteria_match:
        start = criteria_match.start()
        goal.completion_criteria = query[start:].strip().rstrip("?.!")
    return goal


def _assess_clarity(
    query: str,
    context_messages: Optional[List[Dict]] = None,
) -> Tuple[float, List[str]]:
    score = 1.0
    missing = []
    q = query.strip()
    word_count = len(q.split())
    if word_count <= 2:
        score -= 0.4
        missing.append("query_too_short")
    for pattern in _AMBIGUITY_SIGNALS:
        if pattern.search(q):
            score -= 0.3
            missing.append("ambiguous_reference")
            break
    if re.search(r"\b(?:it|this|that)\b", q, re.IGNORECASE):
        has_local_subject = re.search(r"\b(?:it|this|that)\s+\w{2,}", q, re.IGNORECASE)
        if not has_local_subject:
            if not context_messages:
                score -= 0.2
                missing.append("unclear_reference_no_context")
    action_match = _ACTION_PATTERNS.search(q)
    if action_match:
        after_verb = q[action_match.end():].strip()
        if not after_verb or len(after_verb.split()) < 1:
            score -= 0.3
            missing.append("missing_action_target")
    return max(0.0, min(1.0, score)), missing


def _assess_risk(query: str, sub_intent: str) -> str:
    if _HIGH_RISK_ACTIONS.search(query):
        return "high"
    if _MEDIUM_RISK_ACTIONS.search(query):
        return "medium"
    return "low"


def is_reversible(query: str) -> bool:
    return not bool(_IRREVERSIBLE_ACTIONS.search(query))


def _extract_subtasks(query: str, sub_intent: str) -> List[str]:
    subtasks = []
    q_lower = query.lower()
    parts = re.split(r"\b(?:and\s+then|then|and\s+also|,\s*(?:and\s+)?)\b", query, flags=re.IGNORECASE)
    if len(parts) > 1:
        for part in parts:
            part = part.strip()
            if part and len(part) > 5:
                subtasks.append(part)
        return subtasks
    if sub_intent in ("system_design", "coding_complex"):
        subtasks = ["design", "implement", "validate"]
    elif sub_intent == "debugging":
        subtasks = ["diagnose", "fix"]
    elif "and" in q_lower and _TASK_PATTERNS.search(query):
        parts = re.split(r"\band\b", query, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if part and len(part) > 5:
                subtasks.append(part)
    if not subtasks:
        subtasks = [query.strip()]
    return subtasks


def _compute_complexity(
    query: str, sub_intent: str, subtasks: List[str],
    requires_comparison: bool, requires_research: bool,
) -> float:
    score = 0.0
    if len(subtasks) > 3:
        score += 0.4
    elif len(subtasks) > 1:
        score += 0.25
    elif len(subtasks) == 1:
        score += 0.1
    if requires_comparison:
        score += 0.3
    if requires_research:
        score += 0.25
    complex_intents = {"system_design", "coding_complex", "architecture"}
    moderate_intents = {"debugging", "code_explanation", "ui_implementation"}
    if sub_intent in complex_intents:
        score += 0.2
    elif sub_intent in moderate_intents:
        score += 0.1
    word_count = len(query.split())
    if word_count > 30:
        score += 0.1
    elif word_count > 15:
        score += 0.05
    return min(1.0, score)


def _decide(
    action_type: str, clarity_score: float,
    risk_level: str, missing_info: List[str], query: str,
) -> str:
    if clarity_score < 0.5:
        return "ask"
    if risk_level == "high":
        return "confirm"
    if risk_level == "medium" and not is_reversible(query):
        return "confirm"
    return "act"
