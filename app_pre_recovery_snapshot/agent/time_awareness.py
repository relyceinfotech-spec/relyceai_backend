"""
Relyce AI - Time Awareness
Layer 9: Temporal context for the Structured Action Agent.

Detects:
  - Time-sensitive queries (live data needed)
  - Session staleness (cached context too old)
  - Urgency classification (now / later / pending)

Placed after ActionClassifier — temporal context only matters when action is needed.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional


# ============================================
# DATA STRUCTURE
# ============================================

@dataclass
class TemporalContext:
    """Temporal assessment of a query."""
    urgency: str = "now"            # "now" | "later" | "pending"
    is_time_sensitive: bool = False  # needs live/fresh data?
    session_age_minutes: float = 0.0
    stale: bool = False             # session too old for cached context?


# ============================================
# CONFIGURATION
# ============================================

# Session staleness threshold (minutes)
STALENESS_THRESHOLD_MINUTES = float(30)

# ============================================
# PATTERNS
# ============================================

# Time-sensitive query patterns (need live/fresh data)
_TIME_SENSITIVE_PATTERNS = re.compile(
    r"\b(?:current(?:ly)?|right\s+now|today(?:'s)?|latest|live|"
    r"real[- ]?time|up[- ]?to[- ]?date|recent(?:ly)?|"
    r"what\s+time|what\s+day|what\s+date|"
    r"now|at\s+the\s+moment|as\s+of\s+(?:today|now)|"
    r"this\s+(?:week|month|year|morning|evening)|"
    r"stock\s+price|weather|news|score|status|"
    r"trending|happening|breaking|attack|conflict|war|crisis)\b",
    re.IGNORECASE,
)

# "Later" / deferred signals
_LATER_PATTERNS = re.compile(
    r"\b(?:later|tomorrow|next\s+(?:week|month|time)|"
    r"when\s+(?:i\s+have\s+time|possible)|eventually|"
    r"schedule|remind\s+me|plan\s+for|in\s+the\s+future|"
    r"at\s+some\s+point|someday)\b",
    re.IGNORECASE,
)

# "Pending" / waiting signals
_PENDING_PATTERNS = re.compile(
    r"\b(?:waiting\s+(?:for|on)|pending|in\s+progress|"
    r"once\s+(?:it|they|we)|after\s+(?:that|this)|"
    r"when\s+(?:it|they|he|she)\s+(?:finish|complete|respond)|"
    r"follow\s+up|check\s+back)\b",
    re.IGNORECASE,
)


# ============================================
# MAIN FUNCTION
# ============================================

def assess_temporal_context(
    query: str,
    session_start_time: Optional[float] = None,
) -> TemporalContext:
    """
    Assess temporal context of a query.

    Args:
        query: User's input
        session_start_time: Unix timestamp of session start (for staleness)

    Returns:
        TemporalContext with urgency, time sensitivity, and staleness info
    """
    result = TemporalContext()

    # --- Detect time sensitivity ---
    result.is_time_sensitive = bool(_TIME_SENSITIVE_PATTERNS.search(query))

    # --- Classify urgency ---
    if _LATER_PATTERNS.search(query):
        result.urgency = "later"
    elif _PENDING_PATTERNS.search(query):
        result.urgency = "pending"
    else:
        result.urgency = "now"

    # Time-sensitive queries are always "now"
    if result.is_time_sensitive:
        result.urgency = "now"

    # --- Check session staleness ---
    if session_start_time:
        elapsed = time.time() - session_start_time
        result.session_age_minutes = elapsed / 60.0
        result.stale = result.session_age_minutes > STALENESS_THRESHOLD_MINUTES

    return result
