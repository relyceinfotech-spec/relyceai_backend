"""
Content safety policy utilities.

Policy intent:
- Block illegal/exploitative sexual content.
- Block explicit porn/erotic generation requests.
- Allow informational discussion (e.g., movies/games/news/policy/education).
"""
from __future__ import annotations

import re
from typing import Tuple


_HARD_BLOCK_PATTERNS = [
    # minors + sexualization/exploitation
    r"\b(child|minor|underage|teen)\b.{0,40}\b(sex|sexual|nude|explicit|porn|erotic|rape|molest)\b",
    r"\b(sex|sexual|nude|explicit|porn|erotic|rape|molest)\b.{0,40}\b(child|minor|underage|teen)\b",
    # non-consensual sexual violence
    r"\b(rape|sexual assault|forced sex|non[- ]consensual)\b",
    # trafficking/exploitation framing
    r"\b(sex trafficking|exploit(?:ation)?|revenge porn)\b",
]

_EXPLICIT_GENERATION_PATTERNS = [
    r"\b(write|generate|describe|roleplay|narrate)\b.{0,30}\b(sex|sexual|porn|erotic|nude)\b",
    r"\b(blowjob|handjob|penetrat(?:e|ion)|cum|orgasm|fuck(?:ing)?)\b",
    r"\b(explicit sex|porn scene|erotic story|sexual roleplay)\b",
]

_INFORMATIONAL_ALLOW_HINTS = [
    "movie",
    "film",
    "game",
    "series",
    "media",
    "review",
    "rating",
    "plot",
    "news",
    "policy",
    "compliance",
    "education",
    "health",
    "safety",
    "law",
    "research",
    "history",
    "analysis",
]

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?instructions",
    r"override\s+(the\s+)?(system|policy|instructions)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|hidden\s+prompt)",
    r"show\s+(the\s+)?(system\s+prompt|hidden\s+instructions)",
]

_HARD_BLOCK_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _HARD_BLOCK_PATTERNS]
_EXPLICIT_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _EXPLICIT_GENERATION_PATTERNS]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


def sanitize_rag_text(text: str) -> str:
    """Remove common prompt-injection instructions from untrusted retrieved text."""
    if not text:
        return ""
    cleaned = text
    for pat in _INJECTION_RE:
        cleaned = pat.sub("[FILTERED]", cleaned)
    return cleaned


def has_prompt_injection_markers(text: str) -> bool:
    text = text or ""
    return any(p.search(text) for p in _INJECTION_RE)


def classify_nsfw(text: str) -> Tuple[bool, str]:
    """
    Returns (blocked, reason).
    Allows informational/media discussion even with adult-theme mentions.
    """
    t = (text or "").strip()
    if not t:
        return False, ""

    lower = t.lower()
    info_context = any(h in lower for h in _INFORMATIONAL_ALLOW_HINTS)

    if any(p.search(t) for p in _HARD_BLOCK_RE):
        return True, "Blocked by safety policy (illegal or exploitative sexual content)."

    explicit = any(p.search(t) for p in _EXPLICIT_RE)
    if explicit and not info_context:
        return True, "Blocked by safety policy (explicit sexual content)."

    return False, ""
