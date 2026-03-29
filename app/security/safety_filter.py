from __future__ import annotations

import re

from app.safety.safety_filter import detect_injection as _detect_injection

_LEAK_PATTERNS = [
    re.compile(r"(?i)\bsystem prompt\b"),
    re.compile(r"(?i)\bdeveloper instruction"),
    re.compile(r"(?i)\btraceback\s*\("),
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\b"),
]


def detect_injection(text: str) -> bool:
    value = str(text or "")
    if _detect_injection(value):
        return True
    lowered = value.lower()
    if "developer instructions" in lowered or "developer instruction" in lowered:
        return True
    if "hidden policy" in lowered and ("reveal" in lowered or "show" in lowered or "print" in lowered):
        return True
    return False


def check_response_safety(response: str):
    text = str(response or "")
    if detect_injection(text):
        return "[Safety note: potential prompt-injection pattern detected]"
    for pat in _LEAK_PATTERNS:
        if pat.search(text):
            return "[Safety note: potential internal/sensitive data leakage detected]"
    return None
