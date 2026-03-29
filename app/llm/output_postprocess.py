"""
Output post-processing helpers for LLM responses.
"""
from __future__ import annotations

import html
import re
from typing import List


_TOOL_ARTIFACT_PATTERNS = (
    r"^\s*TOOL\b[#:\-\s]*.*$",
    r"^\s*TOOL#\s*.*$",
    r"^\s*_?CALL\s*:\s*.*$",
    r"^\s*TOOL_CALL\s*:\s*.*$",
    r"^\s*Assistant:\s*First,.*$",
    r"^\s*First,\s*the user.*$",
    r"^\s*You have completed all required execution steps\.?\s*$",
    r"^\s*WEB SEARCH\s*$",
    r"^\s*Search Result\s*\d+\s*[:\-].*$",
    r"^\s*(Title|Snippet Details|Link|Source Link|Copy text|Export)\s*[:\-]?.*$",
    r"^\s*Thoughts\s*$",
    r"^\s*Progress\s*$",
    r"^\s*Searching\s+\"?.*\"?\s*$",
    r"^\s*Searched\s+\"?.*\"?\s*$",
    r"^\s*Step completed\s*$",
    r"^\s*Sources collected\..*$",
    r"^\s*Looking for relevant and recent sources.*$",
    r"^\s*##\s*Verification.*$",
    r"^\s*(Verification Report|Accuracy Assessment|Completeness Assessment|Uncertainties|Improvements Recommended).*$",
    r"^\s*.*tools?\s+disabled.*$",
    r"^\s*.*no external fetches possible.*$",
    # Prompt/policy leakage fragments that should never reach end-users.
    r"^\s*(AGENT OPERATIONAL LOGIC|EXECUTION MODE RULES|RUNTIME CONTEXT|CRITICAL (SYSTEM )?OVERRIDE)\s*:?.*$",
    r"^\s*(Rules for FINAL ANSWER|Classify\s*&\s*Plan)\s*:?.*$",
    r"^\s*(Do NOT describe steps or reasoning|Do NOT narrate what you did|Only include sources if asked|Deliver one clean final response)\.?\s*$",
    r"^\s*(BEGIN|END)\s+SYSTEM\s+PROMPT\s*$",
    r"^\s*\[CRITICAL OVERRIDE\].*$",
    r"^\s*This implies that in this simulation.*$",
)


def sanitize_output_text(text: str, trim_outer: bool = True) -> str:
    """Apply lightweight, deterministic cleanup to model output."""
    if not text:
        return text

    text = html.unescape(text)
    is_html_document = bool(
        re.search(r"<!doctype\s+html|<html[\s>]|<head>|<body>", text, flags=re.IGNORECASE)
    )
    if is_html_document:
        return text

    sanitized = text.replace("\u2014", " - ").replace("\u2013", " - ")

    # Remove common mojibake artifacts without matching invalid source bytes.
    sanitized = re.sub(r"(?:Ã.|Â.)+", "", sanitized)

    for pattern in _TOOL_ARTIFACT_PATTERNS:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip() if trim_outer else sanitized


def fix_html_css_output(text: str) -> str:
    """Repair common CSS/HTML token corruption patterns."""
    if not text:
        return text
    content = text

    content = re.sub(r"<!\s*-\s*-", "<!--", content)
    content = re.sub(r"-\s*-\s*>", "-->", content)
    content = re.sub(r"<!\s*-\s*-\s*\[", "<!--[", content)
    content = re.sub(r"\]\s*-\s*-\s*>", "]-->", content)

    def fix_css_property(match: re.Match) -> str:
        indent = match.group(1)
        name = match.group(2)
        return f"{indent}--{name}:"

    content = re.sub(r"(\n\s*)-\s+([a-zA-Z][a-zA-Z0-9-]*)\s*:", fix_css_property, content)
    content = re.sub(r"(\s{2,})-\s+([a-zA-Z][a-zA-Z0-9-]*)\s*:", fix_css_property, content)

    content = re.sub(r"var\s*\(\s*-\s+", "var(--", content)
    content = re.sub(r"var\s*\(\s*-\s*-\s*", "var(--", content)
    content = re.sub(r"--\s+([a-zA-Z])", r"--\1", content)
    content = re.sub(r"--([a-zA-Z][a-zA-Z0-9-]*)\s+([a-zA-Z])", r"--\1\2", content)

    content = content.replace("group: card;", "")
    content = content.replace("group: ;", "")

    if "line-clamp:" not in content and "-webkit-line-clamp:" in content:
        content = re.sub(
            r"(\s*)-webkit-line-clamp:\s*([0-9]+);",
            r"\1line-clamp: \2;\n\1-webkit-line-clamp: \2;",
            content,
        )

    return content


def is_structured_markdown(text: str) -> bool:
    if not text:
        return False
    return "## " in text or "- " in text or "|---" in text


def sanitize_followup_questions(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for q in (items or []):
        if not isinstance(q, str):
            continue
        q2 = re.sub(r"\s+", " ", q).strip(" -?\t\n\r")
        if not q2:
            continue
        words = q2.split()
        if len(words) < 6 or len(words) > 14:
            continue
        key = re.sub(r"[^a-z0-9 ]+", "", q2.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(q2)
        if len(cleaned) >= 3:
            break
    return cleaned


def extract_followups_from_text(text: str) -> List[str]:
    if not text:
        return []
    section = re.search(
        r"(?:^|\n)##\s*Related Questions\s*\n(?P<body>[\s\S]*?)(?:\n##\s|$)",
        text,
        flags=re.IGNORECASE,
    )
    body = section.group("body") if section else text
    candidates: List[str] = []
    for line in body.splitlines():
        m = re.match(r"^\s*(?:[-*]|\u2022)\s+(.+?)\s*$", line)
        if m:
            candidates.append(m.group(1).strip())
    return sanitize_followup_questions(candidates)


def extract_action_chips_from_text(text: str) -> List[str]:
    if not text:
        return []
    chips = re.findall(r"\[([^\]\n]{3,40})\]", text)
    out: List[str] = []
    seen: set[str] = set()
    for chip in chips:
        c = re.sub(r"\s+", " ", chip).strip()
        if not c:
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= 3:
            break
    return out

