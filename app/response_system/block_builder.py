"""Deterministic block extraction/normalization and content intelligence orchestration."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import ALLOWED_BLOCK_TYPES
from .content_intelligence import detect_content_plan


def to_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            item = item.strip()
            if item:
                out.append(item)
    return out


def strip_internal_leakage(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text)
    blocked_patterns = [
        r"^\s*_?CALL\s*:\s*.*$",
        r"^\s*TOOL_CALL\s*:\s*.*$",
        r"^\s*Assistant:\s*First,.*$",
        r"^\s*First,\s*the user.*$",
        r"^\s*AGENT OPERATIONAL LOGIC\s*:?.*$",
        r"^\s*RUNTIME CONTEXT\s*:?.*$",
        r"^\s*CRITICAL SYSTEM OVERRIDE\s*:?.*$",
        r"^\s*Rules for FINAL ANSWER\s*:?.*$",
        r"^\s*Classify\s*&\s*Plan\s*:?.*$",
        r"^\s*Do NOT describe steps or reasoning\.?\s*$",
        r"^\s*Do NOT narrate what you did\.?\s*$",
        r"^\s*Only include sources if asked\.?\s*$",
        r"^\s*Deliver one clean final response\.?\s*$",
        r"^\s*You have completed all required execution steps\.?.*$",
    ]
    for pattern in blocked_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def to_clean_str(value: Any, max_len: int = 1000) -> str:
    if value is None:
        return ""
    text = strip_internal_leakage(str(value).strip())
    return text[:max_len] if text else ""


def normalize_sources(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for src in value:
        if isinstance(src, str):
            url = src.strip()
            if url:
                normalized.append({"name": "", "url": url, "trust": None})
            continue
        if isinstance(src, dict):
            url = str(src.get("url") or src.get("link") or "").strip()
            if not url:
                continue
            name = str(src.get("name") or src.get("title") or "").strip()
            trust = src.get("trust")
            if isinstance(trust, (int, float)):
                trust = max(0.0, min(1.0, float(trust)))
            else:
                trust = None
            normalized.append({"name": name, "url": url, "trust": trust})
    return normalized


def extract_key_points(text: str) -> List[str]:
    bullets: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+.+", stripped):
            point = re.sub(r"^[-*]\s+", "", stripped).strip()
            if point:
                bullets.append(point)
    if bullets:
        return bullets[:8]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) <= 1:
        return []
    return sentences[1:5]


def extract_answer(text: str) -> str:
    clean = (text or "").strip()
    if not clean:
        return ""
    first_line = clean.splitlines()[0].strip()
    if re.match(r"^(#{1,6}\s+|[A-Za-z ]+:$)", first_line):
        for line in clean.splitlines()[1:]:
            line = line.strip()
            if line and not re.match(r"^(#{1,6}\s+|[A-Za-z ]+:$)", line):
                return line
    return first_line


def infer_answer_type(user_query: str, text: str) -> str:
    q = (user_query or "").lower()
    t = (text or "").lower()
    if "compare" in q or " vs " in q or "comparison" in q:
        return "comparison"
    if "timeline" in q or "history" in q:
        return "timeline"
    if re.search(r"^\s*who\s+|^\s*what\s+|^\s*when\s+", q):
        return "fact"
    if "|" in t and re.search(r"\|\s*[-:]{2,}\s*\|", t):
        return "comparison"
    return "summary"


def extract_markdown_table(text: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for i in range(len(lines) - 1):
        header = lines[i]
        sep = lines[i + 1]
        if "|" not in header or "|" not in sep:
            continue
        if not re.search(r"^\|?[\s:\-\|]+\|?$", sep):
            continue
        columns = [c.strip() for c in header.strip("|").split("|")]
        rows: List[List[str]] = []
        for row_line in lines[i + 2:]:
            if "|" not in row_line:
                break
            row = [c.strip() for c in row_line.strip("|").split("|")]
            if len(row) < len(columns):
                row.extend([""] * (len(columns) - len(row)))
            rows.append(row[: len(columns)])
        if columns and rows:
            return columns, rows
    return None


def extract_timeline_events(text: str) -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*(\d{4})\s*[-:]\s*(.+?)\s*$", line)
        if m:
            events.append({"time": m.group(1), "event": m.group(2).strip()[:240]})
    return events[:12]


def extract_card_fields(text: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            continue
        if len(key) > 40 or len(val) > 260:
            continue
        if key.lower() in {"answer", "summary", "sources", "confidence"}:
            continue
        fields[key] = val
        if len(fields) >= 8:
            break
    return fields


def normalize_block(block: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(block, dict):
        return None
    btype = str(block.get("type", "")).strip().lower()
    if btype not in ALLOWED_BLOCK_TYPES:
        return None

    title = to_clean_str(block.get("title"), max_len=120)

    if btype == "text":
        content = to_clean_str(block.get("content"), max_len=2000)
        if not content:
            return None
        return {"type": "text", "title": title or "Answer", "content": content}

    if btype == "list":
        items = to_str_list(block.get("items"))[:12]
        if not items:
            return None
        return {"type": "list", "title": title or "Key Points", "items": items}

    if btype == "table":
        columns = to_str_list(block.get("columns"))[:10]
        rows_raw = block.get("rows")
        rows: List[List[str]] = []
        if isinstance(rows_raw, list):
            for row in rows_raw[:20]:
                if isinstance(row, list):
                    rows.append([to_clean_str(c, 200) for c in row[: len(columns) or 10]])
        if not columns or not rows:
            return None
        return {"type": "table", "title": title or "Table", "columns": columns, "rows": rows}

    if btype == "timeline":
        events_raw = block.get("events")
        events: List[Dict[str, str]] = []
        if isinstance(events_raw, list):
            for ev in events_raw[:20]:
                if isinstance(ev, dict):
                    t = to_clean_str(ev.get("time"), 40)
                    e = to_clean_str(ev.get("event"), 260)
                    if t and e:
                        events.append({"time": t, "event": e})
        if not events:
            return None
        return {"type": "timeline", "title": title or "Timeline", "events": events}

    if btype == "card":
        ctitle = to_clean_str(block.get("title"), 120)
        fields_raw = block.get("fields")
        fields: Dict[str, str] = {}
        if isinstance(fields_raw, dict):
            for k, v in list(fields_raw.items())[:10]:
                key = to_clean_str(k, 60)
                val = to_clean_str(v, 260)
                if key and val:
                    fields[key] = val
        if not ctitle and not fields:
            return None
        return {"type": "card", "title": ctitle or "Details", "fields": fields}

    return None


def normalize_blocks(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for block in value:
        normalized = normalize_block(block)
        if normalized:
            out.append(normalized)
    return out[:8]


def build_blocks_with_intelligence(
    *,
    answer: str,
    key_points: List[str],
    text: str,
    user_query: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build blocks after deciding presentation plan with content intelligence."""
    table = extract_markdown_table(text)
    timeline_events = extract_timeline_events(text)
    card_fields = extract_card_fields(text)

    plan = detect_content_plan(
        user_query=user_query,
        text=text,
        key_points=key_points,
        has_table=bool(table),
        has_timeline_events=bool(timeline_events),
        has_card_fields=bool(card_fields),
    )

    blocks: List[Dict[str, Any]] = []

    def append_text() -> None:
        if answer:
            blocks.append({"type": "text", "title": "Answer", "content": answer})

    def append_list() -> None:
        if key_points:
            blocks.append({"type": "list", "title": "Key Points", "items": key_points[:8]})

    def append_table() -> None:
        if table:
            columns, rows = table
            blocks.append({"type": "table", "title": "Table", "columns": columns, "rows": rows})

    def append_timeline() -> None:
        if timeline_events:
            blocks.append({"type": "timeline", "title": "Timeline", "events": timeline_events})

    def append_card() -> None:
        if card_fields:
            blocks.append({"type": "card", "title": "Details", "fields": card_fields})

    appenders = {
        "text": append_text,
        "list": append_list,
        "table": append_table,
        "timeline": append_timeline,
        "card": append_card,
    }

    primary = plan.get("primary", "text")
    secondary = [s for s in (plan.get("secondary") or []) if s != primary]

    if primary in appenders:
        appenders[primary]()
    for secondary_type in secondary:
        if secondary_type in appenders:
            appenders[secondary_type]()

    if not blocks:
        append_text()
        append_list()
        append_table()
        append_timeline()
        append_card()

    normalized = normalize_blocks(blocks)
    return normalized, plan


def build_blocks(answer: str, key_points: List[str], text: str, user_query: str = "") -> List[Dict[str, Any]]:
    """Compatibility wrapper returning only blocks."""
    blocks, _plan = build_blocks_with_intelligence(
        answer=answer,
        key_points=key_points,
        text=text,
        user_query=user_query,
    )
    return blocks
