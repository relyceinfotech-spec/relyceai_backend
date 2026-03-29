from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(str(text)) / 4))


@dataclass
class SegmentedContext:
    messages: List[Dict[str, str]]
    estimated_tokens: int
    compressed_tool_items: int
    dropped_messages: int


def _compact_tool_results(tool_results: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in tool_results[:limit]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "tool": str(item.get("tool") or "unknown"),
                "status": str(item.get("status") or "unknown"),
                "data": str(item.get("data") or "")[:240],
            }
        )
    return out


def build_segmented_context(
    *,
    system_prompt: str,
    user_query: str,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    tool_results: Optional[List[Dict[str, Any]]] = None,
    max_recent_messages: int = 8,
    max_total_estimated_tokens: int = 5000,
) -> SegmentedContext:
    context_messages = list(context_messages or [])
    tool_results = list(tool_results or [])
    recent = context_messages[-int(max_recent_messages) :]
    dropped_messages = max(0, len(context_messages) - len(recent))
    compact_tools = _compact_tool_results(tool_results, limit=3)

    messages: List[Dict[str, str]] = [{"role": "system", "content": str(system_prompt or "")}]
    if dropped_messages > 0:
        messages.append(
            {
                "role": "system",
                "content": f"[Context trimmed] {dropped_messages} older messages omitted for efficiency.",
            }
        )
    for msg in recent:
        messages.append({"role": str(msg.get("role") or "user"), "content": str(msg.get("content") or "")[:1200]})
    if compact_tools:
        messages.append({"role": "system", "content": f"[Tool summary] {compact_tools}"})
    messages.append({"role": "user", "content": str(user_query or "")})

    est = sum(estimate_tokens(m.get("content", "")) for m in messages)
    # Keep trimming oldest non-system message until under budget.
    while est > int(max_total_estimated_tokens) and len(messages) > 3:
        # remove first non-system message after first element
        removed = False
        for idx in range(1, len(messages) - 1):
            if messages[idx]["role"] != "system":
                messages.pop(idx)
                dropped_messages += 1
                removed = True
                break
        if not removed:
            break
        est = sum(estimate_tokens(m.get("content", "")) for m in messages)

    return SegmentedContext(
        messages=messages,
        estimated_tokens=est,
        compressed_tool_items=len(compact_tools),
        dropped_messages=dropped_messages,
    )

