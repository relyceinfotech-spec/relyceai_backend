"""Universal response adapter that normalizes raw model/agent output."""
from __future__ import annotations

from typing import Any, Dict

from app.llm.output_postprocess import sanitize_output_text
from app.safety.safety_filter import check_response_safety
from app.chat.mode_mapper import normalize_chat_mode, is_structured_mode

from .schema import SCHEMA_VERSION, base_metadata
from .block_builder import (
    build_blocks_with_intelligence,
    extract_answer,
    extract_key_points,
    infer_answer_type,
    normalize_blocks,
    normalize_sources,
    strip_internal_leakage,
    to_str_list,
)


def normalize_chat_response(raw_result: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
    """Convert internal response shape to the stable universal schema contract."""
    result = dict(raw_result or {})
    raw_text = str(result.get("response") or result.get("answer") or "").strip()
    text = strip_internal_leakage(sanitize_output_text(raw_text))
    answer = extract_answer(text)
    raw_points = to_str_list(result.get("key_points")) or extract_key_points(text)

    key_points = []
    for point in raw_points:
        cleaned_point = strip_internal_leakage(point)
        if cleaned_point:
            key_points.append(cleaned_point)

    sources = normalize_sources(result.get("sources"))
    confidence_raw = result.get("confidence")
    confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
    confidence = max(0.0, min(1.0, confidence))
    confidence_level = "HIGH" if confidence >= 0.8 else "MODERATE" if confidence >= 0.55 else "LOW"
    answer_type = str(result.get("answer_type") or infer_answer_type(user_query, text)).strip() or "summary"

    warning = check_response_safety(answer)
    if warning:
        answer = f"{answer}\n\n{warning}".strip()
        text = answer
        result["security_warning"] = warning

    existing_metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    metadata = base_metadata(existing_metadata)

    provided_blocks = normalize_blocks(result.get("blocks"))
    intelligence_plan: Dict[str, Any] = {}
    if provided_blocks:
        blocks = provided_blocks
    else:
        blocks, intelligence_plan = build_blocks_with_intelligence(
            answer=answer,
            key_points=key_points,
            text=text,
            user_query=user_query,
        )

    if intelligence_plan:
        metadata["content_intelligence"] = intelligence_plan

    structured_response = {
        "schema_version": SCHEMA_VERSION,
        "summary": answer,
        "answer": answer,
        "key_points": key_points,
        "sources": sources,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "answer_type": answer_type,
        "metadata": metadata,
        "blocks": blocks,
    }

    result["response"] = text
    result["schema_version"] = SCHEMA_VERSION
    result["summary"] = answer
    result["answer"] = answer
    result["key_points"] = key_points
    result["sources"] = sources
    result["confidence"] = confidence
    result["confidence_level"] = confidence_level
    result["answer_type"] = answer_type
    result["metadata"] = metadata
    result["blocks"] = blocks
    result["structured_response"] = structured_response
    return result


def build_final_answer_payload(raw_result: Dict[str, Any], user_query: str = "", chat_mode: str = "smart") -> Dict[str, Any]:
    """Mode-aware final payload: plain for normal chat, structured for agent/research."""
    normalized = normalize_chat_response(raw_result, user_query=user_query)
    mode = normalize_chat_mode(str(chat_mode or "smart"))
    answer = str(normalized.get("answer") or "").strip()
    if not answer:
        answer = str(normalized.get("response") or "").strip()
    confidence = normalized.get("confidence", 0.0)
    confidence_level = "HIGH" if confidence >= 0.8 else "MODERATE" if confidence >= 0.55 else "LOW"

    if is_structured_mode(mode):
        structured = dict(normalized.get("structured_response", {}) or {})
        structured["answer"] = answer
        structured.setdefault("summary", answer)
        metadata = structured.get("metadata") if isinstance(structured.get("metadata"), dict) else {}
        metadata.setdefault("mode", mode)
        structured["metadata"] = metadata
        structured.setdefault("confidence_level", confidence_level)
        return structured

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "answer_type": "plain",
        "summary": answer,
        "answer": answer,
        "response": normalized.get("response", ""),
        "key_points": [],
        "sources": [],
        "confidence": confidence,
        "confidence_level": confidence_level,
        "metadata": normalized.get("metadata", {}),
        "blocks": [],
    }
