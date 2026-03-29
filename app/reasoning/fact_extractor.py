from __future__ import annotations

from typing import Any, Dict, List


def _parse_year(text: str) -> str:
    for token in (text or "").replace(",", " ").split():
        if token.isdigit() and len(token) == 4:
            return token
    return ""


def extract_facts(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or item.get("fact") or item.get("evidence_span") or "").strip()
        if not value:
            continue
        facts.append(
            {
                "entity": str(item.get("entity") or value[:80]),
                "attribute": str(item.get("attribute") or "extracted_fact"),
                "value": value,
                "time": str(item.get("time") or _parse_year(value)),
                "source": str(item.get("source") or "unknown"),
            }
        )
    return facts


def extract_facts_from_workspace(workspace: Any) -> List[Dict[str, Any]]:
    return extract_facts(list(getattr(workspace, "facts", []) or []))


def extract_claims(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "claim_id": f"claim_{idx}",
            "entity": str(fact.get("entity", "")),
            "attribute": str(fact.get("attribute", "")),
            "value": str(fact.get("value", "")),
            "time": str(fact.get("time", "")),
            "source": str(fact.get("source", "unknown")),
        }
        for idx, fact in enumerate(facts, start=1)
    ]
