from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def verify_claims(claims: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    verified_claims: List[Dict[str, Any]] = []
    uncertain_claims: List[Dict[str, Any]] = []
    conflicted_claims: List[Dict[str, Any]] = []

    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        key = (str(fact.get("entity", "")).lower(), str(fact.get("attribute", "")).lower())
        grouped[key].append(fact)

    for claim in claims:
        entity = str(claim.get("entity", "")).lower()
        attribute = str(claim.get("attribute", "")).lower()
        value = str(claim.get("value", "")).strip().lower()
        cluster = grouped.get((entity, attribute), [])
        supporting = [f for f in cluster if str(f.get("value", "")).strip().lower() == value]
        conflicting = [f for f in cluster if str(f.get("value", "")).strip().lower() != value]
        sources = sorted({str(f.get("source", "unknown")) for f in supporting if str(f.get("source", "")).strip()})
        enriched = {
            **claim,
            "supporting_sources": sources,
            "support_count": len(sources),
            "conflict_count": len(conflicting),
        }
        if len(sources) >= 2 and not conflicting:
            enriched["status"] = "verified"
            verified_claims.append(enriched)
        elif conflicting:
            enriched["status"] = "conflicted"
            conflicted_claims.append(enriched)
        else:
            enriched["status"] = "uncertain"
            uncertain_claims.append(enriched)

    total = max(1, len(claims))
    confidence = (len(verified_claims) + 0.5 * len(uncertain_claims)) / total
    return {
        "verified_claims": verified_claims,
        "uncertain_claims": uncertain_claims,
        "conflicted_claims": conflicted_claims,
        "confidence": round(float(confidence), 4),
    }
