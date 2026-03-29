"""
Evidence Engine

Produces structured facts, extracted claims, verification decisions,
and ranked evidence for synthesis.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


def _parse_time_hint(text: str) -> str:
    text = text or ""
    for token in text.replace("?", " ").replace(",", " ").split():
        if token.isdigit() and len(token) == 4:
            return token
    return ""


def extract_facts_from_workspace(workspace: Any) -> List[Dict[str, Any]]:
    """
    Normalize workspace raw findings into structured fact records.
    Supports both legacy {fact: ...} and structured {value: ...} shapes.
    """
    facts: List[Dict[str, Any]] = []

    for f in list(getattr(workspace, "facts", []) or []):
        if not isinstance(f, dict):
            continue

        value_text = str(f.get("value", "")).strip()
        raw_fact = str(f.get("fact", "")).strip()
        evidence_span = str(f.get("evidence_span", "")).strip()
        canonical_text = value_text or raw_fact or evidence_span
        if not canonical_text:
            continue

        source = str(f.get("source", "unknown"))
        trust = float(f.get("source_trust", f.get("trust_score", f.get("trust", 0.5))))
        confidence = str(f.get("confidence", "medium"))

        facts.append(
            {
                "entity": f.get("entity") or canonical_text[:80],
                "attribute": f.get("attribute") or "extracted_fact",
                "value": canonical_text,
                "time": f.get("time") or _parse_time_hint(canonical_text),
                "source": source,
                "source_trust": trust,
                "evidence_span": evidence_span or canonical_text[:500],
                "confidence": confidence,
            }
        )

    return facts


def extract_claims_from_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for idx, fact in enumerate(facts):
        claim_text = str(fact.get("value", "")).strip()
        if not claim_text:
            continue
        claims.append(
            {
                "claim_id": f"c_{idx+1}",
                "text": claim_text,
                "entity": fact.get("entity", ""),
                "attribute": fact.get("attribute", ""),
                "expected_time": fact.get("time", ""),
                "source": fact.get("source", "unknown"),
            }
        )
    return claims


def _agreement_ratio(claim: Dict[str, Any], facts: List[Dict[str, Any]]) -> float:
    target = str(claim.get("text", "")).strip().lower()
    if not target:
        return 0.0

    comparable = [
        str(f.get("value", "")).strip().lower()
        for f in facts
        if str(f.get("entity", "")).strip().lower() == str(claim.get("entity", "")).strip().lower()
    ]
    if not comparable:
        return 0.0

    matches = sum(1 for c in comparable if c == target)
    return matches / max(1, len(comparable))


def verify_claims(claims: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    verified: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []
    conflicted: List[Dict[str, Any]] = []

    for claim in claims:
        c_text = str(claim.get("text", "")).strip().lower()
        if not c_text:
            uncertain.append({**claim, "reason": "empty_claim"})
            continue

        supporting = [f for f in facts if str(f.get("value", "")).strip().lower() == c_text]
        opposing = [f for f in facts if str(f.get("entity", "")).strip().lower() == str(claim.get("entity", "")).strip().lower() and str(f.get("value", "")).strip().lower() != c_text]

        if supporting and not opposing:
            verified.append({**claim, "support_count": len(supporting)})
        elif supporting and opposing:
            conflicted.append({**claim, "support_count": len(supporting), "opposition_count": len(opposing)})
        else:
            uncertain.append({**claim, "reason": "no_support"})

    total = max(1, len(claims))
    verification_score = (len(verified) + 0.5 * len(uncertain)) / total

    return {
        "verified_claims": verified,
        "uncertain_claims": uncertain,
        "conflicted_claims": conflicted,
        "overall_confidence": round(float(verification_score), 4),
        "followup_triggered": len(conflicted) > 0,
    }


def _recency_factor(source_item: Dict[str, Any]) -> float:
    ts = source_item.get("updated_at") or source_item.get("timestamp")
    if not ts:
        return 0.7
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - dt).days
        if age_days <= 7:
            return 1.0
        if age_days <= 30:
            return 0.9
        if age_days <= 180:
            return 0.75
        return 0.6
    except Exception:
        return 0.7


def rank_evidence(
    facts: List[Dict[str, Any]],
    verification_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    score = source_trust * support_count * recency_factor * agreement_ratio
    """
    support_counter: Dict[str, int] = defaultdict(int)
    for fact in facts:
        support_counter[str(fact.get("value", "")).strip().lower()] += 1

    verified_lookup = {
        str(c.get("text", "")).strip().lower(): c for c in verification_result.get("verified_claims", [])
    }

    ranked: List[Dict[str, Any]] = []
    for fact in facts:
        val = str(fact.get("value", "")).strip().lower()
        if not val:
            continue

        claim_proxy = verified_lookup.get(val, {"text": val, "entity": fact.get("entity", "")})
        agreement = _agreement_ratio(claim_proxy, facts)
        support_count = support_counter.get(val, 1)
        trust = float(fact.get("source_trust", 0.5))
        recency = _recency_factor(fact)

        score = trust * float(support_count) * recency * max(0.01, agreement)
        ranked.append({**fact, "ranking_score": round(score, 6), "agreement_ratio": round(agreement, 4), "support_count": support_count})

    ranked.sort(key=lambda x: x.get("ranking_score", 0.0), reverse=True)
    return ranked


def confidence_progress(verification_result: Dict[str, Any], ranked_facts: List[Dict[str, Any]]) -> float:
    base = float(verification_result.get("overall_confidence", 0.0))
    if not ranked_facts:
        return round(base, 3)
    top_avg = sum(float(f.get("ranking_score", 0.0)) for f in ranked_facts[:3]) / min(3, len(ranked_facts))
    # compress to [0,1]
    normalized_top = min(1.0, top_avg)
    return round((0.7 * base) + (0.3 * normalized_top), 3)
