from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse


TRUST_DOMAINS = {
    "gov": 0.95,
    "edu": 0.9,
    "official": 0.95,
    "news": 0.8,
    "blog": 0.5,
}


def _domain_type(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    if host.endswith(".gov"):
        return "gov"
    if host.endswith(".edu"):
        return "edu"
    if any(tag in host for tag in ["official", "nasa", "who.int", "un.org"]):
        return "official"
    if any(tag in host for tag in ["news", "reuters", "bloomberg", "bbc", "nytimes"]):
        return "news"
    return "blog"


def _source_trust(source: str) -> float:
    return float(TRUST_DOMAINS.get(_domain_type(source or ""), 0.5))


def _recency_factor(time_value: str) -> float:
    if not time_value:
        return 0.7
    now_year = datetime.now(timezone.utc).year
    t = str(time_value).strip()
    if t.isdigit() and len(t) == 4:
        age = max(0, now_year - int(t))
    else:
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            age = max(0, now_year - dt.year)
        except Exception:
            return 0.7
    if age == 0:
        return 1.0
    if age <= 1:
        return 0.95
    if age <= 3:
        return 0.85
    if age <= 10:
        return 0.7
    return 0.55


def _agreement_ratio(claim: Dict[str, Any], facts: List[Dict[str, Any]]) -> float:
    entity = str(claim.get("entity", "")).lower()
    attribute = str(claim.get("attribute", "")).lower()
    value = str(claim.get("value", "")).lower()
    group = [f for f in facts if str(f.get("entity", "")).lower() == entity and str(f.get("attribute", "")).lower() == attribute]
    if not group:
        return 0.0
    agreements = sum(1 for item in group if str(item.get("value", "")).lower() == value)
    return agreements / max(1, len(group))


def rank_evidence(facts: List[Dict[str, Any]], verification_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    support_by_value: Dict[str, int] = {}
    for fact in facts:
        key = str(fact.get("value", "")).strip().lower()
        if key:
            support_by_value[key] = support_by_value.get(key, 0) + 1

    ranked: List[Dict[str, Any]] = []
    for fact in facts:
        value_key = str(fact.get("value", "")).strip().lower()
        if not value_key:
            continue
        source = str(fact.get("source", ""))
        source_trust = _source_trust(source)
        support_count = float(max(1, support_by_value.get(value_key, 1)))
        recency = _recency_factor(str(fact.get("time", "")))
        agreement = _agreement_ratio(fact, facts)
        score = source_trust * support_count * recency * max(0.01, agreement)
        ranked.append(
            {
                **fact,
                "source_trust": round(source_trust, 4),
                "support_count": int(support_count),
                "recency_factor": round(recency, 4),
                "agreement_ratio": round(agreement, 4),
                "score": round(float(score), 6),
            }
        )
    ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return ranked


def confidence_progress(verification_result: Dict[str, Any], ranked_facts: List[Dict[str, Any]]) -> float:
    base = float(verification_result.get("confidence", 0.0))
    if not ranked_facts:
        return round(base, 4)
    top_scores = [float(item.get("score", 0.0)) for item in ranked_facts[:3]]
    normalized = min(1.0, sum(top_scores) / max(1, len(top_scores)))
    return round((0.7 * base) + (0.3 * normalized), 4)
