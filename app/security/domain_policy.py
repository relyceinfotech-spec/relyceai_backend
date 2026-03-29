from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from app.telemetry.metrics_collector import get_metrics_collector

DOMAIN_GENERAL = "general"
DOMAIN_FINANCE = "finance"
DOMAIN_MEDICAL = "medical"
DOMAIN_NEWS = "news_current_events"
DOMAIN_LEGAL = "legal"


@dataclass(frozen=True)
class DomainPolicy:
    strict_mode: bool
    min_sources: int
    recency_hours: int


_POLICIES = {
    DOMAIN_GENERAL: DomainPolicy(strict_mode=False, min_sources=0, recency_hours=0),
    DOMAIN_FINANCE: DomainPolicy(strict_mode=True, min_sources=2, recency_hours=0),
    DOMAIN_MEDICAL: DomainPolicy(strict_mode=True, min_sources=2, recency_hours=72),
    DOMAIN_NEWS: DomainPolicy(strict_mode=True, min_sources=2, recency_hours=24),
    DOMAIN_LEGAL: DomainPolicy(strict_mode=True, min_sources=2, recency_hours=168),
}

_ABSOLUTE_DIRECTIVE_PATTERNS = (
    "guaranteed return",
    "invest all your money",
    "must buy now",
    "always do",
)

_AUTHORITATIVE_HINTS = (
    "reuters.com",
    "apnews.com",
    "sec.gov",
    "ft.com",
    "wsj.com",
    "who.int",
    "nih.gov",
    "cdc.gov",
    "gov",
)


def classify_domain(query: str) -> str:
    q = str(query or "").lower()
    if any(k in q for k in ("dosage", "medicine", "medical", "symptom", "doctor")):
        return DOMAIN_MEDICAL
    if any(k in q for k in ("invest", "stock", "finance", "portfolio", "trading", "market")):
        return DOMAIN_FINANCE
    if any(k in q for k in ("latest", "breaking", "news", "today", "current events")):
        return DOMAIN_NEWS
    if any(k in q for k in ("legal", "law", "contract", "court", "rights")):
        return DOMAIN_LEGAL
    return DOMAIN_GENERAL


def get_domain_policy(domain: str) -> DomainPolicy:
    return _POLICIES.get(str(domain or "").strip().lower(), _POLICIES[DOMAIN_GENERAL])


def is_high_stakes_domain(domain: str) -> bool:
    policy = get_domain_policy(domain)
    return bool(policy.strict_mode)


def _source_is_authoritative(src: Dict[str, Any]) -> bool:
    url = str(src.get("url") or src.get("link") or "").lower()
    return any(hint in url for hint in _AUTHORITATIVE_HINTS)


def _source_is_recent(src: Dict[str, Any], hours: int) -> bool:
    if hours <= 0:
        return True
    ts = str(src.get("timestamp") or src.get("published_at") or "").strip()
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return False
    return dt >= (datetime.now(timezone.utc) - timedelta(hours=hours))


def enrich_payload_with_domain_policy(payload: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    out = dict(payload or {})
    metadata = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    domain = str(metadata.get("domain") or classify_domain(user_query)).strip().lower()
    metadata["domain"] = domain
    policy = get_domain_policy(domain)

    response = str(out.get("response") or out.get("answer") or "").strip()
    sources = out.get("sources") if isinstance(out.get("sources"), list) else []
    normalized_sources = [s for s in sources if isinstance(s, dict)]

    source_met = len([s for s in normalized_sources if _source_is_authoritative(s)]) >= policy.min_sources
    recency_met = all(_source_is_recent(s, policy.recency_hours) for s in normalized_sources[: max(policy.min_sources, 1)])
    absolute_directive = any(p in response.lower() for p in _ABSOLUTE_DIRECTIVE_PATTERNS)

    confidence_level = "HIGH"
    warning = ""
    if policy.strict_mode and (not source_met):
        confidence_level = "LOW"
        warning = "I don't have enough verified authoritative sources to provide that level of certainty."
    if policy.strict_mode and source_met and not recency_met:
        confidence_level = "LOW"
        warning = "I don't have recent enough sources to support a reliable current-events answer."
    if policy.strict_mode and absolute_directive:
        confidence_level = "LOW"
        warning = "I can't provide absolute directives in this high-stakes domain."

    if warning:
        response = warning
        out["answer"] = warning
        out["response"] = warning

    metadata["high_stakes"] = {"enabled": bool(policy.strict_mode)}
    out["metadata"] = metadata
    out["confidence_level"] = confidence_level

    collector = get_metrics_collector()
    collector.record_high_stakes_event(domain)
    return out
