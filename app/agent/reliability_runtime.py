from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "for", "of", "in", "on", "at",
    "what", "which", "who", "where", "when", "why", "how", "please", "tell", "me", "about",
    "latest", "newest",
}


def confidence_to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"high", "verified"}:
        return 0.9
    if normalized in {"medium", "moderate"}:
        return 0.7
    if normalized in {"low", "unverified"}:
        return 0.3
    try:
        return float(normalized)
    except Exception:
        return 0.0


def normalize_intent_signature(query: str) -> str:
    lowered = (query or "").strip().lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    filtered = [tok for tok in tokens if tok not in _STOPWORDS]
    if not filtered:
        filtered = tokens[:]
    compact = " ".join(filtered[:24]).strip()
    return compact


def semantic_intent_hash(query: str) -> str:
    import hashlib

    normalized = normalize_intent_signature(query)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def is_current_info_query(goal: str) -> bool:
    q = (goal or "").lower()
    markers = [
        "today",
        "latest",
        "recent",
        "current",
        "now",
        "breaking",
        "this week",
        "this month",
        "new data",
        "updated",
        "update",
        "as of",
        "price",
        "stock",
        "news",
        "score",
    ]
    if any(m in q for m in markers):
        return True
    years = re.findall(r"\b(20\d{2})\b", q)
    if years:
        try:
            now_year = datetime.now(timezone.utc).year
            return any(int(y) >= (now_year - 1) for y in years)
        except Exception:
            return True
    return False


def needs_tool_first(query_type: str, goal: str) -> bool:
    q = (goal or "").lower()
    if is_current_info_query(goal):
        return True
    if query_type in {"simple_fact", "research", "comparison"}:
        return True
    if any(k in q for k in ("compare", "research", "verify", "source", "citation", "market", "analysis")):
        return True
    return False


def classify_reliability_query_type(query_type: str, goal: str) -> str:
    qt = str(query_type or "").strip().lower()
    q = (goal or "").lower()
    if is_current_info_query(goal):
        return "current_info"
    if qt in {"research", "comparison", "simple_fact"}:
        return "factual" if qt == "simple_fact" else qt
    if any(k in q for k in ("compare", "versus", "vs ", "trade-off", "benchmark")):
        return "comparison"
    if any(k in q for k in ("research", "citation", "source", "evidence", "verify")):
        return "research"
    if any(k in q for k in ("design", "implement", "debug", "code", "fix")):
        return "reasoning"
    if any(k in q for k in ("story", "poem", "creative", "song", "lyrics")):
        return "creative"
    if len(q.split()) <= 5 and any(k in q for k in ("hi", "hello", "hey", "thanks")):
        return "casual"
    return "reasoning"


def query_classifier_confidence(query_type: str, goal: str) -> float:
    qt = classify_reliability_query_type(query_type, goal)
    q = (goal or "").strip().lower()
    if qt in {"current_info", "comparison", "research"}:
        return 0.82
    if qt == "factual":
        return 0.72
    if qt in {"casual", "creative"}:
        return 0.78
    # Ambiguous/short prompts are lower confidence.
    if len(q.split()) < 4:
        return 0.58
    return 0.66


def should_enforce_evidence(query_type: str, goal: str) -> bool:
    qt = classify_reliability_query_type(query_type, goal)
    return qt in {"factual", "current_info", "comparison", "research"}


def source_provider_from_url(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower().strip()
        if host.startswith("www."):
            host = host[4:]
        parts = [p for p in host.split(".") if p]
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host or "unknown"
    except Exception:
        return "unknown"


def parse_timestamp_any(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(raw).astimezone(timezone.utc)
    except Exception:
        return None


def detect_unsupported_claims(
    claims: List[Dict[str, Any]],
    claim_sources: Dict[str, List[str]],
    min_claim_confidence: float = 0.6,
) -> List[Dict[str, Any]]:
    unsupported: List[Dict[str, Any]] = []
    for claim in claims:
        text = str(claim.get("claim") or claim.get("value") or "").strip()
        if not text:
            continue
        conf = confidence_to_float(claim.get("confidence"))
        if conf <= min_claim_confidence:
            continue
        key = text[:180]
        supporting = claim_sources.get(key, [])
        if not supporting:
            unsupported.append({"claim": text[:320], "confidence": conf, "reason": "no_supporting_sources"})
    return unsupported


def evaluate_evidence_quality(
    goal: str,
    query_type: str,
    sources: List[Dict[str, Any]],
    verification_result: Dict[str, Any],
    recency_days: int = 180,
) -> Dict[str, Any]:
    source_count = 0
    providers: Set[str] = set()
    recent_count = 0
    now = datetime.now(timezone.utc)
    for src in sources or []:
        url = str(src.get("url") or "").strip()
        if not url:
            continue
        source_count += 1
        providers.add(source_provider_from_url(url))
        ts = parse_timestamp_any(src.get("timestamp"))
        if ts is not None:
            age_days = max(0, (now - ts).days)
            if age_days <= int(recency_days):
                recent_count += 1

    provider_count = len([p for p in providers if p and p != "unknown"])
    verified = len(verification_result.get("verified_claims", []) if isinstance(verification_result, dict) else [])
    uncertain = len(verification_result.get("uncertain_claims", []) if isinstance(verification_result, dict) else [])
    conflict = len(verification_result.get("conflicted_claims", []) if isinstance(verification_result, dict) else [])

    agreement_ok = verified > 0 and conflict == 0
    quality = "high" if (verified >= 2 and source_count >= 3) else ("medium" if (verified >= 1 or uncertain >= 2) else "low")
    reasons: List[str] = []

    if source_count < 2:
        reasons.append("insufficient_sources")
    if provider_count < 2:
        reasons.append("insufficient_provider_diversity")
    if not agreement_ok:
        reasons.append("agreement_failed")
    if quality == "low":
        reasons.append("evidence_quality_low")

    current_info = is_current_info_query(goal) or query_type in {"current_info"}
    recency_ok = True
    if current_info:
        recency_ok = recent_count >= 1
        if not recency_ok:
            reasons.append("recency_failed")

    passed = source_count >= 2 and provider_count >= 2 and agreement_ok and quality in {"medium", "high"} and recency_ok
    return {
        "pass": passed,
        "source_count": source_count,
        "provider_count": provider_count,
        "agreement": agreement_ok,
        "quality": quality,
        "recency_ok": recency_ok,
        "current_info": current_info,
        "reasons": reasons,
    }


def freshness_decay(age_seconds: float, ttl_seconds: int) -> float:
    if ttl_seconds <= 0:
        return 0.0
    age = max(0.0, float(age_seconds))
    return float(math.exp(-(age / float(ttl_seconds))))
