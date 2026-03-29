from __future__ import annotations

import time

import app.agent.tool_memory_store as tms
from app.agent.reliability_runtime import detect_unsupported_claims, evaluate_evidence_quality, is_current_info_query


def test_detect_unsupported_claims_flags_high_conf_without_sources():
    claims = [
        {"claim": "Redis latest version is 7.2", "confidence": "high"},
        {"claim": "It is popular", "confidence": "low"},
    ]
    claim_sources = {"It is popular": ["https://example.com"]}
    unsupported = detect_unsupported_claims(claims, claim_sources, min_claim_confidence=0.6)
    assert len(unsupported) == 1
    assert "Redis latest version" in unsupported[0]["claim"]


def test_evidence_validator_requires_provider_diversity_and_recency():
    now_iso = "2026-03-13T12:00:00+00:00"
    sources = [
        {"url": "https://news.a.com/post/1", "timestamp": now_iso},
        {"url": "https://www.a.com/post/2", "timestamp": now_iso},  # same provider => should fail diversity
    ]
    verification = {"verified_claims": [{"value": "x"}], "uncertain_claims": [], "conflicted_claims": []}
    out = evaluate_evidence_quality(
        goal="latest redis release",
        query_type="current_info",
        sources=sources,
        verification_result=verification,
        recency_days=180,
    )
    assert out["pass"] is False
    assert "insufficient_provider_diversity" in out["reasons"]
    assert is_current_info_query("latest redis release") is True


def test_tool_memory_store_put_get_and_bounds(monkeypatch):
    monkeypatch.setattr(tms, "TOOL_MEMORY_MAX_ITEMS_PER_USER", 2)
    monkeypatch.setattr(tms, "TOOL_MEMORY_MAX_ITEMS_PER_SESSION", 1)
    monkeypatch.setattr(tms, "TOOL_MEMORY_TTL_SECONDS", 3600)
    monkeypatch.setattr(tms, "TOOL_MEMORY_FRESHNESS_MIN", 0.1)

    store = tms.ToolMemoryStore()
    user_id = "u1"
    session_id = "s1"

    store.put(
        user_id=user_id,
        session_id=session_id,
        query="redis latest version",
        tool_name="search_web",
        fingerprint="fp1",
        key_facts=["Redis is at x.y"],
        source_links=["https://redis.io"],
        confidence=0.8,
    )
    hit = store.get(
        user_id=user_id,
        session_id=session_id,
        query="what is the newest redis version",
        tool_name="search_web",
        current_info=False,
    )
    # normalized signature should still hit
    assert hit is not None
    assert hit.fingerprint == "fp1"

    # bounds prune
    store.put(
        user_id=user_id,
        session_id=session_id,
        query="kafka latest version",
        tool_name="search_web",
        fingerprint="fp2",
        key_facts=["Kafka is at a.b"],
        source_links=["https://kafka.apache.org"],
        confidence=0.8,
    )
    store.put(
        user_id=user_id,
        session_id="s2",
        query="nginx latest version",
        tool_name="search_web",
        fingerprint="fp3",
        key_facts=["Nginx is at c.d"],
        source_links=["https://nginx.org"],
        confidence=0.8,
    )
    # max per user=2 should be enforced
    assert len(store._cache) <= 2

    # freshness decay path
    for entry in store._cache.values():
        entry.timestamp = time.time() - 10
        break
