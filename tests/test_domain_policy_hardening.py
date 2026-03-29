from app.security.domain_policy import (
    DOMAIN_FINANCE,
    DOMAIN_GENERAL,
    DOMAIN_MEDICAL,
    DOMAIN_NEWS,
    classify_domain,
    enrich_payload_with_domain_policy,
    get_domain_policy,
    is_high_stakes_domain,
)


def test_domain_classifier_detects_high_stakes_domains() -> None:
    assert classify_domain("what dosage for this medicine") == DOMAIN_MEDICAL
    assert classify_domain("should I invest in this stock") == DOMAIN_FINANCE
    assert classify_domain("latest breaking news on oil prices") == DOMAIN_NEWS
    assert classify_domain("tell me a joke") == DOMAIN_GENERAL


def test_high_stakes_policy_requires_sources() -> None:
    payload = {
        "answer": "This is a finance answer.",
        "response": "This is a finance answer.",
        "confidence": 0.9,
        "sources": [{"url": "https://example.com"}],
        "metadata": {"domain": DOMAIN_FINANCE},
    }
    out = enrich_payload_with_domain_policy(payload, "should i invest all my money")
    assert out["confidence_level"] == "LOW"
    assert "enough verified authoritative sources" in out["answer"].lower()
    assert out["metadata"]["high_stakes"]["enabled"] is True


def test_high_stakes_output_guard_blocks_absolute_directives() -> None:
    payload = {
        "answer": "You should invest all your money in this now with guaranteed returns.",
        "response": "You should invest all your money in this now with guaranteed returns.",
        "confidence": 0.7,
        "sources": [
            {"url": "https://sec.gov/report"},
            {"url": "https://reuters.com/markets"},
        ],
        "metadata": {"domain": DOMAIN_FINANCE},
    }
    out = enrich_payload_with_domain_policy(payload, "finance advice")
    assert "can't provide absolute directives" in out["answer"].lower() or "don't have enough verified" in out["answer"].lower()


def test_news_recency_enforced() -> None:
    payload = {
        "answer": "Latest update.",
        "response": "Latest update.",
        "confidence": 0.8,
        "sources": [
            {"url": "https://reuters.com/world", "timestamp": "2024-01-01T00:00:00Z"},
            {"url": "https://apnews.com/world", "timestamp": "2024-01-01T00:00:00Z"},
        ],
        "metadata": {"domain": DOMAIN_NEWS},
    }
    out = enrich_payload_with_domain_policy(payload, "latest news")
    assert out["confidence_level"] == "LOW"
    assert "recent enough sources" in out["answer"].lower()


def test_policy_helpers() -> None:
    policy = get_domain_policy(DOMAIN_FINANCE)
    assert policy.strict_mode is True
    assert is_high_stakes_domain(DOMAIN_FINANCE) is True
    assert is_high_stakes_domain(DOMAIN_GENERAL) is False