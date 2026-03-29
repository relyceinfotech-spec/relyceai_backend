import pytest
import datetime
import time

from app.research.source_ranker import compute_trust_score, enforce_domain_diversity
from app.research.recency import compute_recency_score, final_source_score, parse_date_heuristic
from app.research.chunker import split_by_headers, cap_chunk_length, chunk_text
from app.research.conflict_detector import detect_conflicts, adjust_confidence_for_conflicts
from app.research.cache import get_cached, set_cache, clear_cache, TTL_SECONDS

# ============================================
# TRUST SCORE MAPPING
# ============================================

def test_trust_score_mapping():
    # Official docs get max trust
    assert compute_trust_score("https://react.dev/reference/react") == 1.0
    assert compute_trust_score("https://docs.python.org/3/library/os.html") == 1.0
    
    # Low trust domains are penalized
    assert compute_trust_score("https://medium.com/@user/how-to-code") == 0.4
    assert compute_trust_score("https://dev.to/someuser/article") == 0.4
    
    # Unknown domains get neutral
    assert compute_trust_score("https://some-random-blog.com/post") == 0.6


def test_domain_diversity_filter():
    results = [
        {"url": "https://medium.com/1"},
        {"url": "https://medium.com/2"},
        {"url": "https://react.dev/1"},
        {"url": "https://github.com/1"},
        {"url": "https://medium.com/3"},
    ]
    
    unique = enforce_domain_diversity(results, max_results=3, max_per_domain=1)
    assert len(unique) == 3
    domains = [r["url"] for r in unique]
    assert "https://medium.com/1" in domains
    assert "https://react.dev/1" in domains
    assert "https://github.com/1" in domains
    assert "https://medium.com/2" not in domains


# ============================================
# RECENCY SCORING
# ============================================

def test_recency_scoring():
    today = datetime.date.today()
    
    # Very recent (<30 days)
    recent = today - datetime.timedelta(days=10)
    assert compute_recency_score(recent) == 1.0
    
    # Somewhat recent (<180 days)
    mid = today - datetime.timedelta(days=100)
    assert compute_recency_score(mid) == 0.8
    
    # Old (>365 days)
    old = today - datetime.timedelta(days=400)
    assert compute_recency_score(old) == 0.4
    
    # No date fallback
    assert compute_recency_score(None) == 0.5


def test_date_parsing_heuristic():
    assert parse_date_heuristic("2023-10-01T15:30:00Z") == datetime.date(2023, 10, 1)
    assert parse_date_heuristic("2024-05-15") == datetime.date(2024, 5, 15)
    assert parse_date_heuristic("Yesterday") is None


# ============================================
# HEADER-BASED CHUNK SPLITTING
# ============================================

def test_header_based_chunking():
    text = (
        "Introduction text here."
        "\n# Header 1\n"
        "Content 1."
        "\n## Header 2\n"
        "Content 2."
    )
    
    chunks = split_by_headers(text)
    assert len(chunks) == 3
    assert chunks[0] == "Introduction text here."
    assert chunks[1] == "# Header 1\nContent 1."
    assert chunks[2] == "## Header 2\nContent 2."


def test_chunk_capping():
    # A chunk that is too long should be capped, ideally at a space
    long_text = "A" * 1500 + " B" * 600
    capped = cap_chunk_length(long_text, max_chars=2000)
    assert len(capped) <= 2003 # 2000 + "..."


# ============================================
# CONFLICT DETECTION
# ============================================

def test_conflict_detection_trigger():
    # Contradicting signals across summaries
    summaries = [
        "The new API is stable and recommended for production.",
        "Warning: The new API is experimental and deprecated.",
    ]
    assert detect_conflicts(summaries) is True


def test_no_conflict_safe():
    summaries = [
        "The new API is stable.",
        "It has been in production for months.",
    ]
    assert detect_conflicts(summaries) is False


def test_confidence_adjustment():
    conflicts = True
    new_conf = adjust_confidence_for_conflicts(0.9, conflicts)
    assert new_conf == 0.7  # Drops by 0.2


# ============================================
# CACHE EXPIRATION
# ============================================

def test_cache_hit_and_miss():
    clear_cache()
    
    set_cache("test query", "cached result data")
    
    # Immediate hit
    assert get_cached("test query") == "cached result data"
    
    # Cache miss
    assert get_cached("unknown query") is None

def test_cache_expiration(monkeypatch):
    clear_cache()
    set_cache("expired query", "data")
    
    # Fast forward time past TTL
    future_time = time.time() + TTL_SECONDS + 10
    monkeypatch.setattr(time, "time", lambda: future_time)
    
    assert get_cached("expired query") is None
