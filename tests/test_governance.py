"""
Phase 10: Production Cost & Rate Governance Tests
Verifies rate limiting, quota enforcement, token tracking,
spend guard, concurrent limits, daily reset, and no double-counting.
"""
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.governance.usage_store import UsageStore, UserUsage
from app.governance.rate_limiter import RateLimiter
from app.governance.quota_manager import QuotaManager, TIER_LIMITS
from app.governance.token_tracker import TokenTracker
from app.governance.spend_guard import SpendGuard


# ============================================
# USAGE STORE TESTS
# ============================================

def test_usage_store_create_user():
    """New user should be created with zero counters."""
    store = UsageStore()
    usage = store.get_or_create("u1")
    assert usage.user_id == "u1"
    assert usage.daily_requests == 0
    assert usage.daily_tokens == 0
    assert usage.concurrent_tasks == 0


def test_usage_store_increment_requests():
    """Request count and timestamp should be tracked."""
    store = UsageStore()
    store.increment_requests("u1")
    store.increment_requests("u1")
    usage = store.get_or_create("u1")
    assert usage.daily_requests == 2
    assert len(usage.request_timestamps) == 2


def test_usage_store_increment_tokens():
    """Token count should accumulate."""
    store = UsageStore()
    store.increment_tokens("u1", 100)
    store.increment_tokens("u1", 250)
    usage = store.get_or_create("u1")
    assert usage.daily_tokens == 350


def test_usage_store_concurrent_tasks():
    """Concurrent tasks should increment and decrement, floored at 0."""
    store = UsageStore()
    store.increment_concurrent("u1")
    store.increment_concurrent("u1")
    usage = store.get_or_create("u1")
    assert usage.concurrent_tasks == 2

    store.decrement_concurrent("u1")
    usage = store.get_or_create("u1")
    assert usage.concurrent_tasks == 1

    # Floor at 0
    store.decrement_concurrent("u1")
    store.decrement_concurrent("u1")
    usage = store.get_or_create("u1")
    assert usage.concurrent_tasks == 0


def test_usage_store_sliding_window():
    """Burst window should count only recent requests."""
    store = UsageStore()
    # Add timestamps in the past (> 60s ago)
    usage = store.get_or_create("u1")
    usage.request_timestamps = [time.time() - 120, time.time() - 90]
    # Add recent ones
    store.increment_requests("u1")
    count = store.get_recent_request_count("u1", window_seconds=60)
    assert count == 1  # Only the recent one


def test_usage_store_system_token_total():
    """System total should sum across all users."""
    store = UsageStore()
    store.increment_tokens("u1", 1000)
    store.increment_tokens("u2", 2000)
    store.increment_tokens("u3", 3000)
    assert store.get_system_token_total() == 6000


def test_usage_store_reset_user():
    """Reset should remove user state."""
    store = UsageStore()
    store.increment_requests("u1")
    store.increment_tokens("u1", 500)
    store.reset_user("u1")
    usage = store.get_or_create("u1")
    assert usage.daily_requests == 0
    assert usage.daily_tokens == 0


# ============================================
# RATE LIMITER TESTS
# ============================================

def test_rate_limiter_allows_under_limit():
    """User under burst limit should be allowed."""
    store = UsageStore()
    limiter = RateLimiter(max_per_minute=5, store=store)
    allowed, reason = limiter.check_and_record("u1")
    assert allowed is True
    assert reason is None


def test_rate_limiter_blocks_at_limit():
    """User hitting burst limit should be blocked."""
    store = UsageStore()
    limiter = RateLimiter(max_per_minute=3, store=store)

    for _ in range(3):
        allowed, _ = limiter.check_and_record("u1")
        assert allowed is True

    # 4th request should be blocked
    allowed, reason = limiter.check_and_record("u1")
    assert allowed is False
    assert reason == "RATE_LIMIT_EXCEEDED"


def test_rate_limiter_per_user_isolation():
    """Rate limits should be per-user, not global."""
    store = UsageStore()
    limiter = RateLimiter(max_per_minute=2, store=store)

    limiter.check_and_record("u1")
    limiter.check_and_record("u1")
    # u1 is now at limit

    # u2 should still be allowed
    allowed, _ = limiter.check_and_record("u2")
    assert allowed is True


# ============================================
# QUOTA MANAGER TESTS
# ============================================

def test_quota_allows_within_limits():
    """User within tier limits should be allowed."""
    store = UsageStore()
    quota = QuotaManager(store=store)
    allowed, reason = quota.check("u1", tier="free")
    assert allowed is True
    assert reason is None


def test_quota_blocks_daily_request_limit():
    """User exceeding daily request limit should be blocked."""
    store = UsageStore()
    quota = QuotaManager(store=store)

    # Simulate hitting the free tier limit (50)
    usage = store.get_or_create("u1")
    usage.daily_requests = 50

    allowed, reason = quota.check("u1", tier="free")
    assert allowed is False
    assert reason == "DAILY_REQUEST_LIMIT"


def test_quota_blocks_daily_token_limit():
    """User exceeding daily token limit should be blocked."""
    store = UsageStore()
    quota = QuotaManager(store=store)

    store.increment_tokens("u1", 50_001)

    allowed, reason = quota.check("u1", tier="free")
    assert allowed is False
    assert reason == "DAILY_TOKEN_LIMIT"


def test_quota_blocks_concurrent_limit():
    """User exceeding concurrent task limit should be blocked."""
    store = UsageStore()
    quota = QuotaManager(store=store)

    store.increment_concurrent("u1")  # free tier max = 1

    allowed, reason = quota.check("u1", tier="free")
    assert allowed is False
    assert reason == "CONCURRENT_LIMIT_REACHED"


def test_quota_pro_tier_higher_limits():
    """Pro tier should have higher limits than free."""
    store = UsageStore()
    quota = QuotaManager(store=store)

    usage = store.get_or_create("u1")
    usage.daily_requests = 100  # Over free (50) but under pro (500)

    allowed_free, _ = quota.check("u1", tier="free")
    allowed_pro, _ = quota.check("u1", tier="pro")
    assert allowed_free is False
    assert allowed_pro is True


def test_quota_remaining():
    """get_remaining should return correct values."""
    store = UsageStore()
    quota = QuotaManager(store=store)

    store.increment_requests("u1")
    store.increment_tokens("u1", 10_000)

    remaining = quota.get_remaining("u1", tier="free")
    assert remaining["requests_remaining"] == 49  # 50 - 1
    assert remaining["tokens_remaining"] == 40_000  # 50000 - 10000


# ============================================
# TOKEN TRACKER TESTS
# ============================================

def test_token_tracker_record():
    """Token tracker should accumulate totals."""
    store = UsageStore()
    tracker = TokenTracker(store=store)

    tracker.record("u1", prompt_tokens=100, completion_tokens=200)
    tracker.record("u1", prompt_tokens=50, completion_tokens=150)

    totals = tracker.get_totals()
    assert totals["total_prompt_tokens"] == 150
    assert totals["total_completion_tokens"] == 350
    assert totals["total_tokens"] == 500
    assert totals["total_calls"] == 2


def test_token_tracker_feeds_usage_store():
    """Token tracker should update per-user store."""
    store = UsageStore()
    tracker = TokenTracker(store=store)

    tracker.record("u1", prompt_tokens=100, completion_tokens=200)
    usage = store.get_or_create("u1")
    assert usage.daily_tokens == 300


def test_token_tracker_no_double_count():
    """Recording should happen exactly once per call."""
    store = UsageStore()
    tracker = TokenTracker(store=store)

    tracker.record("u1", prompt_tokens=100, completion_tokens=100)
    assert store.get_or_create("u1").daily_tokens == 200

    # No second call — should not change
    assert store.get_or_create("u1").daily_tokens == 200


def test_token_tracker_reset():
    """Reset should clear tracker totals."""
    store = UsageStore()
    tracker = TokenTracker(store=store)
    tracker.record("u1", prompt_tokens=500, completion_tokens=500)
    tracker.reset()
    totals = tracker.get_totals()
    assert totals["total_tokens"] == 0
    assert totals["total_calls"] == 0


# ============================================
# SPEND GUARD TESTS
# ============================================

def test_spend_guard_allows_under_budget():
    """System under daily budget should be allowed."""
    store = UsageStore()
    guard = SpendGuard(max_daily_tokens=1_000_000, store=store)
    assert guard.allow() is True


def test_spend_guard_blocks_over_budget():
    """System exceeding daily budget should block ALL users."""
    store = UsageStore()
    guard = SpendGuard(max_daily_tokens=1000, store=store)

    store.increment_tokens("u1", 500)
    store.increment_tokens("u2", 501)

    assert guard.allow() is False


def test_spend_guard_status():
    """Status should show utilization correctly."""
    store = UsageStore()
    guard = SpendGuard(max_daily_tokens=10_000, store=store)
    store.increment_tokens("u1", 2500)

    status = guard.get_status()
    assert status["total_tokens_today"] == 2500
    assert status["remaining"] == 7500
    assert status["utilization_pct"] == 25.0
    assert status["budget_exhausted"] is False


# ============================================
# END-TO-END GOVERNANCE PIPELINE TEST
# ============================================

def test_full_governance_pipeline():
    """
    Simulate a full request lifecycle:
    1. Rate check passes
    2. Quota check passes
    3. Spend guard passes
    4. LLM call succeeds
    5. Tokens recorded
    6. Concurrent decremented
    7. Next request still works
    8. Eventually hits daily limit
    """
    store = UsageStore()
    limiter = RateLimiter(max_per_minute=100, store=store)
    quota = QuotaManager(store=store)
    tracker = TokenTracker(store=store)
    guard = SpendGuard(max_daily_tokens=5000, store=store)

    user = "test_user"

    # Task 1: Full lifecycle
    store.increment_concurrent(user)
    allowed, _ = limiter.check_and_record(user)
    assert allowed is True
    q_allowed, _ = quota.check(user, tier="pro")
    assert q_allowed is True
    assert guard.allow() is True

    # Simulate LLM call
    tracker.record(user, prompt_tokens=200, completion_tokens=300)
    store.decrement_concurrent(user)

    usage = store.get_or_create(user)
    assert usage.daily_tokens == 500
    assert usage.daily_requests == 1
    assert usage.concurrent_tasks == 0

    # Task 2-10: Burn through budget
    for i in range(9):
        store.increment_concurrent(user)
        limiter.check_and_record(user)
        tracker.record(user, prompt_tokens=200, completion_tokens=300)
        store.decrement_concurrent(user)

    # Total = 10 * 500 = 5000 tokens
    assert guard.allow() is False  # Budget exhausted!
    assert guard.get_status()["budget_exhausted"] is True
