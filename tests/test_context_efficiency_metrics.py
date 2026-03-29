from app.telemetry.metrics_collector import MetricsCollector
from app.dashboard.dashboard_data import DashboardAggregator


def test_context_efficiency_metrics_aggregate():
    m = MetricsCollector()
    m.record_context_optimization(endpoint="/ws/chat", before_tokens=6000, after_tokens=2200)
    m.record_context_optimization(endpoint="/chat/stream", before_tokens=5000, after_tokens=2600)
    snap = m.get_metrics()

    ctx = snap.get("context_efficiency") or {}
    assert ctx.get("events") == 2
    assert ctx.get("tokens_before") == 11000
    assert ctx.get("tokens_after") == 4800
    assert ctx.get("tokens_saved") == 6200
    assert "/ws/chat" in (ctx.get("by_endpoint") or {})


def test_dashboard_context_efficiency_shape(monkeypatch):
    fake = {
        "context_efficiency": {
            "events": 1,
            "tokens_before": 6000,
            "tokens_after": 2000,
            "tokens_saved": 4000,
            "savings_pct": 0.6666,
            "by_endpoint": {
                "/ws/chat": {"before": 6000, "after": 2000, "events": 1},
            },
        }
    }

    class _FakeMetrics:
        def get_metrics(self):
            return fake

    monkeypatch.setattr("app.dashboard.dashboard_data.get_metrics_collector", lambda: _FakeMetrics())
    out = DashboardAggregator.get_context_efficiency_dashboard("gpt-4o")
    assert out["tokens_saved"] == 4000
    assert len(out["by_endpoint"]) == 1
    assert out["by_endpoint"][0]["endpoint"] == "/ws/chat"
