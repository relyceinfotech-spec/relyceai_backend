import asyncio
import pytest

from fastapi import HTTPException

from app.api.routes import admin as admin_route
from app.security.domain_policy import DOMAIN_FINANCE, enrich_payload_with_domain_policy
from app.telemetry.metrics_collector import get_metrics_collector

ADMIN_USER = {
    "uid": "u1",
    "role": "admin",
    "claims": {
        "role": "admin",
        "admin": True,
        "superadmin": False,
    },
}


class _FakeStore:
    def __init__(self, payload):
        self.payload = payload

    def aggregate(self, range_key: str = "24h", thresholds=None):
        out = dict(self.payload)
        out["range"] = range_key
        out["thresholds"] = thresholds or {}
        return out


class _FakeCollector:
    def __init__(self, metrics):
        self.metrics = metrics

    def get_metrics(self):
        return self.metrics


def test_admin_high_stakes_metrics_returns_aggregated_shape(monkeypatch):
    fake_store = _FakeStore(
        {
            "total": 12,
            "domain_counts": {"finance": 7, "news_current_events": 5},
            "strict_mode_total": 12,
            "source_requirement": {"met": 9, "failed": 3},
            "recency_requirement": {"met": 10, "failed": 2},
            "confidence_levels": {"LOW": 4, "MODERATE": 5, "HIGH": 3},
            "domain_drilldown": [{"domain": "finance", "count": 7, "percentage": 58.33}],
            "trend_series": [{"bucket": "2026-03-10", "domain": "finance", "low_confidence_rate": 0.2, "source_fail_rate": 0.1}],
            "alerts": [{"code": "LOW_CONFIDENCE_SPIKE", "severity": "high", "message": "low confidence elevated"}],
        }
    )
    fake_collector = _FakeCollector({"high_stakes_metrics": {"total": 20, "domains": {"finance": 11}}})

    monkeypatch.setattr(admin_route, "get_high_stakes_store", lambda: fake_store)
    monkeypatch.setattr(admin_route, "get_metrics_collector", lambda: fake_collector)
    monkeypatch.setattr(admin_route, "get_high_stakes_thresholds", lambda: {"source_fail_spike": 0.3, "low_confidence_spike": 0.4, "recency_fail_spike": 0.25})
    monkeypatch.setattr(admin_route, "send_admin_alert", lambda payload: None)
    monkeypatch.setattr(admin_route, "emit_security_audit", lambda **kwargs: kwargs)

    out = asyncio.run(admin_route.high_stakes_metrics(range="7d", user_info=ADMIN_USER))

    assert out["success"] is True
    assert out["metrics"]["range"] == "7d"
    assert out["metrics"]["total"] == 12
    assert out["metrics"]["domain_counts"]["finance"] == 7
    assert out["metrics"]["alerts"][0]["code"] == "LOW_CONFIDENCE_SPIKE"
    assert out["metrics"]["rolling_total"] == 20
    assert out["metrics"]["thresholds"]["low_confidence_spike"] == 0.4


def test_admin_high_stakes_metrics_invalid_range():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(admin_route.high_stakes_metrics(range="2h", user_info=ADMIN_USER))
    assert exc.value.status_code == 400


def test_admin_high_stakes_metrics_forbidden_for_non_admin_claim():
    user = {"uid": "u2", "role": "admin", "claims": {"role": "user", "admin": False, "superadmin": False}}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(admin_route.high_stakes_metrics(range="24h", user_info=user))
    assert exc.value.status_code == 403


def test_threshold_routes(monkeypatch):
    monkeypatch.setattr(admin_route, "get_high_stakes_thresholds", lambda: {"source_fail_spike": 0.31, "low_confidence_spike": 0.45, "recency_fail_spike": 0.2})
    monkeypatch.setattr(admin_route, "save_high_stakes_thresholds", lambda thresholds, updated_by="": {"source_fail_spike": float(thresholds.get("source_fail_spike", 0.3)), "low_confidence_spike": 0.5, "recency_fail_spike": 0.22})
    monkeypatch.setattr(admin_route, "emit_security_audit", lambda **kwargs: kwargs)

    get_out = asyncio.run(admin_route.get_thresholds(user_info=ADMIN_USER))
    assert get_out["success"] is True
    assert get_out["thresholds"]["source_fail_spike"] == 0.31

    put_out = asyncio.run(admin_route.update_thresholds(payload={"thresholds": {"source_fail_spike": 0.4}}, user_info=ADMIN_USER))
    assert put_out["success"] is True
    assert put_out["thresholds"]["source_fail_spike"] == 0.4


def test_telemetry_incremented_when_domain_policy_applies():
    collector = get_metrics_collector()
    collector.reset()

    payload = {
        "answer": "Balanced financial information.",
        "response": "Balanced financial information.",
        "confidence": 0.82,
        "sources": [
            {"url": "https://sec.gov/report"},
            {"url": "https://reuters.com/markets"},
        ],
        "metadata": {"domain": DOMAIN_FINANCE},
    }

    _ = enrich_payload_with_domain_policy(payload, "should i invest in this?")

    metrics = collector.get_metrics()
    hs = metrics.get("high_stakes_metrics", {})
    assert hs.get("total", 0) >= 1
    assert hs.get("domains", {}).get("finance", 0) >= 1


def test_admin_high_stakes_metrics_all_range_paginates_and_uses_cache(monkeypatch):
    class _CountingStore:
        def __init__(self):
            self.calls = 0

        def aggregate(self, range_key: str = "24h", thresholds=None):
            self.calls += 1
            return {
                "range": range_key,
                "total": 240,
                "domain_counts": {"finance": 120, "legal": 120},
                "strict_mode_total": 240,
                "source_requirement": {"met": 200, "failed": 40},
                "recency_requirement": {"met": 210, "failed": 30},
                "confidence_levels": {"LOW": 80, "MODERATE": 120, "HIGH": 40},
                "domain_drilldown": [
                    {"domain": f"d{i}", "count": 1, "percentage": 0.4}
                    for i in range(80)
                ],
                "trend_series": [
                    {"bucket": f"2026-03-{(i % 30) + 1:02d}", "domain": "finance", "low_confidence_rate": 0.2, "source_fail_rate": 0.1}
                    for i in range(120)
                ],
                "alerts": [],
                "thresholds": thresholds or {},
            }

    fake_store = _CountingStore()
    fake_collector = _FakeCollector({"high_stakes_metrics": {"total": 5, "domains": {"finance": 3}}})

    monkeypatch.setattr(admin_route, "_ALL_RANGE_CACHE", {"expires_at": 0.0, "threshold_sig": "", "metrics": None})
    monkeypatch.setattr(admin_route, "get_high_stakes_store", lambda: fake_store)
    monkeypatch.setattr(admin_route, "get_metrics_collector", lambda: fake_collector)
    monkeypatch.setattr(admin_route, "get_high_stakes_thresholds", lambda: {"source_fail_spike": 0.3, "low_confidence_spike": 0.4, "recency_fail_spike": 0.25})
    monkeypatch.setattr(admin_route, "send_admin_alert", lambda payload: None)
    monkeypatch.setattr(admin_route, "emit_security_audit", lambda **kwargs: kwargs)

    out1 = asyncio.run(admin_route.high_stakes_metrics(range="all", page=2, page_size=25, user_info=ADMIN_USER))
    out2 = asyncio.run(admin_route.high_stakes_metrics(range="all", page=3, page_size=25, user_info=ADMIN_USER))

    m1 = out1["metrics"]
    m2 = out2["metrics"]

    assert fake_store.calls == 1
    assert len(m1["trend_series"]) == 25
    assert len(m1["domain_drilldown"]) == 25
    assert m1["pagination"]["trend_series"]["total_items"] == 120
    assert m1["pagination"]["trend_series"]["total_pages"] == 5
    assert m1["cache"]["hit"] is False
    assert m2["cache"]["hit"] is True


def test_admin_export_json_and_csv(monkeypatch):
    fake_store = _FakeStore(
        {
            "total": 8,
            "domain_counts": {"finance": 8},
            "strict_mode_total": 8,
            "source_requirement": {"met": 6, "failed": 2},
            "recency_requirement": {"met": 7, "failed": 1},
            "confidence_levels": {"LOW": 2, "MODERATE": 4, "HIGH": 2},
            "domain_drilldown": [{"domain": "finance", "count": 8, "percentage": 100.0}],
            "trend_series": [{"bucket": "2026-03-10", "domain": "finance", "total": 8, "low_confidence_rate": 0.25, "source_fail_rate": 0.25}],
            "alerts": [],
        }
    )
    monkeypatch.setattr(admin_route, "get_high_stakes_store", lambda: fake_store)
    monkeypatch.setattr(admin_route, "get_high_stakes_thresholds", lambda: {"source_fail_spike": 0.3, "low_confidence_spike": 0.4, "recency_fail_spike": 0.25})
    monkeypatch.setattr(admin_route, "emit_security_audit", lambda **kwargs: kwargs)

    json_out = asyncio.run(admin_route.export_high_stakes_metrics(range="30d", format="json", page=1, page_size=100, user_info=ADMIN_USER))
    assert json_out["success"] is True
    assert json_out["export"]["domain_drilldown"][0]["domain"] == "finance"

    csv_out = asyncio.run(admin_route.export_high_stakes_metrics(range="30d", format="csv", page=1, page_size=100, user_info=ADMIN_USER))
    assert csv_out.media_type == "text/csv"
    assert "row_type,range,bucket,domain,count,percentage,total,low_confidence_rate,source_fail_rate" in csv_out.body.decode("utf-8")


def test_admin_metrics_read_returns_signed_audit_trail(monkeypatch):
    fake_store = _FakeStore(
        {
            "total": 1,
            "domain_counts": {"finance": 1},
            "strict_mode_total": 1,
            "source_requirement": {"met": 1, "failed": 0},
            "recency_requirement": {"met": 1, "failed": 0},
            "confidence_levels": {"LOW": 0, "MODERATE": 1, "HIGH": 0},
            "domain_drilldown": [{"domain": "finance", "count": 1, "percentage": 100.0}],
            "trend_series": [{"bucket": "2026-03-10", "domain": "finance", "total": 1, "low_confidence_rate": 0.0, "source_fail_rate": 0.0}],
            "alerts": [],
        }
    )
    fake_collector = _FakeCollector({"high_stakes_metrics": {"total": 1, "domains": {"finance": 1}}})

    monkeypatch.setattr(admin_route, "get_high_stakes_store", lambda: fake_store)
    monkeypatch.setattr(admin_route, "get_metrics_collector", lambda: fake_collector)
    monkeypatch.setattr(admin_route, "get_high_stakes_thresholds", lambda: {"source_fail_spike": 0.3, "low_confidence_spike": 0.4, "recency_fail_spike": 0.25})
    monkeypatch.setattr(admin_route, "emit_security_audit", lambda **kwargs: kwargs)

    out = asyncio.run(admin_route.high_stakes_metrics(range="24h", user_info=ADMIN_USER))
    assert out["audit_trail"]["id"]
    assert out["audit_trail"]["signature"]
    assert out["audit_trail"]["algo"] in {"sha256", "hmac-sha256"}


def test_admin_export_applies_domain_and_date_filters(monkeypatch):
    fake_store = _FakeStore(
        {
            "total": 10,
            "domain_counts": {"finance": 5, "legal": 5},
            "strict_mode_total": 10,
            "source_requirement": {"met": 8, "failed": 2},
            "recency_requirement": {"met": 9, "failed": 1},
            "confidence_levels": {"LOW": 3, "MODERATE": 5, "HIGH": 2},
            "domain_drilldown": [
                {"domain": "finance", "count": 5, "percentage": 50.0},
                {"domain": "legal", "count": 5, "percentage": 50.0},
            ],
            "trend_series": [
                {"bucket": "2026-03-09", "domain": "finance", "total": 3, "low_confidence_rate": 0.3, "source_fail_rate": 0.1},
                {"bucket": "2026-03-10", "domain": "finance", "total": 2, "low_confidence_rate": 0.2, "source_fail_rate": 0.1},
                {"bucket": "2026-03-10", "domain": "legal", "total": 5, "low_confidence_rate": 0.5, "source_fail_rate": 0.4},
            ],
            "alerts": [],
        }
    )
    monkeypatch.setattr(admin_route, "get_high_stakes_store", lambda: fake_store)
    monkeypatch.setattr(admin_route, "get_high_stakes_thresholds", lambda: {"source_fail_spike": 0.3, "low_confidence_spike": 0.4, "recency_fail_spike": 0.25})
    monkeypatch.setattr(admin_route, "emit_security_audit", lambda **kwargs: kwargs)

    out = asyncio.run(
        admin_route.export_high_stakes_metrics(
            range="30d",
            format="json",
            page=1,
            page_size=100,
            domain="finance",
            start_date="2026-03-10",
            end_date="2026-03-10",
            user_info=ADMIN_USER,
        )
    )

    assert out["success"] is True
    assert out["audit_trail"]["id"]
    rows = out["export"]["trend_series"]
    assert len(rows) == 1
    assert rows[0]["domain"] == "finance"
    assert rows[0]["bucket"] == "2026-03-10"
