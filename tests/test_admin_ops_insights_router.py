import asyncio

import pytest
from fastapi import HTTPException

from app.routers import admin as admin_router


def test_require_superadmin_rejects_admin_user():
    with pytest.raises(HTTPException) as exc:
        admin_router.require_superadmin({"uid": "u1", "claims": {"role": "admin", "admin": True}})
    assert exc.value.status_code == 403


def test_build_ops_insights_shape():
    snapshot = {
        "totals": {"runs": 10, "success_rate": 0.8},
        "per_mode_stats": [{"mode": "smart", "avg_latency_ms": 1200}],
        "failure_analyzer": {
            "items": [
                {
                    "timestamp": "2026-03-29T10:00:00+00:00",
                    "query": "latest ai news",
                    "confidence_level": "LOW",
                    "retries": 2,
                    "mode": "smart",
                    "success": False,
                }
            ]
        },
        "confidence_distribution": {"low_pct": 0.4},
        "override_insights": {"recent": [{"at": "2026-03-29T10:00:01+00:00", "auto_selected": "smart", "overridden_to": "research_pro", "reason": "time_sensitive"}]},
        "role_flow": {"recent": [{"at": "2026-03-29T10:00:02+00:00", "node_id": "P1", "role": "researcher", "status": "completed"}]},
        "slo": {"current": {"p95_latency_ms": 15000}},
    }
    out = admin_router._build_ops_insights(snapshot)
    assert "usage" in out
    assert "chat_monitoring" in out
    assert "alerts" in out
    assert "logs" in out
    assert out["usage"]["total_chats"] == 10
    assert isinstance(out["chat_monitoring"]["low_confidence_outputs"], list)
    assert len(out["alerts"]) >= 1


def test_get_ops_insights_endpoint(monkeypatch):
    class _Svc:
        def get_debug_snapshot(self, range_window="24h", limit=25):
            return {
                "totals": {"runs": 3, "success_rate": 1.0},
                "per_mode_stats": [],
                "failure_analyzer": {"items": []},
                "confidence_distribution": {"low_pct": 0.0},
                "override_insights": {"recent": []},
                "role_flow": {"recent": []},
                "slo": {"current": {"p95_latency_ms": 0}},
            }

    monkeypatch.setattr(admin_router, "_get_research_service_or_503", lambda: _Svc())
    out = asyncio.run(
        admin_router.get_ops_insights(
            range="24h",
            limit=25,
            user_info={"uid": "u1", "claims": {"role": "admin", "admin": True}},
        )
    )
    assert out["success"] is True
    assert out["data"]["usage"]["total_chats"] == 3


def test_build_ops_insights_parallel_exception_alert():
    snapshot = {
        "totals": {"runs": 5, "success_rate": 0.95},
        "per_mode_stats": [],
        "per_role_stats": [{"role": "researcher", "parallel_exception_rate": 0.6}],
        "failure_analyzer": {"items": []},
        "confidence_distribution": {"low_pct": 0.0},
        "override_insights": {"recent": []},
        "role_flow": {"recent": []},
        "slo": {
            "current": {"p95_latency_ms": 1000},
            "targets": {"parallel_exception_rate_max": 0.30},
        },
    }
    out = admin_router._build_ops_insights(snapshot)
    codes = {str(a.get("code")) for a in out.get("alerts", [])}
    assert "parallel_exception_spike" in codes
