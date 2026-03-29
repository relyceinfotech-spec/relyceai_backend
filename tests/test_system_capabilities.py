import asyncio

from app.api.routes import system as system_routes


class _Queue:
    async def summary(self):
        return {"paused": False, "queue_depth": 0}


class _SpendGuard:
    def get_status(self):
        return {
            "total_tokens_today": 100,
            "max_daily_tokens": 100000,
            "remaining": 99900,
            "utilization_pct": 0.1,
            "budget_exhausted": False,
        }


def test_capabilities_endpoint_returns_inventory(monkeypatch):
    monkeypatch.setattr(system_routes, "get_task_queue", lambda: _Queue())
    monkeypatch.setattr(system_routes, "get_spend_guard", lambda: _SpendGuard())

    out = asyncio.run(system_routes.capabilities())

    assert out["status"] == "ok"
    assert "tools" in out
    assert out["tools"]["count"] >= 1
    assert "deepsearch" in out["modes"]
    assert out["memory"]["query_enabled"] is True


def test_health_endpoint_ok():
    out = asyncio.run(system_routes.health())
    assert out == {"status": "ok"}
