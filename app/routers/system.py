from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.models import HealthResponse
from app.observability.event_logger import get_event_logger
from app.observability.metrics_collector import get_metrics_collector
from app.observability.anomaly_detector import AnomalyDetector
from app.health.health_monitor import get_health_monitor
from app.health.circuit_breaker import get_circuit_breaker
from app.health.mitigation_policies import get_mitigation_policies
from app.health.worker_lifecycle import get_worker_lifecycle
from app.governance.rate_limiter import get_rate_limiter
from app.governance.quota_manager import get_quota_manager
from app.governance.spend_guard import get_spend_guard
from app.governance.token_tracker import get_token_tracker
from app.governance.usage_store import get_usage_store
from app.governance.ip_tracker import get_ip_tracker
from app.dashboard.dashboard_data import get_dashboard_aggregator, DEFAULT_MODEL


router = APIRouter(tags=["system"])


@router.get("/agent/metrics")
async def agent_metrics():
    metrics = get_metrics_collector()
    return {"metrics": metrics.get_metrics()}


@router.post("/agent/cancel")
async def agent_cancel(request: Request):
    """Cancel a running agent execution for a given chat session."""
    body = await request.json()
    user_id = body.get("user_id", "")
    chat_id = body.get("chat_id", "")
    if not user_id or not chat_id:
        return {"status": "error", "message": "user_id and chat_id required"}

    from app.realtime.websocket_runtime import _cancel_flags

    cancel_key = (user_id, chat_id)
    _cancel_flags[cancel_key] = True
    return {"status": "cancelled", "user_id": user_id, "chat_id": chat_id}


@router.get("/agent/events")
async def agent_events(limit: int = Query(default=100, ge=1, le=500)):
    logger = get_event_logger()
    return {"recent_events": logger.get_recent(limit=limit)}


@router.get("/agent/anomalies")
async def agent_anomalies():
    detector = AnomalyDetector(get_metrics_collector(), get_event_logger())
    alerts = detector.evaluate()
    return {"anomalies": alerts, "count": len(alerts)}


@router.get("/agent/health")
async def agent_health():
    monitor = get_health_monitor()
    breaker = get_circuit_breaker()
    lifecycle = get_worker_lifecycle()
    mitigation = get_mitigation_policies()

    signal = monitor.evaluate()
    breaker.evaluate(signal)

    return {
        "status": signal,
        "circuit_breaker": breaker.get_state(),
        "worker": lifecycle.get_state(),
        "active_mitigations": mitigation.get_active(),
        "metrics_snapshot": get_metrics_collector().get_metrics(),
    }


async def circuit_breaker_gate():
    monitor = get_health_monitor()
    breaker = get_circuit_breaker()
    mitigation = get_mitigation_policies()

    signal = monitor.evaluate()
    breaker.evaluate(signal)
    mitigation.apply(signal)

    if not breaker.allow_execution():
        logger = get_event_logger()
        from app.observability.event_types import EventType

        logger.emit(
            EventType.ANOMALY_ALERT,
            {
                "reason": "Circuit open - execution blocked",
                "signal": signal,
            },
        )
        return JSONResponse(
            content={"error": "SYSTEM_UNAVAILABLE", "signal": signal},
            status_code=503,
        )
    return None


@router.get("/agent/governance")
async def agent_governance(user_id: str = Query(default="anonymous")):
    quota = get_quota_manager()
    spend = get_spend_guard()
    usage = get_usage_store().get_or_create(user_id)
    return {
        "user_id": user_id,
        "usage": {
            "daily_requests": usage.daily_requests,
            "daily_tokens": usage.daily_tokens,
            "concurrent_tasks": usage.concurrent_tasks,
        },
        "quota_remaining": quota.get_remaining(user_id),
        "system_spend": spend.get_status(),
    }


@router.get("/agent/spend")
async def agent_spend():
    spend = get_spend_guard()
    tracker = get_token_tracker()
    return {
        "spend": spend.get_status(),
        "token_totals": tracker.get_totals(),
    }


async def governance_gate(user_id: str, tier: str = "free"):
    limiter = get_rate_limiter()
    allowed, reason = limiter.check_and_record(user_id)
    if not allowed:
        return JSONResponse(content={"error": reason, "retry_after_seconds": 60}, status_code=429)

    quota = get_quota_manager()
    allowed, reason = quota.check(user_id, tier=tier)
    if not allowed:
        return JSONResponse(
            content={"error": reason, "quota": quota.get_remaining(user_id, tier=tier)},
            status_code=429,
        )

    guard = get_spend_guard()
    if not guard.allow():
        return JSONResponse(
            content={"error": "SYSTEM_SPEND_LIMIT_REACHED", "spend": guard.get_status()},
            status_code=503,
        )

    return None


@router.get("/agent/abuse-status")
async def agent_abuse_status(ip: str = Query(default="unknown")):
    tracker = get_ip_tracker()
    return {
        "ip": ip,
        "flagged": tracker.is_flagged(ip),
        "request_count_1m": tracker.get_request_count(ip, window_seconds=60),
        "failure_count_5m": tracker.get_failure_count(ip, window_seconds=300),
        "associated_users": tracker.get_associated_users(ip),
    }


@router.get("/dashboard/overview")
async def dashboard_overview(model: str = Query(default=DEFAULT_MODEL)):
    aggregator = get_dashboard_aggregator()
    return aggregator.get_full_overview(model_name=model)


@router.get("/dashboard/tokens")
async def dashboard_tokens(model: str = Query(default=DEFAULT_MODEL)):
    aggregator = get_dashboard_aggregator()
    return aggregator.get_token_dashboard(model_name=model)


@router.get("/dashboard/errors")
async def dashboard_errors():
    aggregator = get_dashboard_aggregator()
    return aggregator.get_error_breakdown()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version="1.0.0", timestamp=datetime.now())


def parse_gate_json_response(response: JSONResponse) -> dict:
    return json.loads(response.body)

