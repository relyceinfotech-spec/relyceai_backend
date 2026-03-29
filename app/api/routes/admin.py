from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import Response

from app.security.alert_notifier import send_admin_alert
from app.telemetry.high_stakes_store import get_high_stakes_store
from app.telemetry.metrics_collector import get_metrics_collector

_THRESHOLDS = {"source_fail_spike": 0.3, "low_confidence_spike": 0.4, "recency_fail_spike": 0.25}
_ALL_RANGE_CACHE: Dict[str, Any] = {"expires_at": 0.0, "threshold_sig": "", "metrics": None}
_ALL_RANGE_CACHE_TTL_SECONDS = 45.0


def get_high_stakes_thresholds() -> Dict[str, float]:
    return dict(_THRESHOLDS)


def save_high_stakes_thresholds(thresholds: Dict[str, Any], updated_by: str = "") -> Dict[str, float]:
    for key in ("source_fail_spike", "low_confidence_spike", "recency_fail_spike"):
        if key in thresholds:
            _THRESHOLDS[key] = float(thresholds[key])
    return dict(_THRESHOLDS)


def emit_security_audit(**kwargs):
    return kwargs


def _is_admin(user_info: Optional[Dict[str, Any]]) -> bool:
    claims = (user_info or {}).get("claims") if isinstance((user_info or {}).get("claims"), dict) else {}
    return bool(claims.get("admin") or claims.get("role") == "admin" or claims.get("superadmin"))


def _require_admin(user_info: Optional[Dict[str, Any]]) -> None:
    if not _is_admin(user_info):
        raise HTTPException(status_code=403, detail="admin required")


def _validate_range(range_key: str) -> str:
    allowed = {"24h", "7d", "30d", "all"}
    if range_key not in allowed:
        raise HTTPException(status_code=400, detail="invalid range")
    return range_key


def _threshold_signature(thresholds: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(thresholds, sort_keys=True).encode("utf-8")).hexdigest()


def _build_pagination(rows: List[Dict[str, Any]], page: int, page_size: int) -> Dict[str, Any]:
    total_items = len(rows)
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    safe_page = max(1, min(page, total_pages))
    start = (safe_page - 1) * page_size
    end = start + page_size
    return {
        "items": rows[start:end],
        "page": safe_page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
    }


def _audit_trail(payload: Dict[str, Any]) -> Dict[str, str]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "id": f"audit_{uuid.uuid4().hex[:12]}",
        "algo": "sha256",
        "signature": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


async def high_stakes_metrics(
    *,
    range: str = "24h",
    page: int = 1,
    page_size: int = 25,
    user_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _require_admin(user_info)
    range_key = _validate_range(range)
    thresholds = get_high_stakes_thresholds()

    cache_hit = False
    if range_key == "all":
        sig = _threshold_signature(thresholds)
        now = time.time()
        if (
            _ALL_RANGE_CACHE.get("metrics") is not None
            and float(_ALL_RANGE_CACHE.get("expires_at", 0.0)) > now
            and str(_ALL_RANGE_CACHE.get("threshold_sig") or "") == sig
        ):
            cache_hit = True
            metrics = dict(_ALL_RANGE_CACHE["metrics"])
        else:
            metrics = get_high_stakes_store().aggregate(range_key=range_key, thresholds=thresholds)
            _ALL_RANGE_CACHE["metrics"] = dict(metrics)
            _ALL_RANGE_CACHE["threshold_sig"] = sig
            _ALL_RANGE_CACHE["expires_at"] = now + _ALL_RANGE_CACHE_TTL_SECONDS
    else:
        metrics = get_high_stakes_store().aggregate(range_key=range_key, thresholds=thresholds)

    collector_metrics = get_metrics_collector().get_metrics().get("high_stakes_metrics", {})
    metrics["rolling_total"] = int(collector_metrics.get("total", 0))
    metrics["thresholds"] = dict(thresholds)

    trend_pag = _build_pagination(list(metrics.get("trend_series") or []), int(page), int(page_size))
    drill_pag = _build_pagination(list(metrics.get("domain_drilldown") or []), int(page), int(page_size))
    metrics["trend_series"] = trend_pag["items"]
    metrics["domain_drilldown"] = drill_pag["items"]
    metrics["pagination"] = {
        "trend_series": {k: trend_pag[k] for k in ("page", "page_size", "total_items", "total_pages")},
        "domain_drilldown": {k: drill_pag[k] for k in ("page", "page_size", "total_items", "total_pages")},
    }
    metrics["cache"] = {"hit": cache_hit}

    send_admin_alert(metrics)
    emit_security_audit(action="read_high_stakes_metrics", by=(user_info or {}).get("uid"), range=range_key)
    audit = _audit_trail({"range": range_key, "ts": datetime.now(timezone.utc).isoformat(), "total": metrics.get("total", 0)})
    return {"success": True, "metrics": metrics, "audit_trail": audit}


async def get_thresholds(*, user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    emit_security_audit(action="get_thresholds", by=(user_info or {}).get("uid"))
    return {"success": True, "thresholds": get_high_stakes_thresholds()}


async def update_thresholds(*, payload: Dict[str, Any], user_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _require_admin(user_info)
    updated = save_high_stakes_thresholds(dict(payload or {}).get("thresholds", {}), updated_by=str((user_info or {}).get("uid") or ""))
    emit_security_audit(action="update_thresholds", by=(user_info or {}).get("uid"), thresholds=updated)
    return {"success": True, "thresholds": updated}


def _filter_export(
    metrics: Dict[str, Any],
    *,
    domain: str = "",
    start_date: str = "",
    end_date: str = "",
) -> Dict[str, Any]:
    out = dict(metrics)
    rows = list(metrics.get("trend_series") or [])
    if domain:
        rows = [r for r in rows if str(r.get("domain") or "") == domain]
    if start_date:
        rows = [r for r in rows if str(r.get("bucket") or "") >= start_date]
    if end_date:
        rows = [r for r in rows if str(r.get("bucket") or "") <= end_date]
    out["trend_series"] = rows
    if domain:
        out["domain_drilldown"] = [r for r in (metrics.get("domain_drilldown") or []) if str(r.get("domain") or "") == domain]
    return out


async def export_high_stakes_metrics(
    *,
    range: str = "24h",
    format: str = "json",
    page: int = 1,
    page_size: int = 100,
    domain: str = "",
    start_date: str = "",
    end_date: str = "",
    user_info: Optional[Dict[str, Any]] = None,
):
    _require_admin(user_info)
    range_key = _validate_range(range)
    thresholds = get_high_stakes_thresholds()
    metrics = get_high_stakes_store().aggregate(range_key=range_key, thresholds=thresholds)
    metrics = _filter_export(metrics, domain=domain, start_date=start_date, end_date=end_date)

    trend_pag = _build_pagination(list(metrics.get("trend_series") or []), int(page), int(page_size))
    drill_pag = _build_pagination(list(metrics.get("domain_drilldown") or []), int(page), int(page_size))
    metrics["trend_series"] = trend_pag["items"]
    metrics["domain_drilldown"] = drill_pag["items"]
    audit = _audit_trail({"range": range_key, "domain": domain, "start_date": start_date, "end_date": end_date})

    emit_security_audit(action="export_high_stakes_metrics", by=(user_info or {}).get("uid"), format=format)
    if str(format or "").lower() == "json":
        return {"success": True, "export": metrics, "audit_trail": audit}

    if str(format or "").lower() != "csv":
        raise HTTPException(status_code=400, detail="unsupported format")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["row_type", "range", "bucket", "domain", "count", "percentage", "total", "low_confidence_rate", "source_fail_rate"])
    for row in metrics.get("domain_drilldown") or []:
        writer.writerow(
            [
                "domain_drilldown",
                range_key,
                "",
                row.get("domain", ""),
                row.get("count", 0),
                row.get("percentage", 0.0),
                "",
                "",
                "",
            ]
        )
    for row in metrics.get("trend_series") or []:
        writer.writerow(
            [
                "trend_series",
                range_key,
                row.get("bucket", ""),
                row.get("domain", ""),
                "",
                "",
                row.get("total", 0),
                row.get("low_confidence_rate", 0.0),
                row.get("source_fail_rate", 0.0),
            ]
        )
    return Response(content=buf.getvalue().encode("utf-8"), media_type="text/csv")
