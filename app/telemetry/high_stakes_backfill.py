from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict

from app.auth import get_firestore_db


def _day_bucket(ts_epoch: float) -> str:
    return datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).strftime("%Y-%m-%d")


def backfill_high_stakes_daily(start_ts: float, end_ts: float, max_events: int = 1000) -> Dict[str, Any]:
    db = get_firestore_db()
    events = db.collection("high_stakes_metrics_events").stream()
    rows = []
    for snap in events:
        data = snap.to_dict() if hasattr(snap, "to_dict") else {}
        ts = float(data.get("ts_epoch", 0.0) or 0.0)
        if start_ts <= ts <= end_ts:
            rows.append(dict(data))
        if len(rows) >= int(max_events):
            break

    rollup = defaultdict(
        lambda: {"total": 0, "source_met": 0, "source_failed": 0, "recency_met": 0, "recency_failed": 0, "low": 0, "moderate": 0, "high": 0}
    )
    for evt in rows:
        ts = float(evt.get("ts_epoch", 0.0) or 0.0)
        domain = str(evt.get("domain") or "general")
        key = (_day_bucket(ts), domain)
        agg = rollup[key]
        agg["total"] += 1
        if bool(evt.get("source_requirement_met", False)):
            agg["source_met"] += 1
        else:
            agg["source_failed"] += 1
        if bool(evt.get("recency_requirement_met", False)):
            agg["recency_met"] += 1
        else:
            agg["recency_failed"] += 1
        conf = str(evt.get("confidence_level") or "LOW").upper()
        if conf == "HIGH":
            agg["high"] += 1
        elif conf == "MODERATE":
            agg["moderate"] += 1
        else:
            agg["low"] += 1

    daily = db.collection("high_stakes_metrics_daily")
    for (day, domain), agg in rollup.items():
        daily.document(f"{day}_{domain}").set({"day": day, "domain": domain, **agg}, merge=True)

    return {
        "success": True,
        "events_scanned": len(rows),
        "daily_docs_updated": len(rollup),
        "start_ts": float(start_ts),
        "end_ts": float(end_ts),
    }

