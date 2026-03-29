from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.auth import get_firestore_db


def _parse_ts_epoch(evt: Dict[str, Any]) -> float:
    raw = evt.get("ts_epoch")
    if isinstance(raw, (int, float)):
        return float(raw)
    ts = str(evt.get("timestamp") or "").strip()
    if ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            pass
    return 0.0


def _day_bucket(ts_epoch: float) -> str:
    return datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).strftime("%Y-%m-%d")


def _in_window(ts_epoch: float, range_key: str, now: datetime) -> bool:
    if range_key == "all":
        return True
    windows = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}
    delta = windows.get(range_key, timedelta(hours=24))
    return ts_epoch >= (now - delta).timestamp()


@dataclass
class HighStakesStore:
    max_events: int = 5000

    def __post_init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def append(self, event: Dict[str, Any]) -> None:
        self._events.append(dict(event or {}))
        if len(self._events) > int(self.max_events):
            self._events = self._events[-int(self.max_events) :]

    def aggregate(self, range_key: str = "24h", thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        now = self._now()
        thresholds = dict(thresholds or {})
        rows = [e for e in self._events if _in_window(_parse_ts_epoch(e), range_key, now)]
        total = len(rows)

        domain_counts: Dict[str, int] = {}
        source_met = source_failed = recency_met = recency_failed = 0
        confidence_levels = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
        strict_total = 0

        trend_map: Dict[tuple[str, str], Dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "low": 0, "source_failed": 0}
        )

        for evt in rows:
            domain = str(evt.get("domain") or "general")
            domain_counts[domain] = int(domain_counts.get(domain, 0)) + 1
            if bool(evt.get("strict_mode", True)):
                strict_total += 1

            if bool(evt.get("source_requirement_met", False)):
                source_met += 1
            else:
                source_failed += 1

            if bool(evt.get("recency_requirement_met", False)):
                recency_met += 1
            else:
                recency_failed += 1

            conf = str(evt.get("confidence_level") or "LOW").upper()
            if conf not in confidence_levels:
                conf = "LOW"
            confidence_levels[conf] += 1

            ts_epoch = _parse_ts_epoch(evt)
            bucket = _day_bucket(ts_epoch if ts_epoch > 0 else now.timestamp())
            key = (bucket, domain)
            trend = trend_map[key]
            trend["total"] += 1
            if conf == "LOW":
                trend["low"] += 1
            if not bool(evt.get("source_requirement_met", False)):
                trend["source_failed"] += 1

        drilldown = []
        for domain, count in sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True):
            pct = round((count / max(1, total)) * 100, 2)
            drilldown.append({"domain": domain, "count": count, "percentage": pct})

        trend_series = []
        for (bucket, domain), agg in sorted(trend_map.items()):
            total_bucket = max(1, int(agg["total"]))
            trend_series.append(
                {
                    "bucket": bucket,
                    "domain": domain,
                    "total": int(agg["total"]),
                    "low_confidence_rate": round(float(agg["low"]) / total_bucket, 4),
                    "source_fail_rate": round(float(agg["source_failed"]) / total_bucket, 4),
                }
            )

        alerts = []
        if total > 0:
            low_conf_rate = confidence_levels["LOW"] / total
            source_fail_rate = source_failed / total
            recency_fail_rate = recency_failed / total
            if low_conf_rate > float(thresholds.get("low_confidence_spike", 0.4)):
                alerts.append({"code": "LOW_CONFIDENCE_SPIKE", "severity": "high", "message": "low confidence elevated"})
            if source_fail_rate > float(thresholds.get("source_fail_spike", 0.3)):
                alerts.append({"code": "SOURCE_FAIL_SPIKE", "severity": "high", "message": "source requirement failure elevated"})
            if recency_fail_rate > float(thresholds.get("recency_fail_spike", 0.25)):
                alerts.append({"code": "RECENCY_FAIL_SPIKE", "severity": "medium", "message": "recency requirement failure elevated"})

        return {
            "range": range_key,
            "total": total,
            "domain_counts": domain_counts,
            "strict_mode_total": strict_total,
            "source_requirement": {"met": source_met, "failed": source_failed},
            "recency_requirement": {"met": recency_met, "failed": recency_failed},
            "confidence_levels": confidence_levels,
            "domain_drilldown": drilldown,
            "trend_series": trend_series,
            "alerts": alerts,
            "thresholds": thresholds,
        }

    def compact_old_events(self, retention_days: int = 7) -> Dict[str, Any]:
        now = self._now()
        cutoff = (now - timedelta(days=int(retention_days))).timestamp()
        old_rows = [e for e in self._events if _parse_ts_epoch(e) < cutoff]
        self._events = [e for e in self._events if _parse_ts_epoch(e) >= cutoff]

        db = get_firestore_db()
        daily = db.collection("high_stakes_metrics_daily")
        raw_events = db.collection("high_stakes_metrics_events")

        rollup: Dict[tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "source_met": 0, "source_failed": 0, "recency_met": 0, "recency_failed": 0, "low": 0, "moderate": 0, "high": 0}
        )
        for evt in old_rows:
            ts_epoch = _parse_ts_epoch(evt)
            domain = str(evt.get("domain") or "general")
            day = _day_bucket(ts_epoch if ts_epoch > 0 else cutoff)
            key = (day, domain)
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

        for (day, domain), agg in rollup.items():
            doc_id = f"{day}_{domain}"
            daily.document(doc_id).set({"day": day, "domain": domain, **agg}, merge=True)

        deleted = 0
        try:
            for snap in raw_events.where("ts_epoch", "<", cutoff).stream():
                snap.reference.delete()
                deleted += 1
        except Exception:
            deleted = 0

        return {
            "success": True,
            "retention_days": int(retention_days),
            "local_compacted_events": len(old_rows),
            "daily_docs_updated": len(rollup),
            "raw_firestore_deleted": deleted,
        }


_store_singleton: Optional[HighStakesStore] = None


def get_high_stakes_store() -> HighStakesStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = HighStakesStore()
    return _store_singleton

