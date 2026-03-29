from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._context = {
            "events": 0,
            "tokens_before": 0,
            "tokens_after": 0,
            "tokens_saved": 0,
            "savings_pct": 0.0,
            "by_endpoint": {},
        }
        self._high_stakes = {"total": 0, "domains": {}}

    def record_context_optimization(self, endpoint: str, before_tokens: int, after_tokens: int) -> None:
        ep = str(endpoint or "unknown")
        before = int(max(0, before_tokens or 0))
        after = int(max(0, after_tokens or 0))
        saved = max(0, before - after)
        with self._lock:
            self._context["events"] += 1
            self._context["tokens_before"] += before
            self._context["tokens_after"] += after
            self._context["tokens_saved"] += saved
            total_before = max(1, int(self._context["tokens_before"]))
            self._context["savings_pct"] = round(self._context["tokens_saved"] / total_before, 4)
            by_ep = self._context["by_endpoint"].setdefault(ep, {"before": 0, "after": 0, "events": 0})
            by_ep["before"] += before
            by_ep["after"] += after
            by_ep["events"] += 1

    def record_high_stakes_event(self, domain: str) -> None:
        key = str(domain or "general").strip().lower()
        with self._lock:
            self._high_stakes["total"] += 1
            self._high_stakes["domains"][key] = int(self._high_stakes["domains"].get(key, 0)) + 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "context_efficiency": {
                    "events": int(self._context["events"]),
                    "tokens_before": int(self._context["tokens_before"]),
                    "tokens_after": int(self._context["tokens_after"]),
                    "tokens_saved": int(self._context["tokens_saved"]),
                    "savings_pct": float(self._context["savings_pct"]),
                    "by_endpoint": {k: dict(v) for k, v in self._context["by_endpoint"].items()},
                },
                "high_stakes_metrics": {
                    "total": int(self._high_stakes["total"]),
                    "domains": dict(self._high_stakes["domains"]),
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._context = {
                "events": 0,
                "tokens_before": 0,
                "tokens_after": 0,
                "tokens_saved": 0,
                "savings_pct": 0.0,
                "by_endpoint": {},
            }
            self._high_stakes = {"total": 0, "domains": {}}


_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector

