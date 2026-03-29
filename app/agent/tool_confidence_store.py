from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional


@dataclass
class ToolConfidenceRecord:
    tool_name: str
    weighted_success: float = 0.0
    weighted_failure: float = 0.0
    samples: int = 0
    relevance_sum: float = 0.0
    success_count: int = 0
    error_count: int = 0
    latency_ema_ms: float = 0.0
    latency_samples: int = 0
    last_updated_ts: float = 0.0

    def score(self) -> float:
        # Light Bayesian smoothing so a single failure does not collapse a tool.
        alpha = 2.0
        beta = 1.0
        return (alpha + float(self.weighted_success)) / (alpha + beta + float(self.weighted_success) + float(self.weighted_failure))

    def avg_relevance(self) -> float:
        if self.samples <= 0:
            return 0.0
        return float(self.relevance_sum) / float(self.samples)

    def success_rate(self) -> float:
        if self.samples <= 0:
            return 0.0
        return float(self.success_count) / float(max(1, self.samples))

    def error_rate(self) -> float:
        if self.samples <= 0:
            return 0.0
        return float(self.error_count) / float(max(1, self.samples))

    def latency_score(self) -> float:
        if self.latency_samples <= 0:
            return 0.5
        # 0ms -> ~1.0, 5000ms+ -> ~0.0
        return max(0.0, min(1.0, 1.0 - (float(self.latency_ema_ms) / 5000.0)))

    def selection_score(self) -> float:
        # Phase-2 scoring model:
        # score = success_rate*0.6 + (1-error_rate)*0.3 + latency_score*0.1
        sr = self.success_rate()
        er = self.error_rate()
        ls = self.latency_score()
        raw = (sr * 0.6) + ((1.0 - er) * 0.3) + (ls * 0.1)

        # Confidence-aware blending for low sample sizes.
        if self.samples < 3:
            return (raw * 0.4) + (0.55 * 0.6)
        if self.samples < 8:
            return (raw * 0.7) + (0.55 * 0.3)
        return raw


class ToolConfidenceStore:
    def __init__(self) -> None:
        self._records: Dict[str, ToolConfidenceRecord] = {}
        self._lock = Lock()

    def record(
        self,
        *,
        tool_name: str,
        success: bool,
        relevance: float = 0.0,
        latency_ms: Optional[float] = None,
        hard_error: bool = False,
    ) -> None:
        name = str(tool_name or "").strip().lower()
        if not name:
            return
        rel = max(0.0, min(1.0, float(relevance or 0.0)))
        # Blend execution success and semantic relevance quality.
        # Success carries most weight, relevance refines ranking among successful tools.
        blended_success = (0.75 if success else 0.0) + (0.25 * rel)
        blended_success = max(0.0, min(1.0, blended_success))
        blended_failure = 1.0 - blended_success
        now = time.time()
        with self._lock:
            rec = self._records.get(name)
            if rec is None:
                rec = ToolConfidenceRecord(tool_name=name)
                self._records[name] = rec
            rec.weighted_success += blended_success
            rec.weighted_failure += blended_failure
            rec.samples += 1
            rec.relevance_sum += rel
            if bool(success):
                rec.success_count += 1
            if bool(hard_error) or not bool(success):
                rec.error_count += 1
            if latency_ms is not None:
                lat = max(0.0, float(latency_ms))
                rec.latency_samples += 1
                if rec.latency_samples <= 1 or rec.latency_ema_ms <= 0.0:
                    rec.latency_ema_ms = lat
                else:
                    rec.latency_ema_ms = (0.2 * lat) + (0.8 * float(rec.latency_ema_ms))
            rec.last_updated_ts = now

    def get(self, tool_name: str) -> Optional[ToolConfidenceRecord]:
        name = str(tool_name or "").strip().lower()
        if not name:
            return None
        with self._lock:
            rec = self._records.get(name)
            if rec is None:
                return None
            # Return a copy snapshot to avoid accidental mutation.
            return ToolConfidenceRecord(
                tool_name=rec.tool_name,
                weighted_success=float(rec.weighted_success),
                weighted_failure=float(rec.weighted_failure),
                samples=int(rec.samples),
                relevance_sum=float(rec.relevance_sum),
                success_count=int(rec.success_count),
                error_count=int(rec.error_count),
                latency_ema_ms=float(rec.latency_ema_ms),
                latency_samples=int(rec.latency_samples),
                last_updated_ts=float(rec.last_updated_ts),
            )

    def snapshot(self) -> List[ToolConfidenceRecord]:
        with self._lock:
            return [
                ToolConfidenceRecord(
                    tool_name=rec.tool_name,
                    weighted_success=float(rec.weighted_success),
                    weighted_failure=float(rec.weighted_failure),
                    samples=int(rec.samples),
                    relevance_sum=float(rec.relevance_sum),
                    success_count=int(rec.success_count),
                    error_count=int(rec.error_count),
                    latency_ema_ms=float(rec.latency_ema_ms),
                    latency_samples=int(rec.latency_samples),
                    last_updated_ts=float(rec.last_updated_ts),
                )
                for rec in self._records.values()
            ]


_tool_confidence_singleton: Optional[ToolConfidenceStore] = None


def get_tool_confidence_store() -> ToolConfidenceStore:
    global _tool_confidence_singleton
    if _tool_confidence_singleton is None:
        _tool_confidence_singleton = ToolConfidenceStore()
    return _tool_confidence_singleton
