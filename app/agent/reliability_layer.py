"""
Agent Reliability Layer
Deterministic quality checks that run after verification/ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReliabilityAssessment:
    tool_failures: int = 0
    loops_detected: int = 0
    source_count: int = 0
    verification_confidence: float = 0.0
    reliability_score: float = 0.0
    should_research_more: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_failures": int(self.tool_failures),
            "loops_detected": int(self.loops_detected),
            "source_count": int(self.source_count),
            "verification_confidence": float(self.verification_confidence),
            "reliability_score": float(self.reliability_score),
            "should_research_more": bool(self.should_research_more),
            "reasons": list(self.reasons),
        }


class AgentReliabilityLayer:
    """Deterministic guardrail layer for production reliability."""

    def __init__(
        self,
        min_sources: int = 2,
        min_confidence: float = 0.60,
        max_tool_failures: int = 2,
        max_repeat_ratio: float = 0.50,
    ):
        self.min_sources = int(min_sources)
        self.min_confidence = float(min_confidence)
        self.max_tool_failures = int(max_tool_failures)
        self.max_repeat_ratio = float(max_repeat_ratio)

    @staticmethod
    def _count_tool_failures(state: Any) -> int:
        metrics = state.workspace.intermediate_results.get("tool_metrics", {})
        total_failures = 0
        for item in metrics.values():
            try:
                total_failures += int(item.get("failures", 0))
            except Exception:
                continue
        return total_failures

    @staticmethod
    def _count_loop_signals(state: Any) -> int:
        signatures = list(state.workspace.intermediate_results.get("executed_tool_signatures", []))
        if len(signatures) < 4:
            return 0
        window = signatures[-8:]
        unique = len(set(window))
        if unique == 0:
            return 0
        repeat_ratio = 1.0 - (unique / max(1, len(window)))
        return 1 if repeat_ratio >= 0.50 else 0

    @staticmethod
    def _source_quality_score(state: Any) -> float:
        sources = state.workspace.sources
        if not sources:
            return 0.0
        trust_values: List[float] = []
        for src in sources:
            try:
                trust_values.append(float(src.get("trust_score", 0.5)))
            except Exception:
                trust_values.append(0.5)
        avg_trust = sum(trust_values) / max(1, len(trust_values))
        count_factor = min(1.0, len(sources) / 3.0)
        return max(0.0, min(1.0, avg_trust * count_factor))

    def assess(self, state: Any, verification_confidence: float) -> ReliabilityAssessment:
        tool_failures = self._count_tool_failures(state)
        loops_detected = self._count_loop_signals(state)
        source_count = len(state.workspace.sources)
        source_quality = self._source_quality_score(state)
        verification_confidence = float(max(0.0, min(1.0, verification_confidence)))

        # Weighted confidence across verification + source trust/coverage.
        reliability_score = (verification_confidence * 0.65) + (source_quality * 0.35)
        reasons: List[str] = []

        if source_count < self.min_sources:
            reasons.append("insufficient_sources")
        if verification_confidence < self.min_confidence:
            reasons.append("low_verification_confidence")
        if tool_failures >= self.max_tool_failures:
            reasons.append("tool_failures_high")
        if loops_detected > 0:
            reasons.append("loop_pattern_detected")

        should_research_more = bool(reasons)
        return ReliabilityAssessment(
            tool_failures=tool_failures,
            loops_detected=loops_detected,
            source_count=source_count,
            verification_confidence=verification_confidence,
            reliability_score=max(0.0, min(1.0, reliability_score)),
            should_research_more=should_research_more,
            reasons=reasons,
        )
