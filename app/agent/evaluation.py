"""
Evaluation and Regression Gating for the research agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationMetrics:
    answer_accuracy: float
    claim_accuracy: float
    source_reliability_ratio: float
    avg_latency_ms: float
    cache_hit_ratio: float
    tool_failure_ratio: float


@dataclass
class RegressionThresholds:
    answer_accuracy_drop_max: float = 0.03
    claim_accuracy_drop_max: float = 0.03
    source_reliability_drop_max: float = 0.05
    latency_increase_max: float = 0.20
    cache_hit_drop_max: float = 0.10
    tool_failure_increase_max: float = 0.05


def _drop(old: float, new: float) -> float:
    return max(0.0, old - new)


def _increase(old: float, new: float) -> float:
    return max(0.0, new - old)


def check_regression_gate(
    baseline: EvaluationMetrics,
    candidate: EvaluationMetrics,
    thresholds: RegressionThresholds | None = None,
) -> Dict[str, object]:
    t = thresholds or RegressionThresholds()

    failures = []

    if _drop(baseline.answer_accuracy, candidate.answer_accuracy) > t.answer_accuracy_drop_max:
        failures.append('answer_accuracy')
    if _drop(baseline.claim_accuracy, candidate.claim_accuracy) > t.claim_accuracy_drop_max:
        failures.append('claim_accuracy')
    if _drop(baseline.source_reliability_ratio, candidate.source_reliability_ratio) > t.source_reliability_drop_max:
        failures.append('source_reliability_ratio')

    baseline_latency = max(1.0, baseline.avg_latency_ms)
    latency_growth = (candidate.avg_latency_ms - baseline_latency) / baseline_latency
    if latency_growth > t.latency_increase_max:
        failures.append('avg_latency_ms')

    if _drop(baseline.cache_hit_ratio, candidate.cache_hit_ratio) > t.cache_hit_drop_max:
        failures.append('cache_hit_ratio')
    if _increase(baseline.tool_failure_ratio, candidate.tool_failure_ratio) > t.tool_failure_increase_max:
        failures.append('tool_failure_ratio')

    return {
        'passed': len(failures) == 0,
        'failures': failures,
        'latency_growth': round(latency_growth, 4),
    }
