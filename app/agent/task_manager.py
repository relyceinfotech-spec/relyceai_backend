"""
Task Manager
Orchestrates the persistent autonomous loop (Devin-style).

Updated pipeline:
Query -> RetrievalLayer -> Research DAG -> Fact Extraction -> Claim Verification
-> Evidence Ranking -> Synthesis -> Knowledge Upsert -> Run History
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, AsyncGenerator, Tuple, Set
from urllib.parse import urlparse

from app.agent.agent_state import AgentState
from app.agent.graph_builder import compile_plan_graph, classify_query
from app.agent.graph_scheduler import run_plan_graph
from app.agent.agent_orchestrator import run_agent_pipeline
from app.agent.plan_validator import validate_plan
from app.agent.goal_checker import GoalChecker
from app.agent.trace_logger import log_trace
from app.agent.synthesizer import ResponseSynthesizer
from app.agent.reliability_layer import AgentReliabilityLayer
from app.agent.trust_scorer import score_source
from app.agent.tool_executor import ToolCall, ExecutionContext, execute_tool, TOOLS
from app.retrieval.retrieval_layer import RetrievalLayer
from app.reasoning.fact_extractor import extract_facts_from_workspace, extract_claims
from app.reasoning.verification_engine import verify_claims
from app.reasoning.evidence_ranker import rank_evidence, confidence_progress
from app.learning.run_history import log_run
from app.learning.adaptive_store import AdaptiveLearningStore, adaptive_bucket_key, normalize_for_fingerprint
from app.streaming.event_streamer import EventStreamer
from app.observability.metrics_collector import get_metrics_collector
from app.agent.reliability_runtime import (
    classify_reliability_query_type,
    detect_unsupported_claims,
    evaluate_evidence_quality,
    is_current_info_query,
    needs_tool_first,
    query_classifier_confidence,
    should_enforce_evidence,
)
from app.agent.tool_memory_store import get_tool_memory_store
from app.agent.tool_confidence_store import get_tool_confidence_store
from app.agent.role_cognition import ROLE_POLICIES, resolve_plan_node_role
from app.chat.mode_mapper import normalize_chat_mode
from app.platform.mode_routing import select_queue_lane
from app.config import (
    FAST_MODEL,
    FF_CONFIDENCE_CRITIC_LOOPS,
    FF_TOOL_FIRST_EVIDENCE,
    FF_TOOL_MEMORY,
    CRITIC_CONFIDENCE_MIN,
    CRITIC_MAX_REPAIRS,
    FAST_LANE_RELIABILITY_BUDGET_MS,
    HEAVY_LANE_RELIABILITY_BUDGET_MS,
    CURRENT_INFO_RECENCY_DAYS,
    RELIABILITY_TOOL_RETRY_BACKOFF_MS,
    TOOL_SCHEMA_VERSION,
    FF_LLM_INTENT_CLASSIFIER,
    FF_TOOL_CONFIDENCE_SCORING,
    TOOL_CONFIDENCE_MIN_SAMPLES,
)

MIN_FINDINGS = 2
MIN_SOURCES = 2
TRUSTED_SOURCE_THRESHOLD = 0.80
CITATION_COVERAGE_THRESHOLD = 0.70
SOURCE_MIX_SECONDARY_MIN = 0.45


@dataclass(frozen=True)
class ModePolicy:
    mode_name: str
    memory_read_enabled: bool
    memory_write_enabled: bool
    min_reliability_budget_ms: int
    respect_small_talk_for_forced_tools: bool
    live_query_forces_tools: bool
    reasoning_hints_enabled: bool
    evidence_softpass_with_existing_sources: bool
    trusted_source_softpass_with_existing_sources: bool
    critic_repair_requires_missing_sources: bool
    tool_force_for_non_smalltalk: bool
    strict_evidence: bool
    require_multi_source: bool
    require_trusted_sources_only: bool
    max_reliability_retries: int


_BASE_MODE_POLICY: Dict[str, Any] = {
    "memory_read_enabled": True,
    "memory_write_enabled": True,
    "min_reliability_budget_ms": 15000,
    "respect_small_talk_for_forced_tools": False,
    "live_query_forces_tools": True,
    "reasoning_hints_enabled": False,
    "evidence_softpass_with_existing_sources": False,
    "trusted_source_softpass_with_existing_sources": False,
    "critic_repair_requires_missing_sources": False,
    "tool_force_for_non_smalltalk": False,
    "strict_evidence": False,
    "require_multi_source": False,
    "require_trusted_sources_only": False,
    "max_reliability_retries": 2,
}

_MODE_POLICY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Conversational flavor: reason-first, fresh runs, relaxed when at least one live source exists.
    "smart": {
        "memory_read_enabled": False,
        "memory_write_enabled": False,
        "min_reliability_budget_ms": 35000,
        "respect_small_talk_for_forced_tools": True,
        "reasoning_hints_enabled": True,
        "evidence_softpass_with_existing_sources": True,
        "trusted_source_softpass_with_existing_sources": True,
        "critic_repair_requires_missing_sources": True,
    },
    # Structured flavor: deterministic and execution-oriented.
    "agent": {},
    # Deep-trust research flavor: strict evidence, higher budget, tool-first.
    "research_pro": {
        "memory_read_enabled": True,
        "memory_write_enabled": True,
        "min_reliability_budget_ms": 45000,
        "respect_small_talk_for_forced_tools": False,
        "live_query_forces_tools": True,
        "reasoning_hints_enabled": False,
        "evidence_softpass_with_existing_sources": False,
        "trusted_source_softpass_with_existing_sources": False,
        "critic_repair_requires_missing_sources": False,
        "tool_force_for_non_smalltalk": True,
        "strict_evidence": True,
        "require_multi_source": True,
        "require_trusted_sources_only": True,
        "max_reliability_retries": 3,
    },
}

_MODE_POLICY_FIELD_NAMES: Set[str] = {
    key for key in ModePolicy.__dataclass_fields__.keys() if key != "mode_name"
}


class TaskManager:
    def __init__(self, client, model_to_use: str = FAST_MODEL):
        self.client = client
        self.model_to_use = model_to_use
        self.goal_checker = GoalChecker(client, model_to_use)
        self.synthesizer = ResponseSynthesizer(client, model_to_use)
        self.response_cache: Dict[str, str] = {}
        self.retrieval = RetrievalLayer(similarity_threshold=0.90, min_confidence=0.80, ttl_seconds=86400)
        self.event_streamer = EventStreamer()
        self.reliability_layer = AgentReliabilityLayer(min_sources=MIN_SOURCES, min_confidence=0.60)
        self.max_task_latency_ms = 45000
        self.reasoning_hints_cache: Dict[str, List[str]] = {}
        self.intent_context_cache: Dict[str, Dict[str, Any]] = {}
        self.tool_confidence_store = get_tool_confidence_store()
        self.mode_response_profiles: Dict[str, Dict[str, float]] = {}
        self.role_response_profiles: Dict[str, Dict[str, float]] = {}
        self.run_history_records: deque[Dict[str, Any]] = deque(maxlen=1200)
        self.low_confidence_reviews: deque[Dict[str, Any]] = deque(maxlen=250)
        self.auto_tuning_events: deque[Dict[str, Any]] = deque(maxlen=200)
        self.auto_tuning_state: Dict[str, Dict[str, Any]] = {}
        self.auto_remediation_events: deque[Dict[str, Any]] = deque(maxlen=200)
        self.auto_remediation_state: Dict[str, Any] = {}
        self.adaptive_store = AdaptiveLearningStore()
        self.adaptive_learning_state: Dict[str, Dict[str, Any]] = {}
        self.adaptive_learning_events: deque[Dict[str, Any]] = deque(maxlen=300)
        self.adaptive_learning_cases: deque[Dict[str, Any]] = deque(maxlen=500)
        self.adaptive_state_loaded_keys: Set[str] = set()
        self.adaptive_state_cache_meta: Dict[str, Dict[str, Any]] = {}
        self.adaptive_delta_snapshots: deque[Dict[str, Any]] = deque(maxlen=10)
        self.adaptive_snapshot_seq = 0
        self.debug_config: Dict[str, Any] = {
            "auto_tuning_enabled": True,
            "auto_tuning_min_runs": 12,
            "auto_tuning_success_threshold": 0.72,
            "auto_tuning_cooldown_runs": 6,
            "auto_tuning_budget_step_ms": 5000,
            "auto_tuning_retry_step": 1,
            "auto_tuning_max_budget_delta_ms": 20000,
            "auto_tuning_max_retry_delta": 2,
            "failure_confidence_threshold": 0.45,
            "adaptive_enabled": True,
            "adaptive_ema_alpha": 0.2,
            "adaptive_min_cluster_runs": 8,
            "adaptive_low_conf_threshold": 0.35,
            "adaptive_retry_threshold": 0.30,
            "adaptive_cooldown_runs": 6,
            "adaptive_max_bias_delta": 0.1,
            "adaptive_state_cache_ttl_sec": 90,
            "adaptive_state_write_every_runs": 3,
            "adaptive_apply_ratio": 1.0,
            "slo_p95_latency_ms_max": 25000.0,
            "slo_low_conf_rate_max": 0.35,
            "slo_avg_retries_max": 1.5,
            "slo_parallel_exception_rate_max": 0.30,
            "auto_remediation_enabled": True,
            "auto_remediation_min_runs": 20,
            "auto_remediation_cooldown_runs": 10,
            "auto_remediation_step": 0.1,
            "auto_remediation_min_apply_ratio": 0.1,
            "auto_remediation_low_conf_trigger": 0.40,
            "auto_remediation_retry_step_cap": 2,
            "auto_remediation_window_runs": 200,
        }

    def _record_response_profile(
        self,
        *,
        mode: str,
        latency_ms: int,
        retries: int,
        tool_calls: int,
        success: bool,
    ) -> Dict[str, Any]:
        mode_key = normalize_chat_mode(str(mode or "smart"))
        stats = self.mode_response_profiles.setdefault(
            mode_key,
            {
                "runs": 0.0,
                "successes": 0.0,
                "latency_ms_total": 0.0,
                "retries_total": 0.0,
                "tool_calls_total": 0.0,
            },
        )
        stats["runs"] += 1.0
        if success:
            stats["successes"] += 1.0
        stats["latency_ms_total"] += max(0.0, float(latency_ms or 0))
        stats["retries_total"] += max(0.0, float(retries or 0))
        stats["tool_calls_total"] += max(0.0, float(tool_calls or 0))

        runs = max(1.0, float(stats["runs"]))
        success_rate = float(stats["successes"]) / runs
        return {
            "mode": mode_key,
            "latency_ms": int(max(0, int(latency_ms or 0))),
            "retries": int(max(0, int(retries or 0))),
            "tool_calls": int(max(0, int(tool_calls or 0))),
            "success_rate": round(float(success_rate), 4),
            "runs": int(runs),
        }

    def _record_role_profile(
        self,
        *,
        role: str,
        latency_ms: int,
        retries: int,
        tool_calls: int,
        success: bool,
    ) -> Dict[str, Any]:
        role_key = str(role or "executor").strip().lower() or "executor"
        stats = self.role_response_profiles.setdefault(
            role_key,
            {
                "runs": 0.0,
                "successes": 0.0,
                "latency_ms_total": 0.0,
                "retries_total": 0.0,
                "tool_calls_total": 0.0,
            },
        )
        stats["runs"] += 1.0
        if success:
            stats["successes"] += 1.0
        stats["latency_ms_total"] += max(0.0, float(latency_ms or 0))
        stats["retries_total"] += max(0.0, float(retries or 0))
        stats["tool_calls_total"] += max(0.0, float(tool_calls or 0))

        runs = max(1.0, float(stats["runs"]))
        success_rate = float(stats["successes"]) / runs
        return {
            "role": role_key,
            "latency_ms": int(max(0, int(latency_ms or 0))),
            "retries": int(max(0, int(retries or 0))),
            "tool_calls": int(max(0, int(tool_calls or 0))),
            "success_rate": round(float(success_rate), 4),
            "runs": int(runs),
        }

    @staticmethod
    def _confidence_level(confidence: float) -> str:
        c = max(0.0, min(1.0, float(confidence or 0.0)))
        if c >= 0.8:
            return "HIGH"
        if c >= 0.55:
            return "MODERATE"
        return "LOW"

    @staticmethod
    def _range_seconds(range_window: str) -> Optional[int]:
        key = str(range_window or "24h").strip().lower()
        if key == "24h":
            return 24 * 60 * 60
        if key == "7d":
            return 7 * 24 * 60 * 60
        if key == "30d":
            return 30 * 24 * 60 * 60
        return None

    def _maybe_apply_auto_tuning(self, *, mode: str, response_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cfg = dict(self.debug_config or {})
        if not bool(cfg.get("auto_tuning_enabled", True)):
            return None

        mode_key = normalize_chat_mode(str(mode or "smart"))
        runs = int(response_profile.get("runs", 0) or 0)
        success_rate = float(response_profile.get("success_rate", 0.0) or 0.0)
        min_runs = max(3, int(cfg.get("auto_tuning_min_runs", 12) or 12))
        threshold = max(0.2, min(0.99, float(cfg.get("auto_tuning_success_threshold", 0.72) or 0.72)))
        cooldown_runs = max(1, int(cfg.get("auto_tuning_cooldown_runs", 6) or 6))
        if runs < min_runs or success_rate >= threshold:
            return None

        state = dict(self.auto_tuning_state.get(mode_key) or {})
        last_applied_runs = int(state.get("last_applied_runs", 0) or 0)
        if runs - last_applied_runs < cooldown_runs:
            return None

        step_budget_ms = max(1000, int(cfg.get("auto_tuning_budget_step_ms", 5000) or 5000))
        step_retry = max(0, int(cfg.get("auto_tuning_retry_step", 1) or 1))
        max_budget_delta_ms = max(step_budget_ms, int(cfg.get("auto_tuning_max_budget_delta_ms", 20000) or 20000))
        max_retry_delta = max(step_retry, int(cfg.get("auto_tuning_max_retry_delta", 2) or 2))

        prev_budget_delta = int(state.get("budget_delta_ms", 0) or 0)
        prev_retry_delta = int(state.get("retry_delta", 0) or 0)
        new_budget_delta = min(max_budget_delta_ms, prev_budget_delta + step_budget_ms)
        new_retry_delta = min(max_retry_delta, prev_retry_delta + step_retry)
        if new_budget_delta == prev_budget_delta and new_retry_delta == prev_retry_delta:
            return None

        applied_count = int(state.get("applied_count", 0) or 0) + 1
        next_state = {
            "budget_delta_ms": int(new_budget_delta),
            "retry_delta": int(new_retry_delta),
            "applied_count": int(applied_count),
            "last_applied_runs": int(runs),
            "last_success_rate": round(float(success_rate), 4),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.auto_tuning_state[mode_key] = next_state

        event = {
            "mode": mode_key,
            "success_rate": round(float(success_rate), 4),
            "runs": int(runs),
            "budget_delta_ms_before": int(prev_budget_delta),
            "budget_delta_ms_after": int(new_budget_delta),
            "retry_delta_before": int(prev_retry_delta),
            "retry_delta_after": int(new_retry_delta),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.auto_tuning_events.append(event)
        return event

    def _apply_auto_tuning_overrides(self, policy: ModePolicy) -> ModePolicy:
        state = dict(self.auto_tuning_state.get(str(policy.mode_name or "")) or {})
        budget_delta_ms = max(0, int(state.get("budget_delta_ms", 0) or 0))
        retry_delta = max(0, int(state.get("retry_delta", 0) or 0))
        if budget_delta_ms <= 0 and retry_delta <= 0:
            return policy
        return replace(
            policy,
            min_reliability_budget_ms=min(70000, int(policy.min_reliability_budget_ms) + budget_delta_ms),
            max_reliability_retries=min(5, int(policy.max_reliability_retries) + retry_delta),
        )

    @staticmethod
    def _ema_update(previous: float, current: float, alpha: float = 0.2) -> float:
        a = max(0.01, min(1.0, float(alpha or 0.2)))
        prev = max(0.0, min(1.0, float(previous or 0.0)))
        cur = max(0.0, min(1.0, float(current or 0.0)))
        return round((a * cur) + ((1.0 - a) * prev), 6)

    @staticmethod
    def _safe_adaptive_snapshot(state: Dict[str, Any], *, bucket_key: str) -> Dict[str, Any]:
        row = dict(state or {})
        return {
            "bucket_key": str(bucket_key or ""),
            "runs": int(row.get("runs", 0) or 0),
            "ema_low_conf": float(row.get("ema_low_conf", 0.0) or 0.0),
            "ema_retry_rate": float(row.get("ema_retry_rate", 0.0) or 0.0),
            "ema_success": float(row.get("ema_success", 0.0) or 0.0),
            "bias_delta": float(row.get("bias_delta", 0.0) or 0.0),
            "last_update_run": int(row.get("last_update_run", 0) or 0),
            "applied_deltas": int(row.get("applied_deltas", 0) or 0),
            "cooldown_remaining_runs": int(row.get("cooldown_remaining_runs", 0) or 0),
            "policy_delta": dict(row.get("policy_delta") or {}),
            "last_event": dict(row.get("last_event") or {}),
        }

    def _adaptive_bucket_key(self, *, input_text: str, intent: str, mode: str) -> str:
        return adaptive_bucket_key(input_text=input_text, intent=intent, mode=normalize_chat_mode(mode))

    @staticmethod
    def _adaptive_cache_key(*, user_id: str, bucket_key: str) -> str:
        uid = str(user_id or "global").strip() or "global"
        bkey = str(bucket_key or "").strip()
        return f"{uid}::{bkey}"

    @staticmethod
    def _rollout_hash_fraction(seed: str) -> float:
        digest = hashlib.sha1(str(seed or "").encode("utf-8")).hexdigest()[:8]
        value = int(digest, 16)
        return float(value) / float(0xFFFFFFFF)

    @classmethod
    def _is_in_rollout(cls, *, seed: str, ratio: float) -> bool:
        r = max(0.0, min(1.0, float(ratio or 0.0)))
        if r >= 1.0:
            return True
        if r <= 0.0:
            return False
        return bool(cls._rollout_hash_fraction(seed) <= r)

    @staticmethod
    def _p95(values: List[float]) -> float:
        nums = sorted(float(v) for v in values if float(v) >= 0.0)
        if not nums:
            return 0.0
        idx = int(round(0.95 * (len(nums) - 1)))
        idx = max(0, min(len(nums) - 1, idx))
        return round(float(nums[idx]), 2)

    def _compute_recent_signal_metrics(self, *, window_runs: int = 200) -> Dict[str, float]:
        records = list(self.run_history_records)[-max(1, int(window_runs or 1)) :]
        total = len(records)
        if total <= 0:
            return {
                "runs": 0.0,
                "success_rate": 0.0,
                "low_conf_rate": 0.0,
                "avg_retries": 0.0,
                "parallel_exception_rate": 0.0,
            }

        success_rate = float(sum(1 for r in records if bool(r.get("success")))) / float(total)
        low_conf_rate = float(
            sum(
                1
                for r in records
                if str(r.get("confidence_level") or "").strip().upper() in {"LOW", "UNVERIFIED"}
            )
        ) / float(total)
        avg_retries = float(sum(float(r.get("retries", 0) or 0) for r in records)) / float(total)

        researcher_runs = 0.0
        researcher_parallel_exceptions = 0.0
        for r in records:
            role_metrics = dict(r.get("role_metrics") or {})
            researcher = dict(role_metrics.get("researcher") or {})
            researcher_runs += float(researcher.get("runs", 0) or 0)
            researcher_parallel_exceptions += float(researcher.get("parallel_exception_count", 0) or 0)
        parallel_exception_rate = (
            float(researcher_parallel_exceptions) / float(max(1.0, researcher_runs))
            if researcher_runs > 0
            else 0.0
        )

        return {
            "runs": float(total),
            "success_rate": round(success_rate, 4),
            "low_conf_rate": round(low_conf_rate, 4),
            "avg_retries": round(avg_retries, 4),
            "parallel_exception_rate": round(parallel_exception_rate, 4),
        }

    def _maybe_apply_auto_remediation(self) -> Optional[Dict[str, Any]]:
        cfg = dict(self.debug_config or {})
        if not bool(cfg.get("auto_remediation_enabled", True)):
            return None

        total_runs = int(len(self.run_history_records))
        min_runs = max(1, int(cfg.get("auto_remediation_min_runs", 20) or 20))
        if total_runs < min_runs:
            return None

        cooldown_runs = max(1, int(cfg.get("auto_remediation_cooldown_runs", 10) or 10))
        last_action_run = int(self.auto_remediation_state.get("last_action_run", 0) or 0)
        if last_action_run > 0 and (total_runs - last_action_run) < cooldown_runs:
            return None

        step = min(0.1, max(0.01, float(cfg.get("auto_remediation_step", 0.1) or 0.1)))
        min_apply_ratio = max(0.0, min(1.0, float(cfg.get("auto_remediation_min_apply_ratio", 0.1) or 0.1)))
        low_conf_trigger = max(0.0, min(1.0, float(cfg.get("auto_remediation_low_conf_trigger", 0.40) or 0.40)))
        retry_step_cap = max(1, int(cfg.get("auto_remediation_retry_step_cap", 2) or 2))
        window_runs = max(20, int(cfg.get("auto_remediation_window_runs", 200) or 200))

        signals = self._compute_recent_signal_metrics(window_runs=window_runs)
        actions: List[Dict[str, Any]] = []
        next_cfg = dict(cfg)

        parallel_max = max(0.0, min(1.0, float(cfg.get("slo_parallel_exception_rate_max", 0.30) or 0.30)))
        if float(signals.get("parallel_exception_rate", 0.0) or 0.0) > parallel_max:
            before_ratio = float(next_cfg.get("adaptive_apply_ratio", 1.0) or 1.0)
            after_ratio = max(min_apply_ratio, round(before_ratio - step, 4))
            if after_ratio < before_ratio:
                next_cfg["adaptive_apply_ratio"] = after_ratio
                actions.append(
                    {
                        "type": "reduce_adaptive_apply_ratio",
                        "before": before_ratio,
                        "after": after_ratio,
                        "reason": "parallel_exception_rate_above_slo",
                    }
                )
            before_bias = float(next_cfg.get("adaptive_max_bias_delta", 0.1) or 0.1)
            after_bias = max(0.02, round(before_bias - step, 4))
            if after_bias < before_bias:
                next_cfg["adaptive_max_bias_delta"] = after_bias
                actions.append(
                    {
                        "type": "reduce_adaptive_bias_delta",
                        "before": before_bias,
                        "after": after_bias,
                        "reason": "parallel_exception_rate_above_slo",
                    }
                )
        elif float(signals.get("low_conf_rate", 0.0) or 0.0) > low_conf_trigger:
            before_ratio = float(next_cfg.get("adaptive_apply_ratio", 1.0) or 1.0)
            after_ratio = min(1.0, round(before_ratio + step, 4))
            if after_ratio > before_ratio:
                next_cfg["adaptive_apply_ratio"] = after_ratio
                actions.append(
                    {
                        "type": "increase_adaptive_apply_ratio",
                        "before": before_ratio,
                        "after": after_ratio,
                        "reason": "low_conf_rate_above_trigger",
                    }
                )
            before_retry_step = int(next_cfg.get("auto_tuning_retry_step", 1) or 1)
            after_retry_step = min(retry_step_cap, before_retry_step + 1)
            if after_retry_step > before_retry_step:
                next_cfg["auto_tuning_retry_step"] = after_retry_step
                actions.append(
                    {
                        "type": "increase_retry_step",
                        "before": before_retry_step,
                        "after": after_retry_step,
                        "reason": "low_conf_rate_above_trigger",
                    }
                )

        if not actions:
            return None

        self.debug_config = next_cfg
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_count": int(total_runs),
            "signals": dict(signals),
            "actions": actions,
        }
        self.auto_remediation_events.append(event)
        self.auto_remediation_state = {
            "last_action_run": int(total_runs),
            "last_action_at": event["timestamp"],
            "last_signals": dict(signals),
            "last_actions": list(actions),
        }
        return event

    def create_adaptive_snapshot(self, *, label: str = "manual") -> Dict[str, Any]:
        self.adaptive_snapshot_seq += 1
        snapshot_id = f"snap_{self.adaptive_snapshot_seq:04d}"
        payload = {
            "id": snapshot_id,
            "label": str(label or "manual")[:64],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": copy.deepcopy(self.adaptive_learning_state),
            "cache_meta": copy.deepcopy(self.adaptive_state_cache_meta),
        }
        self.adaptive_delta_snapshots.append(payload)
        return {
            "snapshot_id": snapshot_id,
            "created_at": payload["created_at"],
            "label": payload["label"],
            "bucket_count": int(len(payload["state"])),
        }

    def rollback_latest_adaptive_snapshot(self) -> Dict[str, Any]:
        if not self.adaptive_delta_snapshots:
            return {"restored": False, "reason": "no_snapshot_available"}
        latest = dict(self.adaptive_delta_snapshots[-1] or {})
        self.adaptive_learning_state = copy.deepcopy(dict(latest.get("state") or {}))
        self.adaptive_state_cache_meta = copy.deepcopy(dict(latest.get("cache_meta") or {}))
        self.adaptive_state_loaded_keys = set(self.adaptive_learning_state.keys())
        return {
            "restored": True,
            "snapshot_id": str(latest.get("id") or ""),
            "created_at": str(latest.get("created_at") or ""),
            "bucket_count": int(len(self.adaptive_learning_state)),
        }

    def _get_adaptive_bucket_state(self, *, user_id: str, bucket_key: str) -> Dict[str, Any]:
        key = str(bucket_key or "")
        cache_key = self._adaptive_cache_key(user_id=str(user_id or "global"), bucket_key=key)
        ttl_sec = max(0, int((self.debug_config or {}).get("adaptive_state_cache_ttl_sec", 90) or 90))
        now_ts = float(time.time())
        meta = dict(self.adaptive_state_cache_meta.get(cache_key) or {})
        fetched_at = float(meta.get("fetched_at_ts", 0.0) or 0.0)
        fresh = bool(ttl_sec <= 0 or (fetched_at > 0 and (now_ts - fetched_at) <= float(ttl_sec)))

        if fresh and cache_key in self.adaptive_learning_state:
            return dict(self.adaptive_learning_state.get(cache_key) or {})
        if fresh and cache_key in self.adaptive_state_loaded_keys:
            return {}

        self.adaptive_state_loaded_keys.add(cache_key)
        stored = self.adaptive_store.get_state(user_id=str(user_id or "global"), bucket_key=key)
        if isinstance(stored, dict) and stored:
            self.adaptive_learning_state[cache_key] = dict(stored)
            self.adaptive_state_cache_meta[cache_key] = {"fetched_at_ts": now_ts, "source": "firestore_hit"}
            return dict(stored)
        self.adaptive_state_cache_meta[cache_key] = {"fetched_at_ts": now_ts, "source": "firestore_miss"}
        return {}

    def _update_adaptive_learning_state(self, *, payload: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(self.debug_config or {})
        if not bool(cfg.get("adaptive_enabled", True)):
            return {"adaptive_enabled": False}

        query = str(payload.get("query") or "").strip()
        if not query:
            return {"adaptive_enabled": True, "updated": False}
        user_id = str(payload.get("user_id") or "global")
        mode = normalize_chat_mode(str(payload.get("mode") or "smart"))
        intent = str(payload.get("detected_intent") or "general").strip().lower() or "general"
        retries = int(payload.get("retries", 0) or 0)
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        success = bool(payload.get("success", False))
        failure_confidence_threshold = max(
            0.0,
            min(1.0, float(cfg.get("failure_confidence_threshold", 0.45) or 0.45)),
        )
        important_case = bool((confidence <= failure_confidence_threshold) or (retries > 1) or (not success))

        bucket_key = self._adaptive_bucket_key(input_text=query, intent=intent, mode=mode)
        state = self._get_adaptive_bucket_state(user_id=user_id, bucket_key=bucket_key)

        alpha = max(0.01, min(1.0, float(cfg.get("adaptive_ema_alpha", 0.2) or 0.2)))
        min_cluster_runs = max(1, int(cfg.get("adaptive_min_cluster_runs", 8) or 8))
        low_conf_threshold = max(0.05, min(0.95, float(cfg.get("adaptive_low_conf_threshold", 0.35) or 0.35)))
        retry_rate_threshold = max(0.05, min(0.95, float(cfg.get("adaptive_retry_threshold", 0.30) or 0.30)))
        cooldown_runs = max(1, int(cfg.get("adaptive_cooldown_runs", 6) or 6))
        max_bias_delta = max(0.01, min(0.5, float(cfg.get("adaptive_max_bias_delta", 0.1) or 0.1)))
        delta_step = min(0.1, max_bias_delta)

        runs = int(state.get("runs", 0) or 0) + 1
        low_conf_obs = 1.0 if confidence <= failure_confidence_threshold else 0.0
        retry_obs = 1.0 if retries > 1 else 0.0
        success_obs = 1.0 if success else 0.0

        ema_low_conf = self._ema_update(float(state.get("ema_low_conf", 0.0) or 0.0), low_conf_obs, alpha=alpha)
        ema_retry_rate = self._ema_update(float(state.get("ema_retry_rate", 0.0) or 0.0), retry_obs, alpha=alpha)
        ema_success = self._ema_update(float(state.get("ema_success", 0.0) or 0.0), success_obs, alpha=alpha)

        prev_bias_delta = max(0.0, float(state.get("bias_delta", 0.0) or 0.0))
        last_update_run = int(state.get("last_update_run", 0) or 0)
        # First qualifying run should be allowed immediately; cooldown starts after first applied delta.
        runs_since_last_update = cooldown_runs if last_update_run <= 0 else max(0, runs - last_update_run)
        cooldown_remaining_runs = max(0, cooldown_runs - runs_since_last_update)
        trigger = bool((ema_low_conf >= low_conf_threshold) or (ema_retry_rate >= retry_rate_threshold))
        can_apply = bool(runs >= min_cluster_runs and cooldown_remaining_runs <= 0 and trigger)

        new_bias_delta = prev_bias_delta
        applied_deltas = int(state.get("applied_deltas", 0) or 0)
        event: Dict[str, Any] = {}
        if can_apply:
            candidate = min(max_bias_delta, prev_bias_delta + delta_step)
            if candidate > prev_bias_delta + 1e-9:
                new_bias_delta = round(candidate, 6)
                applied_deltas += 1
                last_update_run = runs
                event = {
                    "bucket_key": bucket_key,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": "low_conf_or_retry_cluster",
                    "ema_low_conf": ema_low_conf,
                    "ema_retry_rate": ema_retry_rate,
                    "bias_delta_before": round(prev_bias_delta, 6),
                    "bias_delta_after": round(new_bias_delta, 6),
                    "mode": mode,
                    "intent": intent,
                }
                self.adaptive_learning_events.append(event)

        budget_bonus_ms = int(round(20000.0 * new_bias_delta))
        retry_bonus = 1 if new_bias_delta >= 0.10 else 0
        strict_boost = bool(ema_low_conf >= max(0.50, low_conf_threshold))
        multi_boost = bool(ema_low_conf >= max(0.45, low_conf_threshold) or ema_retry_rate >= retry_rate_threshold)
        force_tools_boost = bool(trigger and intent in {"research", "general"})

        next_state = {
            "user_id": user_id,
            "bucket_key": bucket_key,
            "fingerprint": str(bucket_key.split(":", 1)[0]),
            "intent": intent,
            "mode": mode,
            "normalized_query": normalize_for_fingerprint(query),
            "runs": runs,
            "important_cases": int(state.get("important_cases", 0) or 0) + (1 if important_case else 0),
            "ema_low_conf": ema_low_conf,
            "ema_retry_rate": ema_retry_rate,
            "ema_success": ema_success,
            "bias_delta": new_bias_delta,
            "last_update_run": last_update_run,
            "applied_deltas": applied_deltas,
            "cooldown_remaining_runs": max(0, cooldown_runs - max(0, runs - last_update_run)),
            "policy_delta": {
                "budget_bonus_ms": budget_bonus_ms,
                "retry_bonus": retry_bonus,
                "force_tools": force_tools_boost,
                "strict_evidence": strict_boost,
                "require_multi_source": multi_boost,
            },
            "last_event": event,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_key = self._adaptive_cache_key(user_id=user_id, bucket_key=bucket_key)
        self.adaptive_learning_state[cache_key] = dict(next_state)
        self.adaptive_state_loaded_keys.add(cache_key)
        self.adaptive_state_cache_meta[cache_key] = {"fetched_at_ts": float(time.time()), "source": "local_update"}

        write_every_runs = max(1, int(cfg.get("adaptive_state_write_every_runs", 3) or 3))
        should_persist_state = bool(important_case or bool(event) or runs <= 1 or (runs % write_every_runs == 0))
        if should_persist_state:
            self.adaptive_store.upsert_state(user_id=user_id, bucket_key=bucket_key, state_payload=next_state)
        return {
            "adaptive_enabled": True,
            "updated": True,
            "bucket_key": bucket_key,
            "state_persisted": bool(should_persist_state),
            "snapshot": self._safe_adaptive_snapshot(next_state, bucket_key=bucket_key),
        }

    def _apply_adaptive_learning_overrides(
        self,
        *,
        policy: ModePolicy,
        dynamic_context: Dict[str, Any],
        goal: str,
        user_id: str,
    ) -> Tuple[ModePolicy, Dict[str, Any]]:
        cfg = dict(self.debug_config or {})
        intent = str((dynamic_context or {}).get("intent") or "general").strip().lower() or "general"
        bucket_key = self._adaptive_bucket_key(input_text=goal, intent=intent, mode=policy.mode_name)
        state = self._get_adaptive_bucket_state(user_id=str(user_id or "global"), bucket_key=bucket_key)
        snapshot = self._safe_adaptive_snapshot(state, bucket_key=bucket_key)
        if not bool(cfg.get("adaptive_enabled", True)):
            return policy, {
                "enabled": False,
                "bucket_key": bucket_key,
                "snapshot": snapshot,
                "applied": False,
                "applied_fields": [],
            }

        raw_apply_ratio = cfg.get("adaptive_apply_ratio", 1.0)
        if raw_apply_ratio is None:
            raw_apply_ratio = 1.0
        apply_ratio = max(0.0, min(1.0, float(raw_apply_ratio)))
        rollout_seed = f"{str(user_id or 'global')}::{bucket_key}"
        rollout_applied = self._is_in_rollout(seed=rollout_seed, ratio=apply_ratio)
        if not rollout_applied:
            return policy, {
                "enabled": True,
                "bucket_key": bucket_key,
                "snapshot": snapshot,
                "rollout_applied": False,
                "apply_ratio": round(apply_ratio, 4),
                "applied": False,
                "applied_fields": [],
            }

        deltas = dict(state.get("policy_delta") or {})
        applied_fields: List[str] = []
        candidate = policy

        budget_bonus = max(0, int(deltas.get("budget_bonus_ms", 0) or 0))
        if budget_bonus > 0:
            candidate = replace(
                candidate,
                min_reliability_budget_ms=min(70000, int(candidate.min_reliability_budget_ms) + budget_bonus),
            )
            applied_fields.append("min_reliability_budget_ms")

        retry_bonus = max(0, int(deltas.get("retry_bonus", 0) or 0))
        if retry_bonus > 0:
            candidate = replace(
                candidate,
                max_reliability_retries=min(5, int(candidate.max_reliability_retries) + retry_bonus),
            )
            applied_fields.append("max_reliability_retries")

        if bool(deltas.get("force_tools")):
            candidate = replace(candidate, tool_force_for_non_smalltalk=True)
            applied_fields.append("tool_force_for_non_smalltalk")

        if bool(deltas.get("strict_evidence")):
            candidate = replace(
                candidate,
                strict_evidence=True,
                evidence_softpass_with_existing_sources=False,
                trusted_source_softpass_with_existing_sources=False,
            )
            applied_fields.extend(
                [
                    "strict_evidence",
                    "evidence_softpass_with_existing_sources",
                    "trusted_source_softpass_with_existing_sources",
                ]
            )

        if bool(deltas.get("require_multi_source")):
            candidate = replace(candidate, require_multi_source=True)
            applied_fields.append("require_multi_source")

        return candidate, {
            "enabled": True,
            "bucket_key": bucket_key,
            "snapshot": snapshot,
            "rollout_applied": True,
            "apply_ratio": round(apply_ratio, 4),
            "applied": bool(applied_fields),
            "applied_fields": sorted(set(applied_fields)),
        }

    @staticmethod
    def _final_safety_clamp(
        *,
        mode_policy: ModePolicy,
        candidate_policy: ModePolicy,
    ) -> Tuple[ModePolicy, Dict[str, Any]]:
        updates: Dict[str, Any] = {}
        fields: List[str] = []

        def _mark(field_name: str, value: Any) -> None:
            updates[field_name] = value
            fields.append(field_name)

        if bool(mode_policy.live_query_forces_tools) and not bool(candidate_policy.live_query_forces_tools):
            _mark("live_query_forces_tools", True)
        if bool(mode_policy.tool_force_for_non_smalltalk) and not bool(candidate_policy.tool_force_for_non_smalltalk):
            _mark("tool_force_for_non_smalltalk", True)
        if bool(mode_policy.strict_evidence) and not bool(candidate_policy.strict_evidence):
            _mark("strict_evidence", True)
        if bool(mode_policy.require_multi_source) and not bool(candidate_policy.require_multi_source):
            _mark("require_multi_source", True)
        if bool(mode_policy.require_trusted_sources_only) and not bool(candidate_policy.require_trusted_sources_only):
            _mark("require_trusted_sources_only", True)

        min_budget = max(int(candidate_policy.min_reliability_budget_ms), int(mode_policy.min_reliability_budget_ms))
        if min_budget != int(candidate_policy.min_reliability_budget_ms):
            _mark("min_reliability_budget_ms", min_budget)

        min_retries = max(int(candidate_policy.max_reliability_retries), int(mode_policy.max_reliability_retries))
        if min_retries != int(candidate_policy.max_reliability_retries):
            _mark("max_reliability_retries", min_retries)

        if not updates:
            return candidate_policy, {"clamp_applied": False, "clamp_fields": []}
        return replace(candidate_policy, **updates), {
            "clamp_applied": True,
            "clamp_fields": sorted(set(fields)),
        }

    def _record_debug_run(self, entry: Dict[str, Any]) -> None:
        payload = dict(entry or {})
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        payload["user_id"] = str(payload.get("user_id") or "global")
        payload["detected_intent"] = str(payload.get("detected_intent") or "general").strip().lower() or "general"
        payload["requested_mode_raw"] = str(payload.get("requested_mode_raw") or payload.get("requested_mode") or "smart").strip().lower()
        payload["mode"] = normalize_chat_mode(str(payload.get("mode") or "smart"))
        payload["requested_mode"] = normalize_chat_mode(str(payload.get("requested_mode") or payload["mode"]))
        payload["confidence"] = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
        payload["confidence_level"] = self._confidence_level(float(payload["confidence"]))
        payload["tool_count"] = max(0, int(payload.get("tool_count", 0) or 0))
        payload["retries"] = max(0, int(payload.get("retries", 0) or 0))
        payload["response_time_ms"] = max(0, int(payload.get("response_time_ms", 0) or 0))
        payload["success"] = bool(payload.get("success", False))
        payload["override_applied"] = bool(payload.get("override_applied", False))
        payload["role_flow"] = list(payload.get("role_flow") or [])[-120:]
        payload["role_metrics"] = dict(payload.get("role_metrics") or {})
        payload["role_summary"] = dict(payload.get("role_summary") or {})
        self.run_history_records.append(payload)

        failure_threshold = max(0.0, min(1.0, float(self.debug_config.get("failure_confidence_threshold", 0.45) or 0.45)))
        important_case = bool((not payload["success"]) or (payload["confidence"] <= failure_threshold) or (int(payload["retries"]) > 1))
        if important_case:
            review_item = {
                "timestamp": payload["timestamp"],
                "query": str(payload.get("query") or "")[:600],
                "answer_preview": str(payload.get("answer_preview") or "")[:1200],
                "mode": payload["mode"],
                "requested_mode": payload["requested_mode"],
                "confidence": payload["confidence"],
                "confidence_level": payload["confidence_level"],
                "tools_used": list(payload.get("tools_used") or []),
                "retries": payload["retries"],
                "override_reason": str(payload.get("override_reason") or ""),
                "override_applied": payload["override_applied"],
                "success": payload["success"],
            }
            self.low_confidence_reviews.append(review_item)
            self.adaptive_learning_cases.append(dict(review_item))
            self.adaptive_store.log_case(user_id=payload["user_id"], case_payload=dict(review_item))

        adaptive_update = self._update_adaptive_learning_state(payload=payload)
        payload["adaptive_learning"] = dict(adaptive_update or {})
        remediation_event = self._maybe_apply_auto_remediation()
        if remediation_event:
            payload["auto_remediation"] = dict(remediation_event)

    def get_debug_snapshot(self, *, range_window: str = "24h", limit: int = 25) -> Dict[str, Any]:
        now_ts = datetime.now(timezone.utc).timestamp()
        since_seconds = self._range_seconds(range_window)
        since_ts = (now_ts - since_seconds) if since_seconds is not None else None
        max_items = max(5, min(100, int(limit or 25)))

        records = list(self.run_history_records)
        if since_ts is not None:
            filtered_records = []
            for r in records:
                try:
                    ts = datetime.fromisoformat(str(r.get("timestamp") or "").replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = now_ts
                if ts >= since_ts:
                    filtered_records.append(r)
            records = filtered_records

        total = len(records)
        succeeded = sum(1 for r in records if bool(r.get("success")))
        confidence_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        per_mode: Dict[str, Dict[str, Any]] = {}
        per_role: Dict[str, Dict[str, Any]] = {}
        role_flow_recent: List[Dict[str, Any]] = []
        overrides_recent: List[Dict[str, Any]] = []
        override_reasons: Dict[str, int] = {}
        override_transitions: Dict[str, int] = {}

        for record in records:
            level = str(record.get("confidence_level") or "LOW").upper()
            if level == "MODERATE":
                level = "MEDIUM"
            if level not in confidence_counts:
                level = "LOW"
            confidence_counts[level] += 1

            mode = normalize_chat_mode(str(record.get("mode") or "smart"))
            bucket = per_mode.setdefault(
                mode,
                {"runs": 0, "successes": 0, "latency_total": 0, "retries_total": 0, "tool_runs": 0},
            )
            bucket["runs"] += 1
            bucket["successes"] += 1 if bool(record.get("success")) else 0
            bucket["latency_total"] += int(record.get("response_time_ms", 0) or 0)
            bucket["retries_total"] += int(record.get("retries", 0) or 0)
            bucket["tool_runs"] += 1 if int(record.get("tool_count", 0) or 0) > 0 else 0

            record_role_metrics = dict(record.get("role_metrics") or {})
            for role_name, role_data in record_role_metrics.items():
                role_key = str(role_name or "executor").strip().lower() or "executor"
                role_bucket = per_role.setdefault(
                    role_key,
                    {
                        "runs": 0,
                        "successes": 0,
                        "latency_total": 0,
                        "retries_total": 0,
                        "tool_calls_total": 0,
                        "parallel_exception_count": 0,
                    },
                )
                role_bucket["runs"] += int(role_data.get("runs", 0) or 0)
                role_bucket["successes"] += int(role_data.get("success_count", 0) or 0)
                role_bucket["latency_total"] += int(role_data.get("latency_ms_total", 0) or 0)
                role_bucket["retries_total"] += int(role_data.get("retries_total", 0) or 0)
                role_bucket["tool_calls_total"] += int(role_data.get("tool_calls_total", 0) or 0)
                role_bucket["parallel_exception_count"] += int(role_data.get("parallel_exception_count", 0) or 0)

            for role_row in list(record.get("role_flow") or []):
                if not isinstance(role_row, dict):
                    continue
                row = dict(role_row)
                row["at"] = str(record.get("timestamp") or "")
                role_flow_recent.append(row)

            if bool(record.get("override_applied")):
                reason = str(record.get("override_reason") or "unspecified").strip() or "unspecified"
                override_reasons[reason] = int(override_reasons.get(reason, 0)) + 1
                src_mode = normalize_chat_mode(str(record.get("requested_mode") or mode))
                dst_mode = normalize_chat_mode(str(record.get("mode") or mode))
                transition = f"{src_mode}->{dst_mode}"
                override_transitions[transition] = int(override_transitions.get(transition, 0)) + 1
                overrides_recent.append(
                    {
                        "at": str(record.get("timestamp") or ""),
                        "auto_selected": src_mode,
                        "overridden_to": dst_mode,
                        "reason": reason,
                    }
                )

        per_mode_stats = []
        for mode in ("smart", "agent", "research_pro"):
            data = per_mode.get(mode) or {"runs": 0, "successes": 0, "latency_total": 0, "retries_total": 0, "tool_runs": 0}
            runs = max(1, int(data["runs"]))
            per_mode_stats.append(
                {
                    "mode": mode,
                    "runs": int(data["runs"]),
                    "avg_latency_ms": round(float(data["latency_total"]) / runs, 2) if data["runs"] else 0.0,
                    "success_rate": round(float(data["successes"]) / runs, 4) if data["runs"] else 0.0,
                    "avg_retries": round(float(data["retries_total"]) / runs, 3) if data["runs"] else 0.0,
                    "tool_usage_rate": round(float(data["tool_runs"]) / runs, 4) if data["runs"] else 0.0,
                }
            )

        per_role_stats = []
        for role in ("planner", "researcher", "executor", "critic", "synthesizer"):
            data = per_role.get(role) or {
                "runs": 0,
                "successes": 0,
                "latency_total": 0,
                "retries_total": 0,
                "tool_calls_total": 0,
                "parallel_exception_count": 0,
            }
            runs = max(1, int(data["runs"]))
            parallel_exception_count = int(data.get("parallel_exception_count", 0) or 0)
            per_role_stats.append(
                {
                    "role": role,
                    "runs": int(data["runs"]),
                    "avg_latency_ms": round(float(data["latency_total"]) / runs, 2) if data["runs"] else 0.0,
                    "success_rate": round(float(data["successes"]) / runs, 4) if data["runs"] else 0.0,
                    "avg_retries": round(float(data["retries_total"]) / runs, 3) if data["runs"] else 0.0,
                    "avg_tool_calls": round(float(data["tool_calls_total"]) / runs, 3) if data["runs"] else 0.0,
                    "parallel_exception_count": parallel_exception_count,
                    "avg_parallel_exceptions": round(float(parallel_exception_count) / runs, 3) if data["runs"] else 0.0,
                    "parallel_exception_rate": round(float(parallel_exception_count) / runs, 4) if data["runs"] else 0.0,
                }
            )

        confidence_distribution = {
            "high": confidence_counts["HIGH"],
            "medium": confidence_counts["MEDIUM"],
            "low": confidence_counts["LOW"],
            "high_pct": round((confidence_counts["HIGH"] / total), 4) if total else 0.0,
            "medium_pct": round((confidence_counts["MEDIUM"] / total), 4) if total else 0.0,
            "low_pct": round((confidence_counts["LOW"] / total), 4) if total else 0.0,
        }

        reviews = list(self.low_confidence_reviews)
        if since_ts is not None:
            filtered_reviews = []
            for r in reviews:
                try:
                    ts = datetime.fromisoformat(str(r.get("timestamp") or "").replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = now_ts
                if ts >= since_ts:
                    filtered_reviews.append(r)
            reviews = filtered_reviews

        adaptive_rows = list(self.adaptive_learning_state.values())
        adaptive_rows.sort(
            key=lambda row: (
                (
                    (float(row.get("ema_low_conf", 0.0) or 0.0) * 0.6)
                    + (float(row.get("ema_retry_rate", 0.0) or 0.0) * 0.3)
                    + ((1.0 - float(row.get("ema_success", 0.0) or 0.0)) * 0.1)
                )
                * max(1.0, min(20.0, float(row.get("runs", 0) or 0)))
            ),
            reverse=True,
        )
        top_failing_buckets = []
        low_confidence_clusters = []
        high_retry_queries = []
        for row in adaptive_rows[: max_items]:
            impact_score = (
                (float(row.get("ema_low_conf", 0.0) or 0.0) * 0.6)
                + (float(row.get("ema_retry_rate", 0.0) or 0.0) * 0.3)
                + ((1.0 - float(row.get("ema_success", 0.0) or 0.0)) * 0.1)
            ) * max(1.0, min(20.0, float(row.get("runs", 0) or 0)))
            item = {
                "bucket_key": str(row.get("bucket_key") or ""),
                "fingerprint": str(row.get("fingerprint") or ""),
                "intent": str(row.get("intent") or "general"),
                "mode": str(row.get("mode") or "smart"),
                "runs": int(row.get("runs", 0) or 0),
                "ema_low_conf": round(float(row.get("ema_low_conf", 0.0) or 0.0), 4),
                "ema_retry_rate": round(float(row.get("ema_retry_rate", 0.0) or 0.0), 4),
                "ema_success": round(float(row.get("ema_success", 0.0) or 0.0), 4),
                "impact_score": round(float(impact_score), 4),
                "cooldown_remaining_runs": int(row.get("cooldown_remaining_runs", 0) or 0),
                "policy_delta": dict(row.get("policy_delta") or {}),
            }
            top_failing_buckets.append(item)
            if item["ema_low_conf"] >= float(self.debug_config.get("adaptive_low_conf_threshold", 0.35) or 0.35):
                low_confidence_clusters.append(item)
            if item["ema_retry_rate"] >= float(self.debug_config.get("adaptive_retry_threshold", 0.30) or 0.30):
                high_retry_queries.append(item)
        prompt_counts: Dict[str, int] = {}
        for row in reviews:
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            prompt_counts[query] = int(prompt_counts.get(query, 0)) + 1
        top_failing_prompts = [
            {"query": q, "count": c}
            for q, c in sorted(prompt_counts.items(), key=lambda kv: kv[1], reverse=True)[:max(max_items, 20)]
        ]

        latencies = [float(r.get("response_time_ms", 0) or 0) for r in records]
        retries_all = [float(r.get("retries", 0) or 0) for r in records]
        p95_latency_ms = self._p95(latencies)
        avg_retries = round((sum(retries_all) / max(1, len(retries_all))), 3) if retries_all else 0.0
        low_conf_rate = round((confidence_distribution.get("low_pct", 0.0) or 0.0), 4)
        slo_p95_max = float(self.debug_config.get("slo_p95_latency_ms_max", 25000.0) or 25000.0)
        slo_low_conf_max = float(self.debug_config.get("slo_low_conf_rate_max", 0.35) or 0.35)
        slo_avg_retries_max = float(self.debug_config.get("slo_avg_retries_max", 1.5) or 1.5)
        slo_parallel_exception_rate_max = float(self.debug_config.get("slo_parallel_exception_rate_max", 0.30) or 0.30)
        researcher_stats = next((row for row in per_role_stats if str(row.get("role")) == "researcher"), {}) or {}
        parallel_exception_rate = float(researcher_stats.get("parallel_exception_rate", 0.0) or 0.0)
        slo_status = {
            "p95_latency_ok": bool(p95_latency_ms <= slo_p95_max),
            "low_conf_rate_ok": bool(low_conf_rate <= slo_low_conf_max),
            "avg_retries_ok": bool(avg_retries <= slo_avg_retries_max),
            "parallel_exception_rate_ok": bool(parallel_exception_rate <= slo_parallel_exception_rate_max),
        }
        adaptive_applied_runs = sum(
            1 for r in records if bool(((r.get("adaptive_policy") or {}).get("rollout_applied", True)))
        )
        adaptive_total_runs = sum(1 for r in records if isinstance(r.get("adaptive_policy"), dict))
        adaptive_applied_rate = round((adaptive_applied_runs / adaptive_total_runs), 4) if adaptive_total_runs else 0.0

        def _canary_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
            runs = int(len(rows))
            if runs <= 0:
                return {
                    "runs": 0,
                    "success_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_retries": 0.0,
                    "parallel_exception_rate": 0.0,
                }
            success_rate = float(sum(1 for r in rows if bool(r.get("success")))) / float(runs)
            avg_latency_ms = float(sum(float(r.get("response_time_ms", 0) or 0) for r in rows)) / float(runs)
            avg_retries_local = float(sum(float(r.get("retries", 0) or 0) for r in rows)) / float(runs)
            researcher_runs = 0.0
            researcher_parallel_ex = 0.0
            for r in rows:
                role_metrics = dict(r.get("role_metrics") or {})
                researcher = dict(role_metrics.get("researcher") or {})
                researcher_runs += float(researcher.get("runs", 0) or 0)
                researcher_parallel_ex += float(researcher.get("parallel_exception_count", 0) or 0)
            parallel_ex_rate = (
                float(researcher_parallel_ex) / float(max(1.0, researcher_runs))
                if researcher_runs > 0
                else 0.0
            )
            return {
                "runs": runs,
                "success_rate": round(success_rate, 4),
                "avg_latency_ms": round(avg_latency_ms, 2),
                "avg_retries": round(avg_retries_local, 3),
                "parallel_exception_rate": round(parallel_ex_rate, 4),
            }

        adaptive_rows_for_compare = [
            r for r in records
            if isinstance(r.get("adaptive_policy"), dict)
            and bool((r.get("adaptive_policy") or {}).get("rollout_applied", True))
        ]
        baseline_rows_for_compare = [
            r for r in records
            if isinstance(r.get("adaptive_policy"), dict)
            and not bool((r.get("adaptive_policy") or {}).get("rollout_applied", True))
        ]
        canary_adaptive = _canary_stats(adaptive_rows_for_compare)
        canary_baseline = _canary_stats(baseline_rows_for_compare)
        canary_delta = {
            "success_rate": round(float(canary_adaptive["success_rate"]) - float(canary_baseline["success_rate"]), 4),
            "avg_latency_ms": round(float(canary_adaptive["avg_latency_ms"]) - float(canary_baseline["avg_latency_ms"]), 2),
            "avg_retries": round(float(canary_adaptive["avg_retries"]) - float(canary_baseline["avg_retries"]), 3),
            "parallel_exception_rate": round(
                float(canary_adaptive["parallel_exception_rate"]) - float(canary_baseline["parallel_exception_rate"]),
                4,
            ),
        }

        latest_snapshot = dict(self.adaptive_delta_snapshots[-1] or {}) if self.adaptive_delta_snapshots else {}
        tool_records = sorted(
            list(self.tool_confidence_store.snapshot() or []),
            key=lambda r: float(r.selection_score()),
            reverse=True,
        )
        tool_reliability = {
            "count": int(len(tool_records)),
            "top_tools": [
                {
                    "tool": str(rec.tool_name or ""),
                    "score": round(float(rec.selection_score()), 4),
                    "success_rate": round(float(rec.success_rate()), 4),
                    "error_rate": round(float(rec.error_rate()), 4),
                    "latency_ema_ms": round(float(rec.latency_ema_ms), 2),
                    "latency_score": round(float(rec.latency_score()), 4),
                    "samples": int(rec.samples),
                    "avg_relevance": round(float(rec.avg_relevance()), 4),
                }
                for rec in tool_records[:15]
            ],
            "weak_tools": [
                {
                    "tool": str(rec.tool_name or ""),
                    "score": round(float(rec.selection_score()), 4),
                    "error_rate": round(float(rec.error_rate()), 4),
                    "samples": int(rec.samples),
                }
                for rec in tool_records
                if int(rec.samples) >= int(max(3, TOOL_CONFIDENCE_MIN_SAMPLES)) and float(rec.selection_score()) < 0.30
            ][:10],
        }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "range": str(range_window or "24h"),
            "totals": {
                "runs": int(total),
                "successes": int(succeeded),
                "success_rate": round((float(succeeded) / float(total)), 4) if total else 0.0,
            },
            "per_mode_stats": per_mode_stats,
            "per_role_stats": per_role_stats,
            "role_flow": {
                "count": int(len(role_flow_recent)),
                "recent": role_flow_recent[-max_items:],
            },
            "override_insights": {
                "count": int(len(overrides_recent)),
                "reason_counts": override_reasons,
                "transition_counts": override_transitions,
                "recent": overrides_recent[-max_items:],
            },
            "confidence_distribution": confidence_distribution,
            "failure_analyzer": {
                "count": int(len(reviews)),
                "items": reviews[-max_items:],
            },
            "auto_tuning": {
                "enabled": bool(self.debug_config.get("auto_tuning_enabled", True)),
                "state": dict(self.auto_tuning_state),
                "recent_events": list(self.auto_tuning_events)[-max_items:],
            },
            "auto_remediation": {
                "enabled": bool(self.debug_config.get("auto_remediation_enabled", True)),
                "state": dict(self.auto_remediation_state),
                "recent_events": list(self.auto_remediation_events)[-max_items:],
            },
            "adaptive_learning": {
                "enabled": bool(self.debug_config.get("adaptive_enabled", True)),
                "state_size": int(len(self.adaptive_learning_state)),
                "top_failing_prompts": top_failing_prompts,
                "top_failing_buckets": top_failing_buckets[:max_items],
                "low_confidence_clusters": low_confidence_clusters[:max_items],
                "high_retry_queries": high_retry_queries[:max_items],
                "recent_cases": list(self.adaptive_learning_cases)[-max_items:],
                "rule_events_recent": list(self.adaptive_learning_events)[-max_items:],
                "rollout": {
                    "adaptive_apply_ratio": float(
                        1.0 if self.debug_config.get("adaptive_apply_ratio", 1.0) is None else self.debug_config.get("adaptive_apply_ratio", 1.0)
                    ),
                    "adaptive_applied_rate": adaptive_applied_rate,
                    "runs_observed": int(adaptive_total_runs),
                },
                "canary_compare": {
                    "baseline": canary_baseline,
                    "adaptive": canary_adaptive,
                    "delta": canary_delta,
                },
                "snapshot_status": {
                    "count": int(len(self.adaptive_delta_snapshots)),
                    "latest_id": str(latest_snapshot.get("id") or ""),
                    "latest_created_at": str(latest_snapshot.get("created_at") or ""),
                },
            },
            "slo": {
                "targets": {
                    "p95_latency_ms_max": slo_p95_max,
                    "low_conf_rate_max": slo_low_conf_max,
                    "avg_retries_max": slo_avg_retries_max,
                    "parallel_exception_rate_max": slo_parallel_exception_rate_max,
                },
                "current": {
                    "p95_latency_ms": p95_latency_ms,
                    "low_conf_rate": low_conf_rate,
                    "avg_retries": avg_retries,
                    "parallel_exception_rate": parallel_exception_rate,
                },
                "status": slo_status,
            },
            "tool_reliability": tool_reliability,
            "config": dict(self.debug_config),
        }

    def update_debug_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        incoming = dict(patch or {})
        next_cfg = dict(self.debug_config)
        adaptive_keys = {
            "adaptive_enabled",
            "adaptive_ema_alpha",
            "adaptive_min_cluster_runs",
            "adaptive_low_conf_threshold",
            "adaptive_retry_threshold",
            "adaptive_cooldown_runs",
            "adaptive_max_bias_delta",
            "adaptive_state_cache_ttl_sec",
            "adaptive_state_write_every_runs",
            "adaptive_apply_ratio",
        }
        if any(k in incoming for k in adaptive_keys):
            self.create_adaptive_snapshot(label="pre_config_update")
        if "auto_tuning_enabled" in incoming:
            next_cfg["auto_tuning_enabled"] = bool(incoming.get("auto_tuning_enabled"))
        if "adaptive_enabled" in incoming:
            next_cfg["adaptive_enabled"] = bool(incoming.get("adaptive_enabled"))
        if "auto_remediation_enabled" in incoming:
            next_cfg["auto_remediation_enabled"] = bool(incoming.get("auto_remediation_enabled"))
        for key in (
            "auto_tuning_min_runs",
            "auto_tuning_cooldown_runs",
            "auto_tuning_budget_step_ms",
            "auto_tuning_retry_step",
            "auto_tuning_max_budget_delta_ms",
            "auto_tuning_max_retry_delta",
            "adaptive_min_cluster_runs",
            "adaptive_cooldown_runs",
            "adaptive_state_cache_ttl_sec",
            "adaptive_state_write_every_runs",
            "auto_remediation_min_runs",
            "auto_remediation_cooldown_runs",
            "auto_remediation_retry_step_cap",
            "auto_remediation_window_runs",
        ):
            if key in incoming:
                try:
                    value = int(incoming.get(key))
                    if key == "adaptive_state_cache_ttl_sec":
                        next_cfg[key] = max(0, value)
                    else:
                        next_cfg[key] = max(1, value)
                except Exception:
                    pass
        for key in (
            "auto_tuning_success_threshold",
            "failure_confidence_threshold",
            "adaptive_ema_alpha",
            "adaptive_low_conf_threshold",
            "adaptive_retry_threshold",
            "adaptive_max_bias_delta",
            "adaptive_apply_ratio",
            "slo_low_conf_rate_max",
            "slo_parallel_exception_rate_max",
            "auto_remediation_step",
            "auto_remediation_min_apply_ratio",
            "auto_remediation_low_conf_trigger",
        ):
            if key in incoming:
                try:
                    next_cfg[key] = max(0.0, min(1.0, float(incoming.get(key))))
                except Exception:
                    pass
        for key in ("slo_p95_latency_ms_max", "slo_avg_retries_max"):
            if key in incoming:
                try:
                    floor = 1000.0 if key == "slo_p95_latency_ms_max" else 0.0
                    next_cfg[key] = max(floor, float(incoming.get(key)))
                except Exception:
                    pass
        self.debug_config = next_cfg
        return dict(self.debug_config)

    def _latest_outcome_for_mode_check(self, *, query: str, requested_mode: str) -> Optional[Dict[str, Any]]:
        target_query = str(query or "").strip().lower()
        target_mode = normalize_chat_mode(str(requested_mode or "smart"))
        if not target_query:
            return None
        records = list(self.run_history_records)
        for record in reversed(records):
            rec_query = str(record.get("query") or "").strip().lower()
            if rec_query != target_query:
                continue
            rec_mode = normalize_chat_mode(str(record.get("requested_mode") or "smart"))
            if rec_mode == target_mode:
                return dict(record)
        for record in reversed(records):
            rec_query = str(record.get("query") or "").strip().lower()
            if rec_query == target_query:
                return dict(record)
        return None

    @classmethod
    def _predict_tools_expected(cls, *, goal: str, mode_policy: ModePolicy, small_talk_only: bool) -> bool:
        forced_tooling = False
        if mode_policy.live_query_forces_tools and cls._is_hybrid_live_query(goal):
            forced_tooling = True
        if mode_policy.tool_force_for_non_smalltalk and not small_talk_only:
            forced_tooling = True
        if mode_policy.respect_small_talk_for_forced_tools and small_talk_only:
            forced_tooling = False
        return bool(forced_tooling)

    @staticmethod
    def _map_override_reason_for_ui(reason: Optional[str], *, dynamic_context: Dict[str, Any]) -> str:
        raw = str(reason or "").strip()
        if not raw:
            return "none"
        if raw == "critical_research_trigger" and bool((dynamic_context or {}).get("time_sensitive")):
            return "time_sensitive_query"
        if raw == "auto_intent_research":
            return "auto_research_intent"
        if raw == "auto_intent_task":
            return "auto_task_intent"
        if raw == "auto_intent_chat":
            return "auto_chat_intent"
        return raw

    @classmethod
    def _mode_check_confidence_label(cls, score: float) -> str:
        level = cls._confidence_level(score)
        return "medium" if level == "MODERATE" else level.lower()

    @classmethod
    def _estimate_mode_check_outcome(
        cls,
        *,
        goal: str,
        mode_policy: ModePolicy,
        dynamic_context: Dict[str, Any],
        tools_expected: bool,
    ) -> Dict[str, Any]:
        mode_name = normalize_chat_mode(str(mode_policy.mode_name or "smart"))
        intent = str((dynamic_context or {}).get("intent") or "general").strip().lower()
        risk_level = str((dynamic_context or {}).get("risk_level") or "low").strip().lower()
        time_sensitive = bool((dynamic_context or {}).get("time_sensitive"))

        base_scores = {
            "smart": 0.72,
            "agent": 0.70,
            "research_pro": 0.82,
        }
        score = float(base_scores.get(mode_name, 0.70))

        if tools_expected:
            score += 0.06
        if mode_policy.strict_evidence:
            score += 0.04
        if mode_policy.require_multi_source:
            score += 0.03
        if mode_policy.require_trusted_sources_only:
            score += 0.02
        if intent in {"research", "analysis"}:
            score += 0.03
        if intent in {"chat", "casual"}:
            score += 0.02
        if risk_level == "high":
            score -= 0.03 if mode_name == "research_pro" else 0.08
        if time_sensitive and not tools_expected:
            score -= 0.12
        if not str(goal or "").strip():
            score -= 0.25

        confidence_score = max(0.2, min(0.95, score))
        retries = 0
        if tools_expected:
            retries += 1
        if mode_policy.strict_evidence:
            retries += 1
        if risk_level == "high":
            retries += 1
        retries = max(0, min(int(mode_policy.max_reliability_retries), retries))

        tool_calls = 0
        if tools_expected:
            tool_calls = 1
            if intent in {"research", "analysis"}:
                tool_calls += 1
            if mode_policy.require_multi_source:
                tool_calls += 1
            if mode_policy.require_trusted_sources_only:
                tool_calls += 1
        tool_calls = max(0, min(6, int(tool_calls)))

        return {
            "confidence_score": confidence_score,
            "confidence": cls._mode_check_confidence_label(confidence_score),
            "retries": int(retries),
            "tool_calls": int(tool_calls),
            "success": bool(confidence_score >= 0.55),
            "source": "estimated",
        }

    @staticmethod
    def _estimate_mode_check_role_flow(
        *,
        mode_name: str,
        mode_policy: ModePolicy,
        dynamic_context: Dict[str, Any],
        tools_expected: bool,
    ) -> List[Dict[str, Any]]:
        intent = str((dynamic_context or {}).get("intent") or "general").strip().lower()
        risk_level = str((dynamic_context or {}).get("risk_level") or "low").strip().lower()
        steps: List[Tuple[str, str, str]] = [("P1", "planner", "REASONING")]

        if tools_expected or intent in {"research", "analysis"}:
            steps.append(("R1", "researcher", "TOOL_CALL"))

        steps.append(("E1", "executor", "EXECUTION"))

        if bool(mode_policy.strict_evidence) or risk_level == "high":
            steps.append(("C1", "critic", "VALIDATION"))

        steps.append(("FINAL", "synthesizer", "REASONING"))

        out: List[Dict[str, Any]] = []
        for node_id, declared_role, action_type in steps:
            resolved = resolve_plan_node_role(
                node_id=node_id,
                action_type=action_type,
                declared_role=declared_role,
            )
            role_key = str(resolved.role or "executor").strip().lower() or "executor"
            out.append(
                {
                    "node_id": node_id,
                    "role": role_key,
                    "mode": str(mode_name or "smart"),
                    "status": "PREDICTED",
                    "action_type": action_type,
                    "duration_ms": 0,
                    "retries": 1 if role_key == "critic" and bool(mode_policy.strict_evidence) else 0,
                    "tool_calls": 1 if role_key == "researcher" and tools_expected else 0,
                    "parallel_exception_count": 0,
                    "role_fallback_applied": bool(resolved.role_fallback_applied),
                    "role_resolution_source": str(resolved.role_resolution_source or "estimated"),
                }
            )
        return out

    @staticmethod
    def _summarize_role_metrics_for_mode_check(role_flow: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        role_metrics: Dict[str, Dict[str, Any]] = {}
        for row in list(role_flow or []):
            role = str((row or {}).get("role") or "executor").strip().lower() or "executor"
            status = str((row or {}).get("status") or "").strip().upper()
            duration_ms = int(float((row or {}).get("duration_ms") or 0))
            retries = int(float((row or {}).get("retries") or 0))
            tool_calls = int(float((row or {}).get("tool_calls") or 0))
            parallel_exception_count = int(float((row or {}).get("parallel_exception_count") or 0))
            bucket = role_metrics.setdefault(
                role,
                {
                    "runs": 0,
                    "success_count": 0,
                    "latency_ms_total": 0,
                    "retries_total": 0,
                    "tool_calls_total": 0,
                    "parallel_exception_count": 0,
                },
            )
            bucket["runs"] += 1
            if status in {"COMPLETED", "PREDICTED"}:
                bucket["success_count"] += 1
            bucket["latency_ms_total"] += max(0, duration_ms)
            bucket["retries_total"] += max(0, retries)
            bucket["tool_calls_total"] += max(0, tool_calls)
            bucket["parallel_exception_count"] += max(0, parallel_exception_count)
        return role_metrics

    def get_mode_check(self, *, input_text: str, requested_mode: str = "smart") -> Dict[str, Any]:
        goal = str(input_text or "").strip()
        requested_mode_raw = str(requested_mode or "smart").strip().lower() or "smart"
        requested_mode_normalized = normalize_chat_mode(requested_mode_raw)

        normalized_goal = self.retrieval.normalize_query(goal)
        query_type = classify_query(normalized_goal)
        current_info_query = is_current_info_query(goal)
        reliability_query_type = classify_reliability_query_type(query_type, goal)
        small_talk_only = self._is_small_talk_only(goal)

        dynamic_context = self._infer_dynamic_policy_context(
            goal=goal,
            query_type=query_type,
            reliability_query_type=reliability_query_type,
            small_talk_only=small_talk_only,
            current_info_query=current_info_query,
        )
        dynamic_context["classifier_source"] = "heuristic"
        dynamic_context["classifier_confidence"] = round(float(query_classifier_confidence(query_type, goal)), 4)

        auto_selected_mode, override_reason = self._resolve_effective_mode(
            base_mode=requested_mode_normalized,
            base_mode_raw=requested_mode_raw,
            goal=goal,
            dynamic_context=dynamic_context,
            current_info_query=current_info_query,
        )
        final_mode = auto_selected_mode
        override_applied = bool(final_mode != requested_mode_normalized or requested_mode_raw == "auto")

        base_mode_policy = self._resolve_mode_policy(final_mode)
        mode_policy, mode_policy_meta = self._resolve_effective_policy_stack(
            mode_policy=base_mode_policy,
            role_name="executor",
            dynamic_context=dynamic_context,
            goal=goal,
            user_id="global",
        )
        tools_expected = self._predict_tools_expected(
            goal=goal,
            mode_policy=mode_policy,
            small_talk_only=small_talk_only,
        )

        lane_mode = final_mode if requested_mode_raw == "auto" else requested_mode_raw
        lane = select_queue_lane(chat_mode=lane_mode, user_query=goal)
        override_reason_ui = self._map_override_reason_for_ui(override_reason, dynamic_context=dynamic_context)

        outcome = self._latest_outcome_for_mode_check(query=goal, requested_mode=requested_mode_normalized)
        estimated = self._estimate_mode_check_outcome(
            goal=goal,
            mode_policy=mode_policy,
            dynamic_context=dynamic_context,
            tools_expected=tools_expected,
        )
        confidence_score: Optional[float] = float(estimated["confidence_score"])
        confidence_label = str(estimated["confidence"])
        retries: Optional[int] = int(estimated["retries"])
        tool_calls: Optional[int] = int(estimated["tool_calls"])
        success: Optional[bool] = bool(estimated["success"])
        outcome_source = "estimated"
        role_flow: List[Dict[str, Any]] = self._estimate_mode_check_role_flow(
            mode_name=final_mode,
            mode_policy=mode_policy,
            dynamic_context=dynamic_context,
            tools_expected=tools_expected,
        )
        role_metrics: Dict[str, Any] = self._summarize_role_metrics_for_mode_check(role_flow)
        if outcome:
            outcome_source = "historical"
            confidence_score = max(0.0, min(1.0, float(outcome.get("confidence", confidence_score) or confidence_score or 0.0)))
            confidence_label = self._mode_check_confidence_label(confidence_score)
            retries = int(outcome.get("retries", retries if retries is not None else 0) or 0)
            tool_calls = int(outcome.get("tool_count", tool_calls if tool_calls is not None else 0) or 0)
            success = bool(outcome.get("success", success if success is not None else False))
            historical_flow = list(outcome.get("role_flow") or [])[-25:]
            if historical_flow:
                role_flow = historical_flow
            historical_metrics = dict(outcome.get("role_metrics") or {})
            if historical_metrics:
                role_metrics = historical_metrics
        researcher_metrics = dict((role_metrics or {}).get("researcher") or {})
        parallel_exception_count = int(researcher_metrics.get("parallel_exception_count", 0) or 0)
        researcher_runs = int(researcher_metrics.get("runs", 0) or 0)
        parallel_exception_rate = round(float(parallel_exception_count) / float(max(1, researcher_runs)), 4) if researcher_runs else 0.0

        return {
            "input": goal,
            "requested_mode": requested_mode_normalized,
            "requested_mode_raw": requested_mode_raw,
            "auto_selected_mode": auto_selected_mode,
            "final_mode": final_mode,
            "lane": lane,
            "override_applied": override_applied,
            "override_reason": override_reason_ui,
            "tools_expected": bool(tools_expected),
            "expected_latency_ms": int(mode_policy.min_reliability_budget_ms),
            "confidence": confidence_label,
            "confidence_score": confidence_score,
            "outcome_source": outcome_source,
            "retries": retries,
            "tool_calls": tool_calls,
            "parallel_exception_count": parallel_exception_count,
            "parallel_exception_rate": parallel_exception_rate,
            "success": success,
            "role_flow": role_flow,
            "role_metrics": role_metrics,
            "dynamic_context": {
                "intent": str(dynamic_context.get("intent") or "general"),
                "risk_level": str(dynamic_context.get("risk_level") or "low"),
                "time_sensitive": bool(dynamic_context.get("time_sensitive")),
            },
            "policy_snapshot": {
                "min_budget_ms": int(mode_policy.min_reliability_budget_ms),
                "max_retries": int(mode_policy.max_reliability_retries),
                "strict_evidence": bool(mode_policy.strict_evidence),
                "require_multi_source": bool(mode_policy.require_multi_source),
                "trusted_sources_only": bool(mode_policy.require_trusted_sources_only),
            },
            "adaptive_bucket_key": str((mode_policy_meta.get("adaptive") or {}).get("bucket_key") or ""),
            "adaptive_snapshot": dict((mode_policy_meta.get("adaptive") or {}).get("snapshot") or {}),
            "adaptive_rollout": {
                "rollout_applied": bool((mode_policy_meta.get("adaptive") or {}).get("rollout_applied", True)),
                "apply_ratio": float(
                    (mode_policy_meta.get("adaptive") or {}).get("apply_ratio")
                    if (mode_policy_meta.get("adaptive") or {}).get("apply_ratio") is not None
                    else (self.debug_config.get("adaptive_apply_ratio", 1.0) if self.debug_config.get("adaptive_apply_ratio", 1.0) is not None else 1.0)
                ),
                "applied": bool((mode_policy_meta.get("adaptive") or {}).get("applied", False)),
            },
            "final_clamp": dict(mode_policy_meta.get("final_clamp") or {}),
            "role_policy_meta": mode_policy_meta,
        }

    @staticmethod
    def _resolve_mode_policy(mode_name: str) -> ModePolicy:
        normalized_mode = normalize_chat_mode(str(mode_name or ""))
        merged: Dict[str, Any] = dict(_BASE_MODE_POLICY)
        merged.update(_MODE_POLICY_OVERRIDES.get(normalized_mode, {}))
        return ModePolicy(mode_name=normalized_mode, **merged)

    @staticmethod
    def _resolve_role_policy(role_name: str) -> Tuple[Dict[str, Any], bool]:
        role_key = str(role_name or "").strip().lower()
        role_policy = ROLE_POLICIES.get(role_key)
        if role_policy is None:
            return dict(ROLE_POLICIES.get("executor", {})), True
        return dict(role_policy), False

    @classmethod
    def _apply_role_policy(cls, policy: ModePolicy, role_name: str) -> Tuple[ModePolicy, bool]:
        role_policy, role_fallback_applied = cls._resolve_role_policy(role_name)
        filtered = {
            k: v
            for k, v in role_policy.items()
            if k in _MODE_POLICY_FIELD_NAMES and k != "mode_name"
        }
        if not filtered:
            return policy, role_fallback_applied
        return replace(policy, **filtered), role_fallback_applied

    @staticmethod
    def _apply_hard_constraint_guard(*, mode_policy: ModePolicy, candidate_policy: ModePolicy) -> ModePolicy:
        updates: Dict[str, Any] = {}

        if bool(mode_policy.live_query_forces_tools):
            updates["live_query_forces_tools"] = True
        if bool(mode_policy.tool_force_for_non_smalltalk):
            updates["tool_force_for_non_smalltalk"] = True

        if bool(mode_policy.strict_evidence):
            updates["strict_evidence"] = True
        if bool(mode_policy.require_multi_source):
            updates["require_multi_source"] = True
        if bool(mode_policy.require_trusted_sources_only):
            updates["require_trusted_sources_only"] = True

        updates["min_reliability_budget_ms"] = max(
            int(candidate_policy.min_reliability_budget_ms),
            int(mode_policy.min_reliability_budget_ms),
        )
        updates["max_reliability_retries"] = max(
            int(candidate_policy.max_reliability_retries),
            int(mode_policy.max_reliability_retries),
        )

        return replace(candidate_policy, **updates)

    def _resolve_effective_policy_stack(
        self,
        *,
        mode_policy: ModePolicy,
        role_name: str,
        dynamic_context: Dict[str, Any],
        goal: str,
        user_id: str,
    ) -> Tuple[ModePolicy, Dict[str, Any]]:
        baseline_mode_policy = mode_policy
        role_applied_policy, role_fallback_applied = self._apply_role_policy(mode_policy, role_name)
        dynamic_policy = self._apply_dynamic_overrides(role_applied_policy, dynamic_context)
        adaptive_policy, adaptive_meta = self._apply_adaptive_learning_overrides(
            policy=dynamic_policy,
            dynamic_context=dynamic_context,
            goal=goal,
            user_id=user_id,
        )
        tuned_policy = self._apply_auto_tuning_overrides(adaptive_policy)
        hard_guard_policy = self._apply_hard_constraint_guard(
            mode_policy=baseline_mode_policy,
            candidate_policy=tuned_policy,
        )
        final_policy, final_clamp = self._final_safety_clamp(
            mode_policy=baseline_mode_policy,
            candidate_policy=hard_guard_policy,
        )
        role_key = str(role_name or "").strip().lower() or "executor"
        return final_policy, {
            "role": role_key,
            "role_fallback_applied": bool(role_fallback_applied),
            "adaptive": adaptive_meta,
            "final_clamp": final_clamp,
            "merge_order": [
                "mode_policy",
                "role_policy",
                "dynamic_overrides",
                "adaptive_learning",
                "auto_tuning",
                "hard_constraint_guard",
                "final_safety_clamp",
            ],
        }

    @staticmethod
    def _infer_dynamic_policy_context(
        *,
        goal: str,
        query_type: str,
        reliability_query_type: str,
        small_talk_only: bool,
        current_info_query: bool,
    ) -> Dict[str, Any]:
        q = str(goal or "").strip().lower()
        if small_talk_only or reliability_query_type in {"casual", "creative"}:
            intent = "chat"
        elif query_type == "research" or reliability_query_type in {"research", "current_info", "comparison", "factual"}:
            intent = "research"
        elif any(k in q for k in ("build", "implement", "debug", "fix", "execute", "run", "automate")):
            intent = "task"
        else:
            intent = "general"

        high_risk_markers = {
            "today", "latest", "recent", "new data", "updated", "breaking", "current", "real-time", "earnings", "stock", "price",
            "medical", "health", "legal", "compliance", "election", "war", "conflict", "attack", "sanction",
            "safety", "security", "critical", "urgent", "official",
        }
        medium_risk_markers = {
            "analysis", "compare", "forecast", "market", "sources", "citation", "verify",
        }
        risk_level = "low"
        if current_info_query or any(k in q for k in high_risk_markers):
            risk_level = "high"
        elif intent == "research" or any(k in q for k in medium_risk_markers):
            risk_level = "medium"

        critical_research = bool(intent == "research" and risk_level == "high")
        return {
            "intent": intent,
            "risk_level": risk_level,
            "time_sensitive": bool(current_info_query),
            "critical_research": critical_research,
        }

    @staticmethod
    def _has_critical_research_trigger(goal: str, *, current_info_query: bool, dynamic_context: Dict[str, Any]) -> bool:
        q = str(goal or "").strip().lower()
        if current_info_query:
            return True
        if bool((dynamic_context or {}).get("critical_research")):
            return True
        if len(q) > 120:
            return True
        if re.search(r"\b(latest|today|current|breaking|recent|new data|news|update|updates|updated|with sources|cite|citation|proof|evidence)\b", q):
            return True
        if re.search(r"\b20\d{2}\b", q):
            return True
        return False

    @staticmethod
    def _has_task_execution_trigger(goal: str) -> bool:
        q = str(goal or "").strip().lower()
        return bool(re.search(r"\b(build|generate|analyze|fix|debug|implement|create|code|refactor|optimize|execute)\b", q))

    @classmethod
    def _resolve_effective_mode(
        cls,
        *,
        base_mode: str,
        base_mode_raw: Optional[str],
        goal: str,
        dynamic_context: Dict[str, Any],
        current_info_query: bool,
    ) -> Tuple[str, Optional[str]]:
        raw_base = str(base_mode_raw or "").strip().lower()
        normalized_base = normalize_chat_mode(str(base_mode or ""))
        intent = str((dynamic_context or {}).get("intent") or "").strip().lower()
        if raw_base == "auto":
            if cls._has_critical_research_trigger(goal, current_info_query=current_info_query, dynamic_context=dynamic_context):
                return "research_pro", "auto_intent_research"
            if intent == "research":
                return "research_pro", "auto_intent_research"
            if intent == "task" or cls._has_task_execution_trigger(goal):
                return "agent", "auto_intent_task"
            return "smart", "auto_intent_chat"

        if normalized_base in {"agent", "research_pro"}:
            return normalized_base, None

        if cls._has_critical_research_trigger(goal, current_info_query=current_info_query, dynamic_context=dynamic_context):
            return "research_pro", "critical_research_trigger"

        if intent == "task" or cls._has_task_execution_trigger(goal):
            return "agent", "task_execution_trigger"

        return "smart", None

    @staticmethod
    def _apply_dynamic_overrides(policy: ModePolicy, context: Dict[str, Any]) -> ModePolicy:
        intent = str((context or {}).get("intent") or "general").lower()
        risk = str((context or {}).get("risk_level") or "low").lower()
        time_sensitive = bool((context or {}).get("time_sensitive"))
        critical_research = bool((context or {}).get("critical_research"))

        if critical_research:
            return replace(
                policy,
                memory_read_enabled=True,
                memory_write_enabled=True,
                min_reliability_budget_ms=max(int(policy.min_reliability_budget_ms), 45000),
                reasoning_hints_enabled=False,
                tool_force_for_non_smalltalk=True,
                strict_evidence=True,
                require_multi_source=True,
                require_trusted_sources_only=True,
                evidence_softpass_with_existing_sources=False,
                trusted_source_softpass_with_existing_sources=False,
                critic_repair_requires_missing_sources=False,
                max_reliability_retries=max(int(policy.max_reliability_retries), 3),
            )

        if intent == "research" and (time_sensitive or risk == "high"):
            return replace(
                policy,
                memory_read_enabled=True,
                min_reliability_budget_ms=max(int(policy.min_reliability_budget_ms), 35000),
                tool_force_for_non_smalltalk=True,
                strict_evidence=True,
                require_multi_source=True,
                evidence_softpass_with_existing_sources=False,
                trusted_source_softpass_with_existing_sources=False,
            )

        if intent == "research" and risk == "medium":
            return replace(
                policy,
                tool_force_for_non_smalltalk=True,
                require_multi_source=True,
                max_reliability_retries=max(int(policy.max_reliability_retries), 2),
            )

        return policy

    @staticmethod
    def _context_cache_key(goal: str, query_type: str, reliability_query_type: str) -> str:
        return f"{str(query_type or '').strip().lower()}::{str(reliability_query_type or '').strip().lower()}::{str(goal or '').strip().lower()[:260]}"

    @staticmethod
    def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
        text = str(raw or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    async def _infer_dynamic_policy_context_llm(
        self,
        *,
        goal: str,
        query_type: str,
        reliability_query_type: str,
        small_talk_only: bool,
        current_info_query: bool,
    ) -> Optional[Dict[str, Any]]:
        if not FF_LLM_INTENT_CLASSIFIER:
            return None
        cache_key = self._context_cache_key(goal, query_type, reliability_query_type)
        cached = self.intent_context_cache.get(cache_key)
        if cached:
            return dict(cached)

        if small_talk_only and len(str(goal or "").split()) <= 8:
            out = {
                "intent": "chat",
                "risk_level": "low",
                "time_sensitive": False,
                "critical_research": False,
                "classifier_confidence": 0.9,
                "classifier_source": "llm",
            }
            self.intent_context_cache[cache_key] = dict(out)
            return out

        prompt = (
            "Classify user request into policy fields for tool/research routing.\n"
            "Return ONLY JSON with keys:\n"
            "{\"intent\":\"chat|research|task|general\",\"risk_level\":\"low|medium|high\","
            "\"time_sensitive\":true|false,\"critical_research\":true|false,\"confidence\":0.0}\n"
            "Rules:\n"
            "- intent=research for factual/current-events/comparative verification asks.\n"
            "- risk_level=high for time-sensitive, conflict, legal, medical, financial, safety topics.\n"
            "- critical_research=true only when strong evidence quality and recency are essential.\n"
            "- Be conservative and practical."
        )
        payload = {
            "goal": str(goal or "")[:1200],
            "query_type": str(query_type or ""),
            "reliability_query_type": str(reliability_query_type or ""),
            "small_talk_only": bool(small_talk_only),
            "current_info_query": bool(current_info_query),
        }

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_to_use,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    temperature=0.0,
                    stream=False,
                    max_tokens=120,
                ),
                timeout=3.5,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "")
            parsed = self._safe_parse_json(content)
            if not parsed:
                return None
            intent = str(parsed.get("intent") or "").strip().lower()
            risk = str(parsed.get("risk_level") or "").strip().lower()
            confidence = float(parsed.get("confidence") or 0.0)
            if intent not in {"chat", "research", "task", "general"}:
                return None
            if risk not in {"low", "medium", "high"}:
                return None
            if confidence < 0.55:
                return None
            out = {
                "intent": intent,
                "risk_level": risk,
                "time_sensitive": bool(parsed.get("time_sensitive")),
                "critical_research": bool(parsed.get("critical_research")),
                "classifier_confidence": max(0.0, min(1.0, confidence)),
                "classifier_source": "llm",
            }
            self.intent_context_cache[cache_key] = dict(out)
            return out
        except Exception:
            return None

    @staticmethod
    def _query_terms_for_relevance(value: str) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
            if len(token) < 3:
                continue
            if token in {"latest", "recent", "today", "current", "news", "update", "updates", "research", "reserch"}:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= 12:
                break
        return out

    @classmethod
    def _score_tool_relevance(cls, *, goal: str, tool_name: str, node_result: Any) -> float:
        terms = cls._query_terms_for_relevance(goal)
        if not terms:
            return 0.65
        raw = str(node_result or "").lower()
        if not raw.strip():
            return 0.0
        hits = sum(1 for t in terms if t in raw)
        base = float(hits) / float(max(1, len(terms)))
        name = str(tool_name or "").strip().lower()
        if name == "summarize_url":
            base = min(1.0, base + 0.10)
        return max(0.0, min(1.0, base))

    def _record_tool_confidence_from_graph(self, *, graph: Any, goal: str) -> Dict[str, Any]:
        if not FF_TOOL_CONFIDENCE_SCORING:
            return {"enabled": False, "updated_tools": 0}
        runtime_stats: List[Dict[str, Any]] = list((getattr(graph, "metadata", {}) or {}).get("tool_runtime_stats") or [])
        runtime_map: Dict[str, Dict[str, Any]] = {}
        for row in runtime_stats:
            if not isinstance(row, dict):
                continue
            tool_name = str(row.get("tool") or "").strip().lower()
            if not tool_name:
                continue
            bucket = runtime_map.setdefault(
                tool_name,
                {"latency_ms_sum": 0.0, "latency_count": 0, "hard_error": False},
            )
            lat = float(row.get("latency_ms", 0) or 0)
            if lat > 0:
                bucket["latency_ms_sum"] += lat
                bucket["latency_count"] += 1
            if bool(row.get("hard_error")):
                bucket["hard_error"] = True

        updated = 0
        for node in getattr(graph, "nodes", {}).values():
            if str(getattr(node, "action_type", "")).upper() != "TOOL_CALL":
                continue
            tool_name = str((getattr(node, "payload", {}) or {}).get("tool") or "").strip().lower()
            if not tool_name:
                continue
            status = str(getattr(node, "status", "")).strip().upper()
            success = status == "COMPLETED"
            relevance = self._score_tool_relevance(goal=goal, tool_name=tool_name, node_result=getattr(node, "result", None))
            runtime_bucket = dict(runtime_map.get(tool_name) or {})
            latency_count = int(runtime_bucket.get("latency_count", 0) or 0)
            latency_ms = (
                float(runtime_bucket.get("latency_ms_sum", 0.0) or 0.0) / float(max(1, latency_count))
                if latency_count > 0
                else None
            )
            hard_error = bool(runtime_bucket.get("hard_error", False))
            self.tool_confidence_store.record(
                tool_name=tool_name,
                success=success,
                relevance=relevance,
                latency_ms=latency_ms,
                hard_error=hard_error,
            )
            updated += 1

        top_scores: List[Dict[str, Any]] = []
        for rec in sorted(self.tool_confidence_store.snapshot(), key=lambda r: r.selection_score(), reverse=True)[:10]:
            top_scores.append(
                {
                    "tool": rec.tool_name,
                    "score": round(float(rec.selection_score()), 4),
                    "samples": int(rec.samples),
                    "success_rate": round(float(rec.success_rate()), 4),
                    "error_rate": round(float(rec.error_rate()), 4),
                    "latency_ema_ms": round(float(rec.latency_ema_ms), 2),
                    "avg_relevance": round(float(rec.avg_relevance()), 4),
                }
            )
        return {"enabled": True, "updated_tools": updated, "top_scores": top_scores}

    @staticmethod
    def _collect_role_telemetry_from_graph(
        *,
        graph: Any,
        mode_name: str,
    ) -> Dict[str, Any]:
        role_flow: List[Dict[str, Any]] = []
        role_metrics: Dict[str, Dict[str, Any]] = {}
        role_summary = {
            "fallback_count": 0,
            "final_synthesizer_ok": True,
            "parallel_batches_total": 0,
            "parallel_tools_total": 0,
            "parallel_success_total": 0,
            "parallel_exception_count_total": 0,
        }

        metadata = dict(getattr(graph, "metadata", {}) or {})
        metadata_flow = list(metadata.get("role_flow") or [])
        role_summary["parallel_batches_total"] = int(metadata.get("parallel_batches_total", 0) or 0)
        role_summary["parallel_tools_total"] = int(metadata.get("parallel_tools_total", 0) or 0)
        role_summary["parallel_success_total"] = int(metadata.get("parallel_success_total", 0) or 0)
        role_summary["parallel_exception_count_total"] = int(metadata.get("parallel_exception_count_total", 0) or 0)

        if metadata_flow:
            for row in metadata_flow:
                if not isinstance(row, dict):
                    continue
                role = str(row.get("role") or "executor").strip().lower() or "executor"
                status = str(row.get("status") or "UNKNOWN").strip().upper() or "UNKNOWN"
                duration_ms = int(float(row.get("duration_ms") or 0))
                retries = int(float(row.get("retries") or 0))
                tool_calls = int(float(row.get("tool_calls") or 0))
                parallel_exception_count = int(float(row.get("parallel_exception_count") or 0))
                fallback_applied = bool(row.get("role_fallback_applied"))
                node_id = str(row.get("node_id") or "")

                item = {
                    "node_id": node_id,
                    "role": role,
                    "mode": str(mode_name or "smart"),
                    "status": status,
                    "action_type": str(row.get("action_type") or ""),
                    "duration_ms": duration_ms,
                    "retries": retries,
                    "tool_calls": tool_calls,
                    "parallel_exception_count": parallel_exception_count,
                    "role_fallback_applied": fallback_applied,
                    "role_resolution_source": str(row.get("role_resolution_source") or "unknown"),
                }
                role_flow.append(item)
                if fallback_applied:
                    role_summary["fallback_count"] += 1
                if node_id.upper() == "FINAL" and role != "synthesizer":
                    role_summary["final_synthesizer_ok"] = False

                bucket = role_metrics.setdefault(
                    role,
                    {
                        "runs": 0,
                        "success_count": 0,
                        "latency_ms_total": 0,
                        "retries_total": 0,
                        "tool_calls_total": 0,
                        "parallel_exception_count": 0,
                    },
                )
                bucket["runs"] += 1
                if status == "COMPLETED":
                    bucket["success_count"] += 1
                bucket["latency_ms_total"] += max(0, duration_ms)
                bucket["retries_total"] += max(0, retries)
                bucket["tool_calls_total"] += max(0, tool_calls)
                bucket["parallel_exception_count"] += max(0, parallel_exception_count)

        if not role_flow:
            for node_id, node in getattr(graph, "nodes", {}).items():
                role = str(getattr(node, "role", "executor") or "executor").strip().lower() or "executor"
                status = str(getattr(node, "status", "UNKNOWN") or "UNKNOWN").strip().upper() or "UNKNOWN"
                fallback_applied = bool(getattr(node, "role_fallback_applied", False))
                role_resolution_source = str(getattr(node, "role_resolution_source", "unknown") or "unknown")
                action_type = str(getattr(node, "action_type", "") or "")
                tool_calls = 1 if action_type.upper() == "TOOL_CALL" else 0
                parallel_exception_count = int(getattr(node, "parallel_exception_count", 0) or 0)

                role_flow.append(
                    {
                        "node_id": str(node_id),
                        "role": role,
                        "mode": str(mode_name or "smart"),
                        "status": status,
                        "action_type": action_type,
                        "duration_ms": 0,
                        "retries": 0,
                        "tool_calls": tool_calls,
                        "parallel_exception_count": parallel_exception_count,
                        "role_fallback_applied": fallback_applied,
                        "role_resolution_source": role_resolution_source,
                    }
                )
                if fallback_applied:
                    role_summary["fallback_count"] += 1
                if str(node_id).upper() == "FINAL" and role != "synthesizer":
                    role_summary["final_synthesizer_ok"] = False

                bucket = role_metrics.setdefault(
                    role,
                    {
                        "runs": 0,
                        "success_count": 0,
                        "latency_ms_total": 0,
                        "retries_total": 0,
                        "tool_calls_total": 0,
                        "parallel_exception_count": 0,
                    },
                )
                bucket["runs"] += 1
                if status == "COMPLETED":
                    bucket["success_count"] += 1
                bucket["tool_calls_total"] += tool_calls
                bucket["parallel_exception_count"] += max(0, parallel_exception_count)

        batches = int(role_summary.get("parallel_batches_total", 0) or 0)
        exceptions_total = int(role_summary.get("parallel_exception_count_total", 0) or 0)
        role_summary["parallel_exception_rate"] = round(float(exceptions_total) / float(max(1, batches)), 4) if batches else 0.0

        return {
            "role_flow": role_flow[-120:],
            "role_metrics": role_metrics,
            "role_summary": role_summary,
        }

    def _build_tool_confidence_guidance(self) -> str:
        if not FF_TOOL_CONFIDENCE_SCORING:
            return ""
        records = self.tool_confidence_store.snapshot()
        if not records:
            return ""
        eligible = [r for r in records if int(r.samples) >= int(TOOL_CONFIDENCE_MIN_SAMPLES)]
        if not eligible:
            return ""
        retrieval = [r for r in eligible if r.tool_name in {"search_web", "search_news", "search_scholar", "summarize_url"}]
        if not retrieval:
            return ""
        retrieval.sort(key=lambda r: r.selection_score(), reverse=True)
        best = retrieval[0]
        weak = [r for r in retrieval if r.selection_score() < 0.42]
        lines = [
            f"Tool reliability signal: prefer {best.tool_name} (score={best.selection_score():.2f}, samples={best.samples})."
        ]
        if weak:
            weak_str = ", ".join(f"{r.tool_name}({r.selection_score():.2f})" for r in weak[:2])
            lines.append(f"Avoid overusing lower-performing tools unless needed: {weak_str}.")
        return " ".join(lines)

    @staticmethod
    def _is_time_sensitive_query(goal: str) -> bool:
        q = (goal or "").lower()
        return any(
            token in q
            for token in [
                "today", "latest", "current", "now", "breaking", "this week", "this month",
                "weather", "stock", "price", "news", "score", "result",
            ]
        )

    @staticmethod
    def _force_tool_route(goal: str, query_type: str) -> bool:
        q = (goal or "").lower()
        intent_markers = [
            "research", "reserch", "recent", "latest", "current", "today", "breaking",
            "news", "verify", "source", "citation", "fact check", "timeline",
        ]
        if any(marker in q for marker in intent_markers):
            return True
        if query_type in {"research", "comparison", "current_info"}:
            return True
        return False

    @staticmethod
    def _is_hybrid_live_query(goal: str) -> bool:
        q = (goal or "").lower()
        markers = [
            "news", "recent", "latest", "current", "today", "breaking", "update",
            "war", "conflict", "attack", "ceasefire", "sanction", "election",
            "price", "stock", "market", "score", "result",
            "india", "pakistan", "iran", "israel", "russia", "ukraine",
        ]
        return any(marker in q for marker in markers)

    @staticmethod
    def _is_small_talk_only(goal: str) -> bool:
        q = " ".join(str(goal or "").strip().lower().split())
        if not q:
            return True
        live_markers = [
            "news", "recent", "latest", "current", "today", "breaking",
            "research", "reserch", "search", "verify", "source", "citation",
            "price", "stock", "market", "weather", "score", "result",
        ]
        if any(marker in q for marker in live_markers):
            return False
        greeting_patterns = [
            r"^(hi+|hey+|hello+|yo+|sup+|hola+)(\s+(bro|macha|dude|buddy))?\b[!.?,\s]*$",
            r"^(how are you|how are u)\b[!.?,\s]*$",
            r"^(thanks|thank you)\b[!.?,\s]*$",
            r"^(good)\s+(morning|evening|night)\b[!.?,\s]*$",
            r"^(ok|okay|cool|nice|great|awesome)\b[!.?,\s]*$",
        ]
        if len(q.split()) <= 12 and any(re.match(pattern, q) for pattern in greeting_patterns):
            return True
        if len(q.split()) <= 4:
            casual_tokens = {"hi", "hii", "hey", "heyy", "heyyy", "hello", "yo", "sup", "hola", "bro", "macha", "buddy"}
            tokens = re.findall(r"[a-z]+", q)
            if tokens and all(t in casual_tokens for t in tokens):
                return True
        return False

    def _direct_answer_confidence(self, goal: str, query_type: str) -> float:
        q = (goal or "").strip().lower()
        # Only treat greeting/etiquette as direct-answer safe when the message is
        # actually short and conversational (not mixed with factual/news asks).
        greeting_only_patterns = [
            r"^\s*(hi+|hey+|hello+|yo+|sup+)(\s+(bro|macha|dude|buddy))?\b[!.?,\s]*$",
            r"^\s*(how are you|how are u)\b[!.?,\s]*$",
            r"^\s*(thanks|thank you)\b[!.?,\s]*$",
            r"^\s*good\s+(morning|evening|night)\b[!.?,\s]*$",
        ]
        if len(q.split()) <= 6 and any(re.match(pattern, q) for pattern in greeting_only_patterns):
            return 0.92
        if query_type not in {"simple_fact"}:
            if query_type == "general" and len(q.split()) <= 8 and not self._is_time_sensitive_query(q):
                return 0.86
            return 0.0
        if self._is_time_sensitive_query(q):
            return 0.0
        if len(q.split()) > 20:
            return 0.0
        if any(k in q for k in ["compare", "research", "analyze", "verify", "citation", "source"]):
            return 0.0
        return 0.88

    @staticmethod
    def _is_capability_query(goal: str) -> bool:
        q = (goal or "").strip().lower()
        markers = [
            "what can you do",
            "what are the tasks you can do",
            "what are the tasks u can do",
            "your capabilities",
            "features can you do",
        ]
        return any(m in q for m in markers)

    @staticmethod
    def _capability_response() -> str:
        return (
            "I am Relyce AI. I can help with coding, debugging, UI/frontend implementation, "
            "business analysis, research with sources, document understanding, and step-by-step problem solving."
        )

    async def _try_confidence_gate(self, goal: str, query_type: str, min_threshold: float = 0.85) -> Optional[Tuple[Optional[str], float]]:
        if self._is_capability_query(goal):
            return self._capability_response(), 0.95
        if self._force_tool_route(goal, query_type):
            # Never shortcut to direct synthesis for research/current-info requests.
            return None
        conf = self._direct_answer_confidence(goal, query_type)
        if conf < min_threshold:
            return None
        return None, conf

    @staticmethod
    def _compress_workspace(state: AgentState) -> None:
        ws = state.workspace

        ws.knowledge = list(dict.fromkeys([str(k).strip() for k in ws.knowledge if str(k).strip()]))[-12:]
        ws.reasoning_steps = list(dict.fromkeys([str(r).strip() for r in ws.reasoning_steps if str(r).strip()]))[-10:]
        ws.claims = ws.claims[-18:]

        by_url: Dict[str, Dict[str, Any]] = {}
        for src in ws.sources:
            url = str(src.get("url", "")).strip()
            if not url:
                continue
            existing = by_url.get(url)
            if not existing or float(src.get("trust_score", 0.0)) >= float(existing.get("trust_score", 0.0)):
                by_url[url] = src
        ws.sources = list(by_url.values())[-12:]

        fact_seen: Set[str] = set()
        compressed_facts: List[Dict[str, Any]] = []
        for fact in reversed(ws.facts):
            sig = str(fact.get("value", "")).strip().lower()[:180]
            if not sig or sig in fact_seen:
                continue
            fact_seen.add(sig)
            compressed_facts.append(fact)
            if len(compressed_facts) >= 18:
                break
        ws.facts = list(reversed(compressed_facts))

        top_facts = [str(f.get("value", "")).strip() for f in ws.facts[-3:] if str(f.get("value", "")).strip()]
        top_sources = [str(s.get("url", "")).strip() for s in ws.sources[-3:] if str(s.get("url", "")).strip()]
        ws.progress_summary = (
            f"Facts: {len(ws.facts)} | Sources: {len(ws.sources)} | "
            f"Top facts: {top_facts} | Top sources: {top_sources}"
        )[:800]

    @staticmethod
    def _cache_key_for_goal(goal: str) -> str:
        return "Q::" + hashlib.sha1(goal.strip().lower().encode("utf-8")).hexdigest()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(str(text or "").split()) // 1)

    def _apply_context_budget(
        self,
        *,
        context_messages: List[Dict[str, Any]],
        max_tokens: int = 4000,
        evidence_budget: int = 1200,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Conservative distillation:
        # never touch newest user message/system instructions. Trim oldest turns first.
        msgs = list(context_messages or [])
        if not msgs:
            return msgs, {
                "input_token_count": 0,
                "compression_applied": False,
                "compression_ratio": 1.0,
            }

        token_counts = [self._estimate_tokens(m.get("content", "")) for m in msgs]
        original_tokens = sum(token_counts)
        if original_tokens <= max_tokens:
            return msgs, {
                "input_token_count": original_tokens,
                "compression_applied": False,
                "compression_ratio": 1.0,
            }

        # Preserve last 6 turns and all system prompts.
        system_msgs = [m for m in msgs if str(m.get("role", "")).lower() == "system"]
        non_system = [m for m in msgs if str(m.get("role", "")).lower() != "system"]
        keep_tail = non_system[-6:]
        head = non_system[:-6]

        # Distill older context into one compact summary.
        distilled_lines: List[str] = []
        used = 0
        for m in head[::-1]:
            if used >= evidence_budget:
                break
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            snippet = " ".join(content.split())[:220]
            distilled_lines.append(f"- {m.get('role', 'user')}: {snippet}")
            used += self._estimate_tokens(snippet)
        distilled_lines.reverse()

        distilled_msg = {
            "role": "system",
            "content": "Conversation summary (distilled, older context):\n" + "\n".join(distilled_lines[:40]),
        } if distilled_lines else None

        out: List[Dict[str, Any]] = []
        out.extend(system_msgs)
        if distilled_msg:
            out.append(distilled_msg)
        out.extend(keep_tail)

        new_tokens = sum(self._estimate_tokens(m.get("content", "")) for m in out)
        ratio = float(new_tokens) / float(max(1, original_tokens))
        return out, {
            "input_token_count": int(new_tokens),
            "compression_applied": True,
            "compression_ratio": round(ratio, 4),
            "input_token_count_original": int(original_tokens),
        }

    @staticmethod
    def _extract_sources_from_result(result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        extracted = []
        def _coerce_trust(value: Any, fallback_url: str = "") -> float:
            try:
                return float(value)
            except Exception:
                return float(score_source(fallback_url) if fallback_url else 0.5)
        raw = result_dict.get("result")
        if isinstance(raw, dict):
            sources = raw.get("sources")
            if isinstance(sources, list):
                for s in sources:
                    if isinstance(s, dict) and s.get("url"):
                        extracted.append(s)
            elif raw.get("source"):
                extracted.append(
                    {
                        "url": raw.get("source"),
                        "domain": raw.get("domain", "unknown"),
                        "trust": _coerce_trust(raw.get("trust", 0.5), str(raw.get("source") or "")),
                        "snippet": str(raw.get("content", ""))[:300],
                        "timestamp": raw.get("timestamp") or raw.get("published_at") or raw.get("date") or "",
                    }
                )
            elif raw.get("url"):
                extracted.append(
                    {
                        "url": raw.get("url"),
                        "domain": raw.get("domain", "unknown"),
                        "trust": _coerce_trust(raw.get("trust", 0.5), str(raw.get("url") or "")),
                        "snippet": str(raw.get("summary") or raw.get("content") or raw.get("title") or "")[:300],
                        "timestamp": raw.get("timestamp") or raw.get("published_at") or raw.get("date") or "",
                    }
                )
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or item.get("link") or "").strip()
                if not url:
                    continue
                extracted.append(
                    {
                        "url": url,
                        "domain": item.get("domain", "unknown"),
                        "trust": item.get("trust", score_source(url)),
                        "snippet": str(item.get("snippet") or item.get("title") or "")[:300],
                        "timestamp": item.get("timestamp") or item.get("published_at") or item.get("date") or "",
                    }
                )

        source_value = str(result_dict.get("source") or "").strip()
        if source_value.startswith(("http://", "https://")) and not extracted:
            extracted.append(
                {
                    "url": source_value,
                    "domain": "unknown",
                    "trust": score_source(source_value),
                    "snippet": str(raw)[:300],
                    "timestamp": "",
                }
            )
        return extracted

    @staticmethod
    def _safe_trust_score(value: Any, fallback: float = 0.5) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        normalized = str(value or "").strip().lower()
        if normalized in {"verified", "high"}:
            return 0.9
        if normalized in {"moderate", "medium"}:
            return 0.7
        if normalized in {"low", "unverified"}:
            return 0.3
        try:
            return float(normalized)
        except Exception:
            return float(fallback)

    @staticmethod
    def _normalize_claim_key(value: str) -> str:
        text = " ".join(str(value or "").strip().lower().split())
        return text[:180]

    @classmethod
    def _lookup_supporting_sources(cls, claim_text: str, claim_sources: Dict[str, List[str]]) -> List[str]:
        if not claim_sources:
            return []
        key = cls._normalize_claim_key(claim_text)
        if key in claim_sources:
            return [str(u).strip() for u in (claim_sources.get(key) or []) if str(u).strip()]
        prefix = key[:80]
        for raw_key, urls in claim_sources.items():
            rk = cls._normalize_claim_key(str(raw_key))
            if not rk:
                continue
            if rk.startswith(prefix) or key.startswith(rk):
                return [str(u).strip() for u in (urls or []) if str(u).strip()]
        return []

    @staticmethod
    def _provider_from_url(url: str) -> str:
        try:
            host = (urlparse(str(url or "").strip()).netloc or "").lower()
            if host.startswith("www."):
                host = host[4:]
            parts = [p for p in host.split(".") if p]
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return host or "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _try_parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw).astimezone(timezone.utc)
        except Exception:
            pass
        # Common date snippets in search result metadata.
        for pattern in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(raw[:32], pattern).replace(tzinfo=timezone.utc)
            except Exception:
                continue
        m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", raw)
        if m:
            try:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return datetime(year, month, day, tzinfo=timezone.utc)
            except Exception:
                return None
        return None

    @classmethod
    def _extract_source_datetime(cls, source: Dict[str, Any]) -> Optional[datetime]:
        for key in ("timestamp", "published_at", "published", "date"):
            parsed = cls._try_parse_datetime(source.get(key))
            if parsed is not None:
                return parsed
        snippet = str(source.get("snippet") or "")
        return cls._try_parse_datetime(snippet)

    @classmethod
    def _compute_citation_coverage(
        cls,
        verification_result: Dict[str, Any],
        claim_sources: Dict[str, List[str]],
        workspace_sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        known_urls = {
            str(src.get("url") or "").strip()
            for src in (workspace_sources or [])
            if str(src.get("url") or "").strip()
        }
        claims = list((verification_result or {}).get("verified_claims") or [])
        if not claims:
            claims = list((verification_result or {}).get("uncertain_claims") or [])
        if not claims:
            return {"coverage": 1.0, "total_claims": 0, "covered_claims": 0, "missing_claims": []}

        covered = 0
        missing_claims: List[str] = []
        for claim in claims:
            text = str((claim or {}).get("value") or (claim or {}).get("claim") or "").strip()
            if not text:
                continue
            support_urls = cls._lookup_supporting_sources(text, claim_sources)
            support_urls = [u for u in support_urls if u in known_urls]
            if support_urls:
                covered += 1
            else:
                missing_claims.append(text[:180])
        total = max(1, len([c for c in claims if str((c or {}).get("value") or (c or {}).get("claim") or "").strip()]))
        coverage = round(float(covered) / float(total), 4)
        return {
            "coverage": coverage,
            "total_claims": total,
            "covered_claims": covered,
            "missing_claims": missing_claims[:5],
        }

    @classmethod
    def _compute_freshness_metrics(cls, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        scored: List[float] = []
        recent_30 = 0
        dated_sources = 0
        for src in sources or []:
            dt = cls._extract_source_datetime(src)
            if dt is None:
                continue
            dated_sources += 1
            age_days = max(0, int((now - dt).total_seconds() // 86400))
            if age_days <= 7:
                score = 1.0
                recent_30 += 1
            elif age_days <= 30:
                score = 0.85
                recent_30 += 1
            elif age_days <= 90:
                score = 0.70
            elif age_days <= 180:
                score = 0.55
            elif age_days <= 365:
                score = 0.35
            else:
                score = 0.20
            scored.append(score)

        freshness = round(sum(scored) / float(len(scored)), 4) if scored else 0.0
        return {
            "freshness_score": freshness,
            "dated_sources": dated_sources,
            "recent_30d_sources": recent_30,
            "has_fresh_source": recent_30 > 0,
        }

    @staticmethod
    def _compute_source_mix(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        trusted_urls: Set[str] = set()
        secondary_urls: Set[str] = set()
        providers: Set[str] = set()
        for src in sources or []:
            url = str(src.get("url") or "").strip()
            if not url:
                continue
            trust = float(src.get("trust_score", src.get("trust", 0.0)) or 0.0)
            if trust >= TRUSTED_SOURCE_THRESHOLD:
                trusted_urls.add(url)
            elif trust >= SOURCE_MIX_SECONDARY_MIN:
                secondary_urls.add(url)
            provider = TaskManager._provider_from_url(url)
            if provider and provider != "unknown":
                providers.add(provider)
        return {
            "trusted_sources": len(trusted_urls),
            "secondary_sources": len(secondary_urls),
            "provider_diversity": len(providers),
            "passes_mix": bool(trusted_urls) and bool(secondary_urls),
        }

    @staticmethod
    def _compute_contradiction_metrics(verification_result: Dict[str, Any]) -> Dict[str, Any]:
        verified = len((verification_result or {}).get("verified_claims") or [])
        uncertain = len((verification_result or {}).get("uncertain_claims") or [])
        conflicted = len((verification_result or {}).get("conflicted_claims") or [])
        total = max(1, verified + uncertain + conflicted)
        ratio = round(float(conflicted) / float(total), 4)
        return {
            "verified_claims": verified,
            "uncertain_claims": uncertain,
            "conflicted_claims": conflicted,
            "conflict_ratio": ratio,
            "passes_conflict_gate": conflicted == 0,
        }

    def _evaluate_research_policy(
        self,
        *,
        goal: str,
        query_type: str,
        mode_policy: Optional[ModePolicy],
        verification_result: Dict[str, Any],
        claim_sources: Dict[str, List[str]],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        policy = mode_policy or self._resolve_mode_policy("agent")
        current_info = bool(is_current_info_query(goal) or query_type in {"current_info"})
        citation = self._compute_citation_coverage(verification_result, claim_sources, sources)
        freshness = self._compute_freshness_metrics(sources)
        contradiction = self._compute_contradiction_metrics(verification_result)
        source_mix = self._compute_source_mix(sources)

        policy_flags: List[str] = []
        reasons: List[str] = []
        should_research_more = False
        confidence_cap = 0.99
        read_depth_hint = 2

        citation_threshold = CITATION_COVERAGE_THRESHOLD if current_info else 0.60
        if policy.strict_evidence:
            citation_threshold = 0.85 if current_info else 0.75
        if float(citation.get("coverage", 0.0)) < citation_threshold:
            policy_flags.append("citation_coverage_low")
            reasons.append("citation_coverage_low")
            should_research_more = True
            confidence_cap = min(confidence_cap, 0.70)
            read_depth_hint = max(read_depth_hint, 3)

        total_sources = len(sources or [])
        min_sources_required = 3 if (policy.require_multi_source or current_info) else 2
        if total_sources < min_sources_required:
            policy_flags.append("insufficient_sources_strict" if policy.strict_evidence else "insufficient_sources")
            reasons.append("insufficient_sources")
            should_research_more = True
            confidence_cap = min(confidence_cap, 0.68 if policy.strict_evidence else 0.75)
            read_depth_hint = max(read_depth_hint, 3)

        if policy.require_trusted_sources_only:
            trusted_count = int(source_mix.get("trusted_sources", 0) or 0)
            if trusted_count < max(2, min_sources_required - 1):
                policy_flags.append("insufficient_trusted_sources_strict")
                reasons.append("insufficient_trusted_sources")
                should_research_more = True
                confidence_cap = min(confidence_cap, 0.64)
                read_depth_hint = max(read_depth_hint, 4)
            elif total_sources > 0 and trusted_count < total_sources:
                policy_flags.append("non_trusted_source_present")
                reasons.append("non_trusted_source_present")
                should_research_more = True
                confidence_cap = min(confidence_cap, 0.70)
                read_depth_hint = max(read_depth_hint, 4)
        elif not bool(source_mix.get("passes_mix", False)):
            policy_flags.append("source_mix_incomplete")
            reasons.append("source_mix_incomplete")
            should_research_more = True
            confidence_cap = min(confidence_cap, 0.72)
            read_depth_hint = max(read_depth_hint, 3)

        freshness_score = float(freshness.get("freshness_score", 0.0))
        has_fresh = bool(freshness.get("has_fresh_source", False))
        freshness_threshold = 0.70 if policy.strict_evidence else 0.55
        if current_info and (not has_fresh or freshness_score < freshness_threshold):
            policy_flags.append("freshness_low")
            reasons.append("freshness_low")
            should_research_more = True
            confidence_cap = min(confidence_cap, 0.68)
            read_depth_hint = max(read_depth_hint, 3)
        elif not current_info and freshness_score > 0.0 and freshness_score < 0.30:
            policy_flags.append("freshness_low")
            confidence_cap = min(confidence_cap, 0.78)

        if not bool(contradiction.get("passes_conflict_gate", True)):
            policy_flags.append("cross_source_conflict")
            reasons.append("cross_source_conflict")
            should_research_more = True
            confidence_cap = min(confidence_cap, 0.62)
            read_depth_hint = max(read_depth_hint, 4)

        if policy.strict_evidence and not policy_flags:
            confidence_cap = min(confidence_cap, 0.93)
            read_depth_hint = max(read_depth_hint, 3)

        if len(sources or []) >= 4 and not policy_flags:
            read_depth_hint = 3 if policy.strict_evidence else 2

        return {
            "current_info": current_info,
            "citation": citation,
            "freshness": freshness,
            "contradiction": contradiction,
            "source_mix": source_mix,
            "policy_flags": policy_flags,
            "reasons": reasons,
            "should_research_more": should_research_more,
            "confidence_cap": confidence_cap,
            "read_depth_hint": read_depth_hint,
        }

    @staticmethod
    def _is_fast_path_query(query_type: str) -> bool:
        return query_type in {"math", "weather", "simple_fact"}

    @staticmethod
    def _math_expr_from_query(goal: str) -> str:
        candidates = re.findall(r"[0-9\s\+\-\*\/\(\)\.]+", goal or "")
        for candidate in sorted(candidates, key=len, reverse=True):
            expr = candidate.strip()
            if expr and any(op in expr for op in ["+", "-", "*", "/"]) and any(ch.isdigit() for ch in expr):
                return expr
        return goal

    @staticmethod
    def _fast_path_tool_for_query(query_type: str, goal: str) -> Tuple[str, str]:
        if query_type == "math":
            return "calculate", TaskManager._math_expr_from_query(goal)
        if query_type == "weather":
            return "search_weather", goal
        return "search_web", goal

    async def _execute_fast_path(self, goal: str, query_type: str, user_scope: str, metrics: Any) -> Optional[Tuple[str, float]]:
        if not self._is_fast_path_query(query_type):
            return None

        tool_name, tool_args = self._fast_path_tool_for_query(query_type, goal)
        metrics.record_planner_bypass()

        call = ToolCall(name=tool_name, args=tool_args, session_id="fast_path", user_id=user_scope)
        exec_ctx = ExecutionContext(user_id=user_scope)
        tool_result = await execute_tool(call, exec_ctx)

        if not tool_result.success:
            return None

        obs = [f"Fast path via {tool_name}: {str(tool_result.data)[:1200]}"]
        sources = []
        if tool_result.source:
            sources.append(
                {
                    "url": str(tool_result.source),
                    "domain": "tool",
                    "trust_score": 0.8,
                    "snippet": str(tool_result.data)[:300],
                }
            )

        final_answer = await self.synthesizer.synthesize(goal, obs, sources)
        confidence = 0.9 if query_type == "math" else 0.8
        return final_answer, confidence

    def _update_workspace_from_node(self, state: AgentState, node: Any) -> Optional[str]:
        if not node.result:
            return None

        if isinstance(node.result, dict) and "result" in node.result:
            res_val = node.result["result"]
        else:
            res_val = node.result

        if isinstance(res_val, dict):
            if "content" in res_val:
                res_text = str(res_val.get("content", ""))
            elif "raw_output" in res_val:
                res_text = str(res_val.get("raw_output", ""))
            else:
                res_text = str(res_val)
        else:
            res_text = str(res_val)

        if not str(res_text).strip():
            return None

        refusals = [
            "i cannot access",
            "i do not have access",
            "i am unable to access",
            "i'm unable to access",
            "knowledge cutoff",
            "as an ai",
            "real-time data",
            "live data",
        ]
        if any(r in res_text.lower() for r in refusals):
            return None

        finding = f"Node {node.node_id} ({node.action_type}): {res_text}"
        if finding in state.observations:
            return None

        state.workspace.knowledge.append(res_text)
        state.workspace.claims.append(
            {
                "claim": res_text[:300],
                "confidence": node.result.get("confidence", "medium") if isinstance(node.result, dict) else "medium",
            }
        )
        primary_source = node.result.get("source") if isinstance(node.result, dict) else "unknown"
        primary_trust = self._safe_trust_score(node.result.get("trust", 0.5)) if isinstance(node.result, dict) else 0.5
        state.workspace.facts.append(
            {
                "entity": res_text[:80],
                "attribute": "extracted_fact",
                "value": res_text[:400],
                "time": "",
                "source": primary_source,
                "source_trust": primary_trust,
                "evidence_span": res_text[:500],
                "confidence": (node.result.get("confidence") if isinstance(node.result, dict) else "medium"),
            }
        )

        sources = self._extract_sources_from_result(node.result if isinstance(node.result, dict) else {})
        claim_key = res_text[:180]
        if claim_key not in state.workspace.claim_sources:
            state.workspace.claim_sources[claim_key] = []

        for src in sources:
            url = str(src.get("url", "")).strip()
            if not url:
                continue
            if not any(s.get("url") == url for s in state.workspace.sources):
                state.workspace.sources.append(
                    {
                        "url": url,
                        "domain": src.get("domain", "unknown"),
                        "trust_score": self._safe_trust_score(src.get("trust", score_source(url)), fallback=score_source(url)),
                        "snippet": str(src.get("snippet", res_text[:300])),
                        "timestamp": src.get("timestamp", ""),
                    }
                )
            if url not in state.workspace.claim_sources[claim_key]:
                state.workspace.claim_sources[claim_key].append(url)

        if node.action_type == "REASONING":
            state.workspace.reasoning_steps.append(res_text[:500])

        state.add_trace(
            action="WORKSPACE_UPDATE",
            node_id=node.node_id,
            metadata={
                "facts": len(state.workspace.facts),
                "sources": len(state.workspace.sources),
                "claims": len(state.workspace.claims),
            },
        )
        return finding

    @staticmethod
    def _summarize_verified_claims(verification_result: Dict[str, Any], ranked_facts: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for c in verification_result.get("verified_claims", []):
            val = str(c.get("value", "")).strip()
            if val:
                out.append(val)
        if not out:
            for rf in ranked_facts[:3]:
                val = str(rf.get("value", "")).strip()
                if val:
                    out.append(val)
        return [o for o in out if o]

    @staticmethod
    def _compact_planner_context(observations: List[str], max_items: int = 6, max_chars: int = 1200) -> str:
        if not observations:
            return ""
        items = [str(o).strip() for o in observations[-max_items:] if str(o).strip()]
        compact = "\n".join(f"- {i[:220]}" for i in items)
        return compact[:max_chars]

    @staticmethod
    def _build_hybrid_reasoning_hints(
        goal: str,
        planned_steps: List[str],
        *,
        small_talk_only: bool = False,
        tool_first_required: bool = False,
    ) -> List[str]:
        cleaned_goal = " ".join(str(goal or "").split())
        if len(cleaned_goal) > 180:
            cleaned_goal = cleaned_goal[:177] + "..."

        hints: List[str] = []
        if small_talk_only:
            return [
                "This looks like a casual message.",
                "I'll reply directly without running web tools.",
            ]

        if cleaned_goal:
            hints.append(f'I interpreted your request as: "{cleaned_goal}".')
        if planned_steps and tool_first_required:
            first = planned_steps[0]
            hints.append(f"I'll start by searching for: {first}.")
            if len(planned_steps) > 1:
                hints.append(f"Then I'll cross-check with: {planned_steps[1]}.")
            hints.append("After evidence collection, I'll synthesize the most supported answer with citations.")
        elif not planned_steps:
            hints.append("I can answer this directly without external tool calls.")
            hints.append("I'll keep the reply concise and natural.")
        else:
            hints.append("I'll use available context and provide a direct answer.")
        return hints[:4]

    @staticmethod
    def _build_direct_reasoning_hints(goal: str, query_type: str, route_type: str) -> List[str]:
        cleaned_goal = " ".join(str(goal or "").split())
        if len(cleaned_goal) > 160:
            cleaned_goal = cleaned_goal[:157] + "..."

        is_casual = TaskManager._is_small_talk_only(goal)
        notes: List[str] = []

        if route_type == "retrieval_hit":
            notes.append("I found a strong trusted match from recent verified context.")
            notes.append("I will answer directly from that validated information.")
            return notes

        if route_type == "fast_path":
            notes.append("This looks like a quick request that needs one fast lookup.")
            notes.append("I will use that result and respond concisely.")
            return notes

        # confidence_gate
        if is_casual:
            notes.append("I interpreted this as a casual conversational message.")
            notes.append("No external tools are needed, so I will reply directly.")
            return notes

        if cleaned_goal:
            notes.append(f'I interpreted your request as: "{cleaned_goal}".')
        if query_type in {"general", "simple_fact"}:
            notes.append("The request is clear enough to answer directly with available context.")
        else:
            notes.append("I can provide a direct answer with current context.")
        notes.append("No external tools are required for this response.")
        return notes[:4]

    @staticmethod
    def _normalize_reasoning_lines(raw_text: str, max_lines: int = 4) -> List[str]:
        text = str(raw_text or "").replace("\r", "\n").strip()
        if not text:
            return []
        lines: List[str] = []
        seen: Set[str] = set()
        blocked_phrases = {
            "direct response path selected",
            "no tools needed for this request",
            "responding directly based on confidence gate",
        }
        for raw_line in text.split("\n"):
            line = str(raw_line or "").strip()
            if not line:
                continue
            line = re.sub(r"^\s*[-*]\s*", "", line)
            line = re.sub(r"^\s*\d+[.)]\s*", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("recognizing this as"):
                line = "This looks like a casual message."
                lower = line.lower()
            if lower.startswith("i interpreted this as a casual conversational message"):
                line = "This looks like a casual message."
                lower = line.lower()
            if lower in blocked_phrases:
                continue
            key = lower[:220]
            if key in seen:
                continue
            seen.add(key)
            lines.append(line[:220])
            if len(lines) >= max_lines:
                break
        return lines

    @staticmethod
    def _looks_mechanical_reasoning(lines: List[str]) -> bool:
        if not lines:
            return True
        mechanical_hits = 0
        for line in lines:
            l = str(line or "").strip().lower()
            if not l:
                continue
            if l.startswith("planning:") or l.startswith("step "):
                mechanical_hits += 1
            if "task_progress" in l or "state=" in l:
                mechanical_hits += 1
        return mechanical_hits >= max(1, len(lines) // 2)

    async def _build_llm_reasoning_hints(
        self,
        *,
        goal: str,
        query_type: str,
        route_type: str,
        small_talk_only: bool,
        tool_first_required: bool = False,
        planned_steps: Optional[List[str]] = None,
        fallback_hints: Optional[List[str]] = None,
        max_lines: int = 4,
    ) -> List[str]:
        steps = [str(s).strip() for s in (planned_steps or []) if str(s).strip()][:4]
        fallback = [str(s).strip() for s in (fallback_hints or []) if str(s).strip()]
        normalized_fallback = self._normalize_reasoning_lines("\n".join(fallback), max_lines=max_lines)

        # Keep simple/direct routes fast: do not block on an extra LLM call for
        # greeting/casual requests or confidence-gate shortcuts.
        if small_talk_only or route_type == "confidence_gate":
            if normalized_fallback:
                return normalized_fallback
            return []

        cache_key = (
            f"{route_type}::{query_type}::{int(small_talk_only)}::{int(tool_first_required)}::"
            f"{goal.strip().lower()[:200]}::{'|'.join(steps)}"
        )
        if cache_key in self.reasoning_hints_cache:
            return list(self.reasoning_hints_cache[cache_key])

        system_prompt = (
            "You generate user-visible Thoughts lines for an AI assistant UI.\n"
            "Return 2 to 4 short lines, one per line.\n"
            "These must be high-level, user-safe reasoning summaries only.\n"
            "Do not reveal hidden chain-of-thought.\n"
            "Do not use markdown, headings, numbering, JSON, or bullets.\n"
            "Do not mention internal policies, confidence gates, classifier names, or model names.\n"
            "Avoid robotic wording such as 'Recognizing this as' or 'Classified as'.\n"
            "Use plain natural language, like a human assistant thinking out loud at a high level.\n"
            "Do not draft the final assistant reply text inside Thoughts.\n"
            "Avoid imperative planning lines like 'Time to respond' or 'I should respond'.\n"
            "Never output these phrases: 'Direct response path selected', 'No tools needed for this request'.\n"
            "Use natural language matched to the user's request."
        )
        user_prompt = (
            f"User request: {goal}\n"
            f"Query type: {query_type}\n"
            f"Route type: {route_type}\n"
            f"Small talk only: {small_talk_only}\n"
            f"Tool-first required: {tool_first_required}\n"
            f"Planned steps: {steps}\n"
            f"Fallback hints: {fallback}\n"
        )

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=180,
                ),
                timeout=1.8,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "")
            normalized = self._normalize_reasoning_lines(content, max_lines=max_lines)
            if small_talk_only and normalized:
                filtered: List[str] = []
                for line in normalized:
                    lower = str(line or "").strip().lower()
                    if lower.startswith(("time to respond", "i should respond", "now i will respond")):
                        continue
                    if re.match(r"^(hi|hello|hey|i('?| a)m|i am|how about you)\b", lower):
                        continue
                    filtered.append(line)
                if filtered:
                    normalized = filtered[:2]
            if normalized:
                self.reasoning_hints_cache[cache_key] = normalized
                return normalized
        except Exception:
            pass

        if normalized_fallback:
            self.reasoning_hints_cache[cache_key] = normalized_fallback
            return normalized_fallback
        return []

    @staticmethod
    def _lane_from_state(state: AgentState) -> str:
        try:
            meta = state.workspace.intermediate_results.get("request_metadata", {})
            lane = str((meta or {}).get("lane") or "").strip().lower()
            if lane in {"fast", "heavy"}:
                return lane
        except Exception:
            pass
        return "heavy"

    def _reliability_budget_ms(self, state: AgentState, mode_policy: Optional[ModePolicy] = None) -> int:
        lane_budget = FAST_LANE_RELIABILITY_BUDGET_MS if self._lane_from_state(state) == "fast" else HEAVY_LANE_RELIABILITY_BUDGET_MS
        if int(lane_budget or 0) <= 0:
            return int(self.max_task_latency_ms)
        policy = mode_policy or self._resolve_mode_policy(str(state.mode or ""))
        min_budget = int(policy.min_reliability_budget_ms)
        return int(max(min_budget, int(lane_budget)))

    @staticmethod
    def _parse_backoff_ms() -> List[int]:
        out: List[int] = []
        for part in str(RELIABILITY_TOOL_RETRY_BACKOFF_MS or "").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(max(0, int(part)))
            except Exception:
                continue
        return out or [200, 600]

    async def _synthesize_safe_fallback(
        self,
        *,
        goal: str,
        observations: List[str],
        sources: List[Dict[str, Any]],
    ) -> str:
        # Evidence-backed concise fallback framing with explicit uncertainty and next-step guidance.
        evidence_lines = [str(o).strip() for o in observations if str(o).strip()][:6]
        if not evidence_lines:
            evidence_lines = ["Evidence is limited in this run."]

        scaffold = [
            "Based on available sources, the most supported findings are:",
            *[f"- {line[:280]}" for line in evidence_lines],
            "Confidence is limited due to incomplete or conflicting evidence.",
            "Next step: validate with additional independent, recent sources before final decisions.",
        ]
        return await self.synthesizer.synthesize(
            goal=f"[SAFE_FALLBACK] {goal}",
            observations=scaffold,
            sources=sources,
        )

    @staticmethod
    def _chunk_stream_text(text: str, chunk_size: int = 48) -> List[str]:
        normalized = str(text or "")
        if not normalized:
            return []
        size = max(16, int(chunk_size or 48))
        return [normalized[i:i + size] for i in range(0, len(normalized), size)]

    async def _emit_final_answer_stream(self, answer: str) -> AsyncGenerator[str, None]:
        text = str(answer or "").strip()
        yield "\n[FINAL_ANSWER]\n"
        if not text:
            yield "\n"
            return
        for piece in self._chunk_stream_text(text, chunk_size=52):
            if not piece:
                continue
            yield piece
            await asyncio.sleep(0)
        yield "\n"

    async def _generate_direct_answer(self, goal: str, query_type: str, *, small_talk_only: bool) -> str:
        if small_talk_only:
            quick = self._quick_small_talk_response(goal)
            if quick:
                return quick
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=FAST_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are Relyce AI. Reply to casual greetings naturally and warmly.\n"
                                    "Use 1-2 short sentences.\n"
                                    "Match user tone lightly (including slang if present) without overdoing it.\n"
                                    "Avoid robotic/meta phrasing like 'I interpreted this as'.\n"
                                    "Do not mention internal process or tools.\n"
                                    "Do not mention being AI unless asked."
                                ),
                            },
                            {"role": "user", "content": goal},
                        ],
                        temperature=0.6,
                        max_tokens=80,
                    ),
                    timeout=1.8,
                )
                casual = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
                if casual:
                    return casual
            except Exception:
                pass
            fallback = self._quick_small_talk_response(goal)
            if fallback:
                return fallback
            return "Hey macha, all good here. What's on your mind?"

        return await self.synthesizer.synthesize(goal=goal, observations=[goal], sources=[])

    @staticmethod
    def _quick_small_talk_response(goal: str) -> Optional[str]:
        q = " ".join(str(goal or "").strip().lower().split())
        if not q:
            return "Hey macha! What's on your mind?"

        has_bro_tone = any(tok in q for tok in ("macha", "bro", "buddy", "dude"))

        if re.search(r"\bhow are (you|u)\b", q):
            return (
                "I'm doing great, macha. How are you doing?"
                if has_bro_tone
                else "I'm doing great, thanks for asking. How are you?"
            )

        if re.match(r"^(thanks|thank you)\b", q):
            return "Anytime. Happy to help."

        if re.match(r"^(hi+|hey+|hello+|yo+|sup+|hola+)\b", q):
            return (
                "Hey macha! I'm here. What do you want to work on?"
                if has_bro_tone
                else "Hey! I'm here. What can I help you with?"
            )

        if len(q.split()) <= 5:
            return "Got you. Tell me what you want and we'll do it."

        return None


    async def run_goal(
        self,
        goal: str,
        initial_state: Optional[AgentState] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Runs the autonomous loop until the goal is satisfied or budget exceeded.
        """
        metrics = get_metrics_collector()
        metrics.record_task_started()
        started_at = time.time()

        state = initial_state or AgentState(query=goal, goal=goal)
        state.workspace.goal = goal
        log_trace(state.trace_id, "TASK_START", {"goal": goal})
        req_meta = dict(state.workspace.intermediate_results.get("request_metadata") or {})
        queue_wait_ms = int(float(req_meta.get("queue_wait_ms") or 0))
        state.workspace.intermediate_results["queue_wait_ms"] = queue_wait_ms
        optimized_ctx, budget_meta = self._apply_context_budget(context_messages=state.context_messages)
        state.context_messages = optimized_ctx
        state.workspace.intermediate_results.update(budget_meta)

        yield f"[TaskManager] Starting autonomous task loop | Trace: {state.trace_id}\n"
        yield f"[TaskManager] Goal: {goal}\n"
        yield self.event_streamer.emit("task_started", trace_id=state.trace_id, goal=goal) + "\n"
        yield self.event_streamer.emit(
            "progress",
            state="analyzing_request",
            label="Understanding the request and deciding next steps",
        ) + "\n"

        user_scope = str(state.user_id or "global")
        normalized_goal = self.retrieval.normalize_query(goal)
        query_type = classify_query(normalized_goal)
        current_info_query = is_current_info_query(goal)
        classifier_conf = query_classifier_confidence(query_type, goal)
        reliability_query_type = classify_reliability_query_type(query_type, goal)
        requested_mode_raw = str(req_meta.get("requested_mode_raw") or state.mode or "smart").strip().lower() or "smart"
        mode_name = normalize_chat_mode(str(requested_mode_raw or state.mode or "smart"))
        base_mode_name = mode_name
        base_mode_display = "auto" if requested_mode_raw == "auto" else base_mode_name
        small_talk_only = self._is_small_talk_only(goal)
        dynamic_context = self._infer_dynamic_policy_context(
            goal=goal,
            query_type=query_type,
            reliability_query_type=reliability_query_type,
            small_talk_only=small_talk_only,
            current_info_query=current_info_query,
        )
        llm_dynamic_context = await self._infer_dynamic_policy_context_llm(
            goal=goal,
            query_type=query_type,
            reliability_query_type=reliability_query_type,
            small_talk_only=small_talk_only,
            current_info_query=current_info_query,
        )
        if llm_dynamic_context:
            merged_context = dict(dynamic_context)
            for key in ("intent", "risk_level", "time_sensitive", "critical_research"):
                if key in llm_dynamic_context:
                    merged_context[key] = llm_dynamic_context[key]
            merged_context["classifier_source"] = llm_dynamic_context.get("classifier_source", "llm")
            merged_context["classifier_confidence"] = float(llm_dynamic_context.get("classifier_confidence", 0.0) or 0.0)
            dynamic_context = merged_context
        else:
            dynamic_context["classifier_source"] = "heuristic"
            dynamic_context["classifier_confidence"] = round(float(classifier_conf), 4)

        effective_mode, override_reason = self._resolve_effective_mode(
            base_mode=base_mode_name,
            base_mode_raw=requested_mode_raw,
            goal=goal,
            dynamic_context=dynamic_context,
            current_info_query=current_info_query,
        )
        override_applied = bool(effective_mode != base_mode_name or requested_mode_raw == "auto")
        mode_name = effective_mode
        state.mode = mode_name
        base_mode_policy = self._resolve_mode_policy(mode_name)
        mode_policy, mode_policy_meta = self._resolve_effective_policy_stack(
            mode_policy=base_mode_policy,
            role_name="executor",
            dynamic_context=dynamic_context,
            goal=goal,
            user_id=user_scope,
        )

        override_telemetry: Dict[str, Any] = {
            "input": str(goal or "")[:1200],
            "base_mode": base_mode_display,
            "requested_mode_raw": requested_mode_raw,
            "detected_intent": str(dynamic_context.get("intent") or "general"),
            "override_applied": bool(override_applied),
            "final_mode": mode_name,
            "tools_used": [],
            "retries": 0,
            "response_time": 0,
            "success_flag": False,
            "role_flow": [],
            "role_metrics": {},
            "role_summary": {},
        }
        if override_reason:
            override_telemetry["override_reason"] = override_reason
        task_success = False
        telemetry_flushed = False
        run_outcome_recorded = False

        def _flush_override_telemetry(success: bool) -> None:
            nonlocal telemetry_flushed
            if telemetry_flushed:
                return
            try:
                t_metrics = state.workspace.intermediate_results.get("tool_metrics", {})
                tools_used = sorted(str(name) for name in (t_metrics or {}).keys())
                tool_calls = int(
                    sum(int(float((ms or {}).get("calls", 0) or 0)) for ms in (t_metrics or {}).values())
                )
                retries = int(state.workspace.intermediate_results.get("reliability_retries", 0) or 0)
                response_time_ms = int((time.time() - started_at) * 1000)
                role_flow = list(state.workspace.intermediate_results.get("role_flow") or [])
                role_metrics = dict(state.workspace.intermediate_results.get("role_metrics") or {})
                role_summary = dict(state.workspace.intermediate_results.get("role_summary") or {})
                override_telemetry["tools_used"] = tools_used
                override_telemetry["retries"] = retries
                override_telemetry["response_time"] = response_time_ms
                override_telemetry["success_flag"] = bool(success)
                override_telemetry["role_flow"] = role_flow
                override_telemetry["role_metrics"] = role_metrics
                override_telemetry["role_summary"] = role_summary
                state.workspace.intermediate_results["mode_override_telemetry"] = dict(override_telemetry)
                log_trace(state.trace_id, "MODE_OVERRIDE_TELEMETRY", dict(override_telemetry))

                response_profile = self._record_response_profile(
                    mode=mode_name,
                    latency_ms=response_time_ms,
                    retries=retries,
                    tool_calls=tool_calls,
                    success=bool(success),
                )
                state.workspace.intermediate_results["response_profile"] = response_profile
                log_trace(state.trace_id, "MODE_RESPONSE_PROFILE", dict(response_profile))
                for role_name, role_data in role_metrics.items():
                    role_key = str(role_name or "executor").strip().lower() or "executor"
                    stats = self.role_response_profiles.setdefault(
                        role_key,
                        {
                            "runs": 0.0,
                            "successes": 0.0,
                            "latency_ms_total": 0.0,
                            "retries_total": 0.0,
                            "tool_calls_total": 0.0,
                            "parallel_exception_count": 0.0,
                        },
                    )
                    stats["runs"] += max(0.0, float(role_data.get("runs", 0) or 0))
                    stats["successes"] += max(0.0, float(role_data.get("success_count", 0) or 0))
                    stats["latency_ms_total"] += max(0.0, float(role_data.get("latency_ms_total", 0) or 0))
                    stats["retries_total"] += max(0.0, float(role_data.get("retries_total", 0) or 0))
                    stats["tool_calls_total"] += max(0.0, float(role_data.get("tool_calls_total", 0) or 0))
                    stats["parallel_exception_count"] += max(0.0, float(role_data.get("parallel_exception_count", 0) or 0))
                tuning_event = self._maybe_apply_auto_tuning(mode=mode_name, response_profile=response_profile)
                if tuning_event:
                    state.workspace.intermediate_results["mode_auto_tuning_event"] = dict(tuning_event)
                    log_trace(state.trace_id, "MODE_AUTO_TUNING", dict(tuning_event))
                telemetry_flushed = True
            except Exception:
                pass

        def _record_run_outcome(*, answer: str, confidence: float, success: bool) -> None:
            nonlocal run_outcome_recorded
            if run_outcome_recorded:
                return
            try:
                telemetry = state.workspace.intermediate_results.get("mode_override_telemetry", {})
                tools_used = list(telemetry.get("tools_used") or [])
                retries = int(telemetry.get("retries", 0) or 0)
                response_time_ms = int(telemetry.get("response_time", int((time.time() - started_at) * 1000)) or 0)
                role_flow = list(telemetry.get("role_flow") or [])
                role_metrics = dict(telemetry.get("role_metrics") or {})
                role_summary = dict(telemetry.get("role_summary") or {})
                adaptive_meta = dict(mode_policy_meta.get("adaptive") or {})
                self._record_debug_run(
                    {
                        "query": str(goal or "")[:1200],
                        "answer_preview": str(answer or "")[:1200],
                        "requested_mode": base_mode_name,
                        "requested_mode_raw": requested_mode_raw,
                        "mode": mode_name,
                        "user_id": user_scope,
                        "detected_intent": str(dynamic_context.get("intent") or "general"),
                        "override_applied": bool(override_applied),
                        "override_reason": override_reason or "",
                        "tools_used": tools_used,
                        "tool_count": len(tools_used),
                        "retries": retries,
                        "response_time_ms": response_time_ms,
                        "confidence": float(confidence or 0.0),
                        "success": bool(success),
                        "role_flow": role_flow[-60:],
                        "role_metrics": role_metrics,
                        "role_summary": role_summary,
                        "adaptive_policy": {
                            "bucket_key": str(adaptive_meta.get("bucket_key") or ""),
                            "rollout_applied": bool(adaptive_meta.get("rollout_applied", True)),
                            "apply_ratio": float(
                                adaptive_meta.get("apply_ratio")
                                if adaptive_meta.get("apply_ratio") is not None
                                else (self.debug_config.get("adaptive_apply_ratio", 1.0) if self.debug_config.get("adaptive_apply_ratio", 1.0) is not None else 1.0)
                            ),
                            "applied": bool(adaptive_meta.get("applied", False)),
                        },
                    }
                )
                run_outcome_recorded = True
            except Exception:
                pass

        state.workspace.intermediate_results["mode_override_telemetry"] = dict(override_telemetry)
        log_trace(
            state.trace_id,
            "MODE_OVERRIDE",
            {
                "base_mode": base_mode_display,
                "final_mode": mode_name,
                "override_applied": bool(override_applied),
                "detected_intent": override_telemetry["detected_intent"],
                "reason": override_reason or "none",
            },
        )
        yield self.event_streamer.emit(
            "progress",
            state="mode_override",
            label=(
                f"Auto mode selected {mode_name} based on intent"
                if requested_mode_raw == "auto"
                else (
                    f"Mode tuned from {base_mode_name} to {mode_name}"
                    if override_applied
                    else f"Mode retained as {mode_name}"
                )
            ),
            base_mode=base_mode_display,
            final_mode=mode_name,
            override_applied=bool(override_applied),
            override_reason=override_reason or "",
            detected_intent=str(dynamic_context.get("intent") or "general"),
        ) + "\n"

        state.workspace.intermediate_results["mode_policy"] = {
            "mode_name": mode_policy.mode_name,
            "base_min_reliability_budget_ms": int(base_mode_policy.min_reliability_budget_ms),
            "base_max_reliability_retries": int(base_mode_policy.max_reliability_retries),
            "memory_read_enabled": bool(mode_policy.memory_read_enabled),
            "memory_write_enabled": bool(mode_policy.memory_write_enabled),
            "min_reliability_budget_ms": int(mode_policy.min_reliability_budget_ms),
            "respect_small_talk_for_forced_tools": bool(mode_policy.respect_small_talk_for_forced_tools),
            "live_query_forces_tools": bool(mode_policy.live_query_forces_tools),
            "reasoning_hints_enabled": bool(mode_policy.reasoning_hints_enabled),
            "tool_force_for_non_smalltalk": bool(mode_policy.tool_force_for_non_smalltalk),
            "strict_evidence": bool(mode_policy.strict_evidence),
            "require_multi_source": bool(mode_policy.require_multi_source),
            "require_trusted_sources_only": bool(mode_policy.require_trusted_sources_only),
            "max_reliability_retries": int(mode_policy.max_reliability_retries),
            "role_policy_meta": mode_policy_meta,
            "adaptive_bucket_key": str((mode_policy_meta.get("adaptive") or {}).get("bucket_key") or ""),
            "adaptive_snapshot": dict((mode_policy_meta.get("adaptive") or {}).get("snapshot") or {}),
            "final_clamp": dict(mode_policy_meta.get("final_clamp") or {}),
        }
        state.workspace.intermediate_results["dynamic_policy_context"] = dynamic_context
        if llm_dynamic_context:
            state.workspace.intermediate_results["dynamic_policy_context_llm"] = llm_dynamic_context
        base_force_tool_route = self._force_tool_route(goal, query_type)
        force_tool_route = bool(base_force_tool_route)
        if mode_policy.live_query_forces_tools and self._is_hybrid_live_query(goal):
            force_tool_route = True
        if mode_policy.tool_force_for_non_smalltalk and not small_talk_only:
            force_tool_route = True
        if mode_policy.respect_small_talk_for_forced_tools and small_talk_only:
            force_tool_route = False
        if classifier_conf < 0.65 and not force_tool_route:
            reliability_query_type = "reasoning"
        enforce_evidence = should_enforce_evidence(reliability_query_type, goal)
        tool_first_required = (
            force_tool_route
            or (FF_TOOL_FIRST_EVIDENCE and enforce_evidence and needs_tool_first(query_type, goal))
        )
        repair_attempts = 0
        unsupported_claims: List[Dict[str, Any]] = []
        evidence_quality = "low"
        reliability_budget_exhausted = False
        fallback_used = False
        tool_memory_hits = 0
        reliability_budget_ms = self._reliability_budget_ms(state, mode_policy=mode_policy)

        route_label = (
            f"Classified request as {query_type.replace('_', ' ')}; "
            + ("tool-first evidence enabled" if tool_first_required else "direct/tool route evaluation")
        )
        yield self.event_streamer.emit(
            "progress",
            state="route_selected",
            label=route_label,
            query_type=query_type,
        ) + "\n"

        query_cache_key = self._cache_key_for_goal(normalized_goal)
        yield self.event_streamer.emit(
            "progress",
            state="cache_lookup",
            label="Checking recent query cache and retrieval memory",
        ) + "\n"
        if query_cache_key in self.response_cache and not tool_first_required:
            cached = self.response_cache[query_cache_key]
            metrics.record_query_cache_hit()
            metrics.record_termination("query_cache_hit")
            yield "[TaskManager] Query cache hit. Returning cached answer.\n"
            yield self.event_streamer.emit(
                "progress",
                state="retrieval_hit",
                label="Using cached answer from recent matching query",
            ) + "\n"
            yield self.event_streamer.emit("planning_complete", mode="query_cache_hit") + "\n"
            yield self.event_streamer.emit(
                "final_answer",
                confidence=0.9,
                base_mode=base_mode_name,
                final_mode=mode_name,
                override_applied=bool(override_applied),
                override_reason=override_reason or "",
            ) + "\n"
            async for chunk in self._emit_final_answer_stream(cached):
                yield chunk
            metrics.record_task_completed()
            task_success = True
            _flush_override_telemetry(task_success)
            _record_run_outcome(answer=cached, confidence=0.9, success=task_success)
            return

        if FF_TOOL_MEMORY and mode_policy.memory_read_enabled:
            mem_entry = get_tool_memory_store().get(
                user_id=user_scope,
                session_id=str(state.session_id or ""),
                query=normalized_goal,
                tool_name="research_graph",
                schema_version=TOOL_SCHEMA_VERSION,
                current_info=current_info_query,
            )
            if mem_entry:
                tool_memory_hits += 1
                metrics.record_custom_counter("tool_memory_hit")
                state.workspace.intermediate_results["tool_memory_hit"] = True
                state.workspace.intermediate_results["tool_memory_freshness"] = mem_entry.freshness_score
                if mem_entry.key_facts:
                    state.observations.extend(mem_entry.key_facts[:4])
                for link in mem_entry.source_links[:4]:
                    if link and not any(s.get("url") == link for s in state.workspace.sources):
                        state.workspace.sources.append(
                            {
                                "url": link,
                                "domain": "memory",
                                "trust_score": max(0.5, float(mem_entry.confidence)),
                                "snippet": "Reused from tool memory",
                                "timestamp": "",
                            }
                        )
                yield self.event_streamer.emit(
                    "tool_memory_hit",
                    freshness=round(float(mem_entry.freshness_score), 4),
                    confidence=round(float(mem_entry.confidence), 4),
                    query_type=reliability_query_type,
                ) + "\n"

        retrieval = await self.retrieval.lookup(normalized_goal, user_id=user_scope)
        metrics.record_research_call()
        if retrieval.hit and retrieval.record and not tool_first_required:
            metrics.record_research_cache_hit()
            record = retrieval.record
            cached_answer = str(record.get("summary", "")).strip()
            if cached_answer:
                state.add_trace(action="RETRIEVAL_HIT", metadata={"similarity": retrieval.similarity, "topic_id": retrieval.topic_id})
                metrics.record_termination("retrieval_hit")
                yield self.event_streamer.emit(
                    "progress",
                    state="retrieval_hit",
                    label="Using trusted cached knowledge",
                ) + "\n"
                retrieval_fallback = self._build_direct_reasoning_hints(goal, query_type, "retrieval_hit")
                retrieval_hints = await self._build_llm_reasoning_hints(
                    goal=goal,
                    query_type=query_type,
                    route_type="retrieval_hit",
                    small_talk_only=small_talk_only,
                    fallback_hints=retrieval_fallback,
                )
                for hint in retrieval_hints:
                    yield self.event_streamer.emit(
                        "progress",
                        state="task_progress",
                        label=hint,
                        topic=hint,
                    ) + "\n"
                    await asyncio.sleep(0.06)
                retrieval_conf = float(record.get("confidence", 0.8))
                yield self.event_streamer.emit("planning_complete", mode="retrieval_hit", similarity=round(retrieval.similarity, 4)) + "\n"
                yield self.event_streamer.emit(
                    "final_answer",
                    confidence=retrieval_conf,
                    base_mode=base_mode_name,
                    final_mode=mode_name,
                    override_applied=bool(override_applied),
                    override_reason=override_reason or "",
                ) + "\n"
                async for chunk in self._emit_final_answer_stream(cached_answer):
                    yield chunk
                log_run(
                {
                    "query": normalized_goal,
                    "query_type": query_type,
                    "route_type": "retrieval_hit",
                        "plan": ["retrieval_hit"],
                        "tools_used": [],
                        "sources": record.get("sources", []),
                        "claims": record.get("claims", []),
                    "verification_result": {"cached": True},
                    "latency_ms": 0,
                    "confidence": retrieval_conf,
                    "queue_wait_ms": int(queue_wait_ms),
                    "input_token_count": int(state.workspace.intermediate_results.get("input_token_count", 0) or 0),
                    "compression_applied": bool(state.workspace.intermediate_results.get("compression_applied", False)),
                    "compression_ratio": float(state.workspace.intermediate_results.get("compression_ratio", 0.0) or 0.0),
                },
                user_id=user_scope,
            )
                metrics.record_task_completed()
                task_success = True
                _flush_override_telemetry(task_success)
                _record_run_outcome(answer=cached_answer, confidence=retrieval_conf, success=task_success)
                return

        confidence_gate = None if tool_first_required else await self._try_confidence_gate(goal=goal, query_type=query_type)
        if confidence_gate:
            preset_answer, conf = confidence_gate
            metrics.record_termination("confidence_gate_direct_answer")
            confidence_fallback = self._build_direct_reasoning_hints(goal, query_type, "confidence_gate")
            confidence_hints = await self._build_llm_reasoning_hints(
                goal=goal,
                query_type=query_type,
                route_type="confidence_gate",
                small_talk_only=small_talk_only,
                fallback_hints=confidence_fallback,
            )
            for hint in confidence_hints:
                yield self.event_streamer.emit(
                    "progress",
                    state="task_progress",
                    label=hint,
                    topic=hint,
                ) + "\n"
                await asyncio.sleep(0.08)
            yield self.event_streamer.emit("planning_complete", mode="confidence_gate", confidence=round(conf, 2)) + "\n"
            yield self.event_streamer.emit(
                "progress",
                state="synthesis_started",
                label="Drafting final response",
            ) + "\n"
            final_answer = preset_answer or await self._generate_direct_answer(
                goal=goal,
                query_type=query_type,
                small_talk_only=small_talk_only,
            )
            self.response_cache[query_cache_key] = final_answer
            yield self.event_streamer.emit(
                "final_answer",
                confidence=conf,
                base_mode=base_mode_name,
                final_mode=mode_name,
                override_applied=bool(override_applied),
                override_reason=override_reason or "",
            ) + "\n"
            async for chunk in self._emit_final_answer_stream(final_answer):
                yield chunk
            log_run(
                {
                    "query": normalized_goal,
                    "query_type": query_type,
                    "route_type": "confidence_gate",
                    "plan": ["confidence_gate"],
                    "tools_used": [],
                    "sources": [],
                    "claims": [],
                    "verification_result": {"confidence_gate": True},
                    "latency_ms": int((time.time() - started_at) * 1000),
                    "confidence": conf,
                    "queue_wait_ms": int(queue_wait_ms),
                    "input_token_count": int(state.workspace.intermediate_results.get("input_token_count", 0) or 0),
                    "compression_applied": bool(state.workspace.intermediate_results.get("compression_applied", False)),
                    "compression_ratio": float(state.workspace.intermediate_results.get("compression_ratio", 0.0) or 0.0),
                },
                user_id=user_scope,
            )
            metrics.record_task_completed()
            task_success = True
            _flush_override_telemetry(task_success)
            _record_run_outcome(answer=final_answer, confidence=conf, success=task_success)
            return
        fast_path_result = None if tool_first_required else await self._execute_fast_path(goal=goal, query_type=query_type, user_scope=user_scope, metrics=metrics)
        if fast_path_result:
            final_answer, fp_confidence = fast_path_result
            fp_tool_name, fp_tool_args = self._fast_path_tool_for_query(query_type, goal)
            fast_fallback = self._build_direct_reasoning_hints(goal, query_type, "fast_path")
            fast_hints = await self._build_llm_reasoning_hints(
                goal=goal,
                query_type=query_type,
                route_type="fast_path",
                small_talk_only=small_talk_only,
                fallback_hints=fast_fallback,
            )
            for hint in fast_hints:
                yield self.event_streamer.emit(
                    "progress",
                    state="task_progress",
                    label=hint,
                    topic=hint,
                ) + "\n"
                await asyncio.sleep(0.06)
            yield self.event_streamer.emit(
                "tool_call",
                tool=fp_tool_name,
                status="started",
                args_preview=fp_tool_args[:180],
            ) + "\n"
            yield self.event_streamer.emit(
                "tool_result",
                tool=fp_tool_name,
                status="ok",
                args_preview=fp_tool_args[:180],
            ) + "\n"
            self.response_cache[query_cache_key] = final_answer
            metrics.record_termination("fast_path")
            yield self.event_streamer.emit("planning_complete", mode="fast_path", query_type=query_type) + "\n"
            yield self.event_streamer.emit(
                "progress",
                state="synthesis_started",
                label="Drafting final response from gathered tool result",
            ) + "\n"
            yield self.event_streamer.emit(
                "final_answer",
                confidence=fp_confidence,
                base_mode=base_mode_name,
                final_mode=mode_name,
                override_applied=bool(override_applied),
                override_reason=override_reason or "",
            ) + "\n"
            async for chunk in self._emit_final_answer_stream(final_answer):
                yield chunk
            log_run(
                {
                    "query": normalized_goal,
                    "query_type": query_type,
                    "route_type": "fast_path",
                    "plan": ["fast_path", query_type],
                    "tools_used": [{"name": self._fast_path_tool_for_query(query_type, goal)[0], "success": True, "latency_ms": 0}],
                    "sources": [],
                    "claims": [],
                    "verification_result": {"fast_path": True},
                    "latency_ms": int((time.time() - started_at) * 1000),
                    "confidence": fp_confidence,
                    "queue_wait_ms": int(queue_wait_ms),
                    "input_token_count": int(state.workspace.intermediate_results.get("input_token_count", 0) or 0),
                    "compression_applied": bool(state.workspace.intermediate_results.get("compression_applied", False)),
                    "compression_ratio": float(state.workspace.intermediate_results.get("compression_ratio", 0.0) or 0.0),
                },
                user_id=user_scope,
            )
            metrics.record_task_completed()
            task_success = True
            _flush_override_telemetry(task_success)
            _record_run_outcome(answer=final_answer, confidence=fp_confidence, success=task_success)
            return

        try:
            final_confidence = 0.0
            final_verification: Dict[str, Any] = {}
            ranked_facts: List[Dict[str, Any]] = []
            stagnant_iterations = 0

            while not state.should_terminate():
                elapsed_ms = (time.time() - started_at) * 1000
                if elapsed_ms >= self.max_task_latency_ms:
                    state.terminate = True
                    metrics.record_termination("max_latency")
                    yield "\n[TaskManager] Max latency reached. Finalizing with available findings...\n"
                    break
                reliability_elapsed_ms = elapsed_ms
                # Allow at least one full planning/tool iteration before enforcing the reliability budget.
                if FF_CONFIDENCE_CRITIC_LOOPS and state.iteration > 0 and reliability_elapsed_ms >= reliability_budget_ms:
                    reliability_budget_exhausted = True
                    metrics.record_custom_counter("reliability_budget_exhausted")
                    metrics.record_termination("reliability_budget_exhausted")
                    yield self.event_streamer.emit(
                        "progress",
                        state="reliability_budget_exhausted",
                        label="Reliability budget exhausted; finalizing best-safe answer",
                        budget_ms=reliability_budget_ms,
                    ) + "\n"
                    break

                state.record_iteration()
                log_trace(state.trace_id, "ITERATION_START", {"iteration": state.iteration})
                yield f"\n--- Iteration {state.iteration} ---\n"

                pipeline_result = await run_agent_pipeline(
                    user_query=goal,
                    context_messages=state.context_messages,
                    session_id=state.session_id,
                )
                # Product decision: do not permission-block tools in agent execution.
                # Keep normal per-tool validation/error handling in ToolExecutor.
                pipeline_result.tool_allowed = True
                existing_tools = set(getattr(pipeline_result, "allowed_tools", []) or [])
                existing_tools.update(TOOLS.keys())
                pipeline_result.allowed_tools = sorted(existing_tools)

                yield "[TaskManager] Planning step...\n"
                state.add_trace(action="PLANNER_START", metadata={"goal": goal[:120]})
                plan_start_t = time.time()
                policy_flags_ctx = [
                    str(f).strip()
                    for f in (state.workspace.intermediate_results.get("policy_flags") or [])
                    if str(f).strip()
                ]
                read_depth_ctx = int(state.workspace.intermediate_results.get("required_read_depth", 0) or 0)
                planner_context = self._compact_planner_context(state.observations)
                if policy_flags_ctx:
                    planner_context = (
                        f"{planner_context}\nPolicy gaps from previous iteration: {', '.join(policy_flags_ctx)}."
                    ).strip()
                if read_depth_ctx > 0:
                    planner_context = (
                        f"{planner_context}\nRead-depth target: {max(2, min(4, read_depth_ctx))}."
                    ).strip()
                tool_confidence_guidance = self._build_tool_confidence_guidance()
                if tool_confidence_guidance:
                    planner_context = (
                        f"{planner_context}\n{tool_confidence_guidance}"
                    ).strip()

                graph = await compile_plan_graph(
                    session_id=state.session_id or "default",
                    task_id=f"{state.trace_id}_{state.iteration}",
                    query=goal,
                    context=planner_context,
                    client=self.client,
                    model_to_use=self.model_to_use,
                )
                if tool_first_required:
                    has_tool_nodes = any(
                        str(getattr(node, "action_type", "")).upper() == "TOOL_CALL"
                        for node in graph.nodes.values()
                    )
                    if not has_tool_nodes:
                        # Hard reliability fallback: research/current-info queries must include at least one tool step.
                        graph = await compile_plan_graph(
                            session_id=state.session_id or "default",
                            task_id=f"{state.trace_id}_{state.iteration}_forced",
                            query=f"research latest verified sources for: {goal}",
                            context=planner_context,
                            client=self.client,
                            model_to_use=self.model_to_use,
                        )
                role_resolution_rows: List[Dict[str, Any]] = []
                for node in graph.nodes.values():
                    role_resolution = resolve_plan_node_role(
                        node_id=str(getattr(node, "node_id", "")),
                        action_type=str(getattr(node, "action_type", "")),
                        declared_role=str(getattr(node, "role", "") or ""),
                    )
                    node.role = role_resolution.role
                    node.role_fallback_applied = bool(role_resolution.role_fallback_applied)
                    node.role_resolution_source = str(role_resolution.role_resolution_source or "action_mapping")
                    if role_resolution.warning:
                        print(f"[Role Resolver] {role_resolution.warning} (node={node.node_id})")
                    role_resolution_rows.append(
                        {
                            "node_id": str(node.node_id),
                            "action_type": str(node.action_type),
                            "role": str(node.role),
                            "role_fallback_applied": bool(node.role_fallback_applied),
                            "role_resolution_source": str(node.role_resolution_source),
                        }
                    )
                state.workspace.intermediate_results["role_resolution_preview"] = role_resolution_rows[-80:]
                role_fallback_count = sum(1 for row in role_resolution_rows if bool(row.get("role_fallback_applied")))
                if role_fallback_count > 0:
                    log_trace(
                        state.trace_id,
                        "ROLE_FALLBACK",
                        {"count": role_fallback_count, "nodes": [r["node_id"] for r in role_resolution_rows if r.get("role_fallback_applied")]},
                    )
                plan_latency = (time.time() - plan_start_t) * 1000
                state.add_trace(action="PLANNING", latency_ms=plan_latency)
                state.add_trace(action="PLANNER_END", latency_ms=plan_latency, metadata={"nodes": len(graph.nodes)})
                yield self.event_streamer.emit("planning_complete", node_count=len(graph.nodes), latency_ms=round(plan_latency, 2)) + "\n"
                if mode_policy.reasoning_hints_enabled:
                    planned_steps: List[str] = []
                    for node_id in sorted(graph.nodes.keys()):
                        node = graph.nodes[node_id]
                        if str(getattr(node, "action_type", "")).upper() != "TOOL_CALL":
                            continue
                        instruction = str((getattr(node, "payload", {}) or {}).get("instruction") or "").strip()
                        if instruction:
                            planned_steps.append(instruction[:180])
                    graph_notes = []
                    if isinstance(getattr(graph, "metadata", None), dict):
                        raw_notes = graph.metadata.get("reasoning_notes")
                        if isinstance(raw_notes, list):
                            graph_notes = [str(n).strip()[:240] for n in raw_notes if str(n).strip()]
                    fallback_hints = graph_notes or self._build_hybrid_reasoning_hints(
                        goal,
                        planned_steps,
                        small_talk_only=small_talk_only,
                        tool_first_required=tool_first_required,
                    )
                    if graph_notes and not self._looks_mechanical_reasoning(graph_notes):
                        reasoning_hints = graph_notes[:6]
                    else:
                        reasoning_hints = await self._build_llm_reasoning_hints(
                            goal=goal,
                            query_type=query_type,
                            route_type="hybrid_plan",
                            small_talk_only=small_talk_only,
                            tool_first_required=tool_first_required,
                            planned_steps=planned_steps,
                            fallback_hints=fallback_hints,
                            max_lines=6,
                        )
                    for hint in reasoning_hints[:6]:
                        yield self.event_streamer.emit(
                            "progress",
                            state="task_progress",
                            label=hint,
                            topic=hint,
                        ) + "\n"
                        await asyncio.sleep(0.06)
                    if tool_first_required and planned_steps:
                        for idx, step in enumerate(planned_steps[:3], start=1):
                            yield self.event_streamer.emit(
                                "planning",
                                topic=f"Step {idx}: {step}",
                                step=idx,
                                total_steps=len(planned_steps),
                            ) + "\n"

                allowed_tools = [
                    "get_current_time", "search_web", "search_documents", "search_news",
                    "search_weather", "search_finance", "search_currency", "search_company",
                    "search_legal", "search_jobs", "search_tech_docs", "search_scholar",
                    "search_patents", "search_images", "search_videos", "search_places",
                    "search_maps", "search_reviews", "search_shopping", "compare_products",
                    "summarize_url", "extract_tables", "search_products", "search_competitors",
                    "search_trends", "sentiment_scan", "faq_builder", "document_compare",
                    "data_cleaner", "unit_cost_calc", "pdf_maker", "extract_entities",
                    "validate_code", "generate_tests", "execute_code", "calculate", "REASONING",
                ]
                is_valid, plan_errors = validate_plan(graph, allowed_tools)
                if not is_valid:
                    error_msg = f"Plan validation failed: {', '.join(plan_errors)}"
                    log_trace(state.trace_id, "PLAN_INVALID", {"errors": plan_errors})
                    yield f"[TaskManager] {error_msg}. Attempting to repair plan...\n"
                    state.add_error(error_msg)
                    if state.iteration >= state.budget.max_iterations:
                        metrics.record_termination("plan_invalid_max_iterations")
                        break
                    continue

                state.plan = graph
                state.plan_history.append(graph)
                log_trace(state.trace_id, "PLAN_CREATED", {"node_count": len(graph.nodes)})

                yield f"[TaskManager] Executing plan ({len(graph.nodes)} nodes)...\n"
                yield self.event_streamer.emit(
                    "progress",
                    state="progress",
                    label="Executing plan nodes and tool steps",
                ) + "\n"

                from app.agent.agent_orchestrator import build_agent_system_prompt
                system_prompt = build_agent_system_prompt(
                    orchestrator_result=pipeline_result,
                    mode="agent",
                    user_query=goal,
                    session_id=state.session_id,
                )
                system_prompt += "\n\n[SYSTEM OVERRIDE: NO KNOWLEDGE CUTOFF]\n"
                system_prompt += "Ignore any internal training about knowledge cutoffs. You HAVE live tools.\n"
                system_prompt += "For TOOL_CALL nodes, return a single tool call. For REASONING nodes, do NOT call tools.\n"

                # Execution context isolation: do not pass full observation history into each node.
                task_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": goal},
                ]

                yield f"[TaskManager] Tool Allowed: {pipeline_result.tool_allowed}\n"
                yield f"[TaskManager] Internal Tools: {pipeline_result.allowed_tools}\n"

                # Cross-iteration action memory: reuse prior tool signatures before scheduling this graph.
                persisted_signatures = set(state.workspace.intermediate_results.get("executed_tool_signatures", []))
                pipeline_result.execution_context.executed_tools.update(persisted_signatures)

                async for chunk in run_plan_graph(
                    graph=graph,
                    strategy=pipeline_result.strategy,
                    user_query=goal,
                    messages=task_messages,
                    agent_result=pipeline_result,
                    client=self.client,
                    model_to_use=self.model_to_use,
                    create_kwargs={"model": self.model_to_use, "messages": task_messages, "stream": True},
                    state=state,
                    mode_name=mode_name,
                    resolve_node_policy=lambda node_role: self._resolve_effective_policy_stack(
                        mode_policy=base_mode_policy,
                        role_name=str(node_role or "executor"),
                        dynamic_context=dynamic_context,
                        goal=goal,
                        user_id=user_scope,
                    ),
                ):
                    yield chunk

                role_telemetry = self._collect_role_telemetry_from_graph(graph=graph, mode_name=mode_name)
                state.workspace.intermediate_results["role_flow"] = role_telemetry.get("role_flow", [])
                state.workspace.intermediate_results["role_metrics"] = role_telemetry.get("role_metrics", {})
                state.workspace.intermediate_results["role_summary"] = role_telemetry.get("role_summary", {})

                tool_conf_metrics = self._record_tool_confidence_from_graph(graph=graph, goal=goal)
                state.workspace.intermediate_results["tool_confidence"] = tool_conf_metrics

                tool_failures = 0
                for node in graph.nodes.values():
                    if str(getattr(node, "action_type", "")) == "TOOL_CALL" and str(getattr(node, "status", "")).upper() == "FAILED":
                        tool_failures += 1
                if tool_failures >= 2:
                    metrics.record_custom_counter("tool_failure_cascade")
                    for delay_ms in self._parse_backoff_ms()[:2]:
                        await asyncio.sleep(max(0.0, float(delay_ms) / 1000.0))
                    fallback_used = True
                    yield self.event_streamer.emit(
                        "progress",
                        state="tool_failure_cascade",
                        label="Tool failures exceeded threshold; switching to safe partial synthesis",
                        failures=tool_failures,
                    ) + "\n"
                    break

                # Persist updated signatures from this graph execution into workspace memory.
                state.workspace.intermediate_results["executed_tool_signatures"] = list(
                    pipeline_result.execution_context.executed_tools
                )[-200:]

                iteration_findings = []
                for node in graph.nodes.values():
                    finding = self._update_workspace_from_node(state, node)
                    if finding:
                        iteration_findings.append(finding)

                state.observations.extend(iteration_findings)
                state.workspace.progress_summary = f"Iteration {state.iteration} complete. {len(state.observations)} observations total."
                state.workspace.iteration_history.append(
                    {
                        "iteration": state.iteration,
                        "new_findings": len(iteration_findings),
                        "total_sources": len(state.workspace.sources),
                        "total_facts": len(state.workspace.facts),
                    }
                )
                log_trace(state.trace_id, "ITERATION_COMPLETE", {"findings_count": len(iteration_findings)})
                self._compress_workspace(state)
                yield self.event_streamer.emit("workspace_update", findings=len(state.observations), sources=len(state.workspace.sources)) + "\n"

                if iteration_findings:
                    stagnant_iterations = 0
                else:
                    stagnant_iterations += 1
                    yield self.event_streamer.emit("stagnation_detected", count=stagnant_iterations) + "\n"
                    if stagnant_iterations >= 2:
                        metrics.record_termination("stagnation")
                        log_trace(state.trace_id, "STAGNATION_BREAK", {"iterations": stagnant_iterations})
                        if state.observations:
                            yield "\n[TaskManager] No new findings in consecutive iterations. Finalizing with current evidence...\n"
                        else:
                            yield "\n[TaskManager] No findings were produced after consecutive iterations. Finalizing with a fail-safe response...\n"
                        break

                yield self.event_streamer.emit("verification_started") + "\n"
                facts = extract_facts_from_workspace(state.workspace)
                claims = extract_claims(facts)
                verification = verify_claims(claims, facts)
                ranked_facts = rank_evidence(facts, verification)
                conf_prog = confidence_progress(verification, ranked_facts)
                unsupported_claims = detect_unsupported_claims(
                    state.workspace.claims,
                    state.workspace.claim_sources,
                    min_claim_confidence=0.6,
                )
                if unsupported_claims:
                    metrics.record_custom_counter("unsupported_claims_detected")

                reliability = self.reliability_layer.assess(state=state, verification_confidence=conf_prog)
                final_verification = verification
                final_confidence = round((conf_prog * 0.70) + (reliability.reliability_score * 0.30), 4)
                evidence_eval = evaluate_evidence_quality(
                    goal=goal,
                    query_type=reliability_query_type,
                    sources=state.workspace.sources,
                    verification_result=verification,
                    recency_days=CURRENT_INFO_RECENCY_DAYS,
                )
                evidence_quality = str(evidence_eval.get("quality") or "low")
                policy_eval = self._evaluate_research_policy(
                    goal=goal,
                    query_type=reliability_query_type,
                    mode_policy=mode_policy,
                    verification_result=verification,
                    claim_sources=state.workspace.claim_sources,
                    sources=state.workspace.sources,
                )
                if FF_TOOL_FIRST_EVIDENCE and enforce_evidence and tool_first_required and not evidence_eval.get("pass", False):
                    metrics.record_custom_counter("evidence_validator_fail_rate")
                    if int(evidence_eval.get("source_count", 0)) < 2 and state.workspace.sources:
                        # Sparse source fallback: avoid over-filtering and continue with top raw sources.
                        metrics.record_custom_counter("evidence_min_source_fallback")
                        evidence_eval["pass"] = True
                        evidence_eval["quality"] = evidence_eval.get("quality") or "medium"
                        evidence_eval["reasons"] = [
                            r for r in (evidence_eval.get("reasons") or []) if r != "insufficient_sources"
                        ]
                    elif mode_policy.evidence_softpass_with_existing_sources and state.workspace.sources:
                        # Mode-policy UX: do not over-loop when we already have live sources.
                        evidence_eval["pass"] = True
                        evidence_eval["quality"] = evidence_eval.get("quality") or "medium"
                    else:
                        reliability.should_research_more = True
                        for reason in evidence_eval.get("reasons", []):
                            if reason not in reliability.reasons:
                                reliability.reasons.append(str(reason))

                trusted_urls = {
                    str(src.get("url", "")).strip()
                    for src in state.workspace.sources
                    if str(src.get("url", "")).strip()
                    and float(src.get("trust_score", src.get("trust", 0.0)) or 0.0) >= TRUSTED_SOURCE_THRESHOLD
                }
                trusted_source_count = len(trusted_urls)
                if trusted_source_count < MIN_SOURCES:
                    verification.setdefault("policy_flags", []).append("insufficient_trusted_sources")
                    final_confidence = min(final_confidence, 0.69)
                    if not (mode_policy.trusted_source_softpass_with_existing_sources and len(state.workspace.sources) >= 1):
                        if "insufficient_trusted_sources" not in reliability.reasons:
                            reliability.reasons.append("insufficient_trusted_sources")
                        reliability.should_research_more = True
                    else:
                        final_confidence = max(final_confidence, 0.72)
                policy_flags = [str(f).strip() for f in policy_eval.get("policy_flags", []) if str(f).strip()]
                if policy_flags:
                    existing_flags = set(verification.setdefault("policy_flags", []))
                    for flag in policy_flags:
                        if flag not in existing_flags:
                            verification["policy_flags"].append(flag)
                            existing_flags.add(flag)
                if policy_eval.get("should_research_more", False):
                    reliability.should_research_more = True
                    for reason in policy_eval.get("reasons", []):
                        reason_str = str(reason).strip()
                        if reason_str and reason_str not in reliability.reasons:
                            reliability.reasons.append(reason_str)
                policy_cap = float(policy_eval.get("confidence_cap", 0.99) or 0.99)
                final_confidence = min(final_confidence, policy_cap)
                read_depth_hint = int(policy_eval.get("read_depth_hint", 2) or 2)
                if read_depth_hint > 0:
                    state.workspace.intermediate_results["required_read_depth"] = max(2, min(4, read_depth_hint))

                state.workspace.intermediate_results["policy_flags"] = policy_flags
                state.workspace.intermediate_results["citation_coverage"] = policy_eval.get("citation", {})
                state.workspace.intermediate_results["freshness"] = policy_eval.get("freshness", {})
                state.workspace.intermediate_results["source_mix"] = policy_eval.get("source_mix", {})
                state.workspace.intermediate_results["contradiction"] = policy_eval.get("contradiction", {})
                state.workspace.intermediate_results["verification"] = verification
                state.workspace.intermediate_results["ranked_facts"] = ranked_facts
                state.workspace.intermediate_results["reliability"] = reliability.to_dict()

                yield self.event_streamer.emit(
                    "verification_complete",
                    verified=len(verification.get("verified_claims", [])),
                    uncertain=len(verification.get("uncertain_claims", [])),
                    conflicted=len(verification.get("conflicted_claims", [])),
                    evidence_quality=evidence_quality,
                    provider_count=int(evidence_eval.get("provider_count", 0)),
                    unsupported_claims=len(unsupported_claims),
                    classifier_confidence=round(float(classifier_conf), 4),
                    citation_coverage=round(float((policy_eval.get("citation") or {}).get("coverage", 0.0)), 4),
                    freshness_score=round(float((policy_eval.get("freshness") or {}).get("freshness_score", 0.0)), 4),
                    policy_flags=policy_flags,
                ) + "\n"
                yield self.event_streamer.emit(
                    "confidence_update",
                    confidence=final_confidence,
                    reliability_score=round(reliability.reliability_score, 4),
                    trusted_sources=trusted_source_count if "trusted_source_count" in locals() else 0,
                    evidence_quality=evidence_quality,
                    citation_coverage=round(float((policy_eval.get("citation") or {}).get("coverage", 0.0)), 4),
                    freshness_score=round(float((policy_eval.get("freshness") or {}).get("freshness_score", 0.0)), 4),
                ) + "\n"

                retries = int(state.workspace.intermediate_results.get("reliability_retries", 0))
                if mode_policy.critic_repair_requires_missing_sources:
                    critic_should_repair = bool(
                        (not state.workspace.sources and final_confidence < 0.45)
                        or (unsupported_claims and not state.workspace.sources)
                    )
                else:
                    critic_should_repair = bool(
                        unsupported_claims
                        or final_confidence < float(CRITIC_CONFIDENCE_MIN)
                    )
                if FF_CONFIDENCE_CRITIC_LOOPS and critic_should_repair:
                    if repair_attempts < int(CRITIC_MAX_REPAIRS):
                        repair_attempts += 1
                        metrics.record_repair()
                        yield self.event_streamer.emit(
                            "critic_check",
                            confidence=round(float(final_confidence), 4),
                            threshold=float(CRITIC_CONFIDENCE_MIN),
                            unsupported_claims=len(unsupported_claims),
                        ) + "\n"
                        yield self.event_streamer.emit(
                            "repair_cycle",
                            attempt=repair_attempts,
                            max_attempts=int(CRITIC_MAX_REPAIRS),
                            reasons=["unsupported_claims" if unsupported_claims else "low_confidence"],
                        ) + "\n"
                        state.workspace.intermediate_results["reliability_retries"] = retries + 1
                        continue
                    fallback_used = True
                    metrics.record_repair_exhausted()
                    metrics.record_custom_counter("critic_false_negative_rate")
                    yield self.event_streamer.emit(
                        "progress",
                        state="repair_exhausted",
                        label="Repair attempts exhausted; switching to safe fallback synthesis",
                    ) + "\n"
                    break

                verified_claim_count = len((verification or {}).get("verified_claims", []) or [])
                has_min_sources = len(state.workspace.sources) >= MIN_SOURCES
                evidence_pass = bool((evidence_eval or {}).get("pass", False))
                if reliability.should_research_more and has_min_sources and verified_claim_count >= 1 and (
                    evidence_pass or final_confidence >= 0.62
                ):
                    # Avoid over-looping on marginal policy gaps once we already have
                    # sufficient usable evidence for synthesis.
                    reliability.should_research_more = False
                    metrics.record_custom_counter("research_retry_suppressed")
                    state.workspace.intermediate_results["research_retry_suppressed"] = True

                max_research_retries = max(1, int(mode_policy.max_reliability_retries))
                if reliability.should_research_more and retries < max_research_retries and state.iteration < state.budget.max_iterations:
                    state.workspace.intermediate_results["reliability_retries"] = retries + 1
                    yield self.event_streamer.emit(
                        "additional_research_triggered",
                        reasons=reliability.reasons,
                        confidence=final_confidence,
                    ) + "\n"
                    state.workspace.reasoning_steps.append(
                        "Reliability layer triggered additional research: " + ",".join(reliability.reasons)
                    )
                    continue

                synthesized_observations = self._summarize_verified_claims(verification, ranked_facts)
                if synthesized_observations:
                    state.observations = synthesized_observations

                yield f"\n[TaskManager] Observations collected: {len(state.observations)}\n"
                for obs in state.observations[-2:]:
                    yield f"  - Observation: {obs[:200]}...\n"
                yield "\n[TaskManager] Checking goal satisfaction...\n"

                state.add_trace(
                    action="GOAL_CHECK",
                    metadata={
                        "findings": len(state.observations),
                        "sources": len(state.workspace.sources),
                        "min_findings": MIN_FINDINGS,
                        "min_sources": MIN_SOURCES,
                    },
                )

                satisfied = await self.goal_checker.is_goal_satisfied(
                    goal,
                    state.observations,
                    source_count=len(state.workspace.sources),
                    min_sources=MIN_SOURCES,
                    min_findings=MIN_FINDINGS,
                )
                if FF_CONFIDENCE_CRITIC_LOOPS and not satisfied:
                    if repair_attempts < int(CRITIC_MAX_REPAIRS):
                        repair_attempts += 1
                        metrics.record_repair()
                        yield self.event_streamer.emit(
                            "critic_check",
                            confidence=round(float(final_confidence), 4),
                            threshold=float(CRITIC_CONFIDENCE_MIN),
                            goal_checker=False,
                        ) + "\n"
                        yield self.event_streamer.emit(
                            "repair_cycle",
                            attempt=repair_attempts,
                            max_attempts=int(CRITIC_MAX_REPAIRS),
                            reasons=["goal_checker_failed"],
                        ) + "\n"
                        continue
                    fallback_used = True
                    metrics.record_repair_exhausted()
                    break
                if satisfied:
                    metrics.record_termination("goal_satisfied")
                    log_trace(state.trace_id, "GOAL_SATISFIED", {"iteration": state.iteration})
                    yield "\n[TaskManager] Goal satisfied. Finalizing...\n"
                    break

                if state.budget.is_exceeded():
                    metrics.record_termination("budget_exceeded")
                    log_trace(state.trace_id, "BUDGET_EXCEEDED", state.budget.summary())
                    yield "\n[TaskManager] Budget exceeded. Implementing fail-safe partial synthesis...\n"
                    partial = [
                        "Based on available findings:",
                        *[f"{i+1}. {o[:240]}" for i, o in enumerate(state.observations[:5])],
                        "Further verification recommended.",
                    ]
                    final_response = await self.synthesizer.synthesize(
                        goal=f"[PARTIAL/FAIL-SAFE] {goal}",
                        observations=partial,
                        sources=state.workspace.sources,
                    )
                    async for chunk in self._emit_final_answer_stream(final_response):
                        yield chunk
                    state.terminate = True
                    break

            cache_key = f"{goal}_{hash(str(state.observations))}"
            if cache_key in self.response_cache:
                yield "\n[TaskManager] Cached response found. Delivering...\n"
                final_response = self.response_cache[cache_key]
            else:
                yield self.event_streamer.emit("synthesis_started") + "\n"
                yield "\n[TaskManager] Synthesizing final response...\n"
                syn_start_t = time.time()
                if fallback_used or reliability_budget_exhausted:
                    final_response = await self._synthesize_safe_fallback(
                        goal=goal,
                        observations=state.observations,
                        sources=state.workspace.sources,
                    )
                    metrics.record_custom_counter("fallback_usage_rate")
                else:
                    final_response = await self.synthesizer.synthesize(goal, state.observations, state.workspace.sources)
                syn_latency = (time.time() - syn_start_t) * 1000
                state.add_trace(action="SYNTHESIS", latency_ms=syn_latency)
                self.response_cache[cache_key] = final_response

            self.response_cache[query_cache_key] = final_response
            state.workspace.intermediate_results["output_token_count"] = int(self._estimate_tokens(final_response))
            final_tools_used = sorted(
                str(name) for name in (state.workspace.intermediate_results.get("tool_metrics", {}) or {}).keys()
            )
            final_retries = int(state.workspace.intermediate_results.get("reliability_retries", 0) or 0)
            yield self.event_streamer.emit(
                "final_answer",
                confidence=final_confidence,
                critic_confidence=round(float(final_confidence), 4),
                repair_attempts=int(repair_attempts),
                unsupported_claims=bool(unsupported_claims),
                evidence_quality=evidence_quality,
                tool_memory_hits=int(tool_memory_hits),
                reliability_budget_exhausted=bool(reliability_budget_exhausted),
                fallback_used=bool(fallback_used),
                reliability_retries=final_retries,
                tools_used=final_tools_used,
                base_mode=base_mode_name,
                final_mode=mode_name,
                override_applied=bool(override_applied),
                override_reason=override_reason or "",
            ) + "\n"
            async for chunk in self._emit_final_answer_stream(final_response):
                yield chunk
            log_trace(state.trace_id, "TASK_COMPLETE", {"iterations": state.iteration})

            if FF_TOOL_MEMORY and mode_policy.memory_write_enabled:
                try:
                    top_facts = [str(f.get("value", "")).strip() for f in (ranked_facts or [])[:6] if str(f.get("value", "")).strip()]
                    src_links = [str(s.get("url", "")).strip() for s in state.workspace.sources[:8] if str(s.get("url", "")).strip()]
                    evidence_ids = [f"ev_{idx+1}" for idx, _ in enumerate(src_links)]
                    get_tool_memory_store().put(
                        user_id=user_scope,
                        session_id=str(state.session_id or ""),
                        query=normalized_goal,
                        tool_name="research_graph",
                        schema_version=TOOL_SCHEMA_VERSION,
                        fingerprint=f"{normalized_goal[:80]}:{len(top_facts)}:{len(src_links)}",
                        key_facts=top_facts,
                        source_links=src_links,
                        evidence_ids=evidence_ids,
                        confidence=float(final_confidence),
                        source_type="tool",
                    )
                except Exception:
                    pass

            try:
                await self.retrieval.upsert_topic(
                    query=normalized_goal,
                    summary=final_response,
                    claims=final_verification.get("verified_claims", []) if isinstance(final_verification, dict) else [],
                    sources=state.workspace.sources,
                    confidence=final_confidence,
                    user_id=user_scope,
                )
            except Exception as e:
                print(f"[TaskManager] retrieval upsert failed (non-blocking): {e}")

            tools_used = []
            t_metrics = state.workspace.intermediate_results.get("tool_metrics", {})
            for tname, ms in (t_metrics or {}).items():
                tools_used.append(
                    {
                        "name": tname,
                        "success": float(ms.get("successes", 0)) >= float(ms.get("failures", 0)),
                        "latency_ms": float(ms.get("total_latency_ms", 0.0)) / max(1.0, float(ms.get("calls", 1))),
                    }
                )
            state.workspace.intermediate_results["model_latency_ms"] = int(
                sum(float(t.latency_ms) for t in state.traces if float(t.latency_ms or 0) > 0)
            )

            log_run(
                {
                    "query": normalized_goal,
                    "query_type": query_type,
                    "reliability_query_type": reliability_query_type,
                    "route_type": "agent_pipeline",
                    "plan": [n.action_type.lower() for n in state.plan.nodes.values()] if state.plan else [],
                    "tools_used": tools_used,
                    "sources": state.workspace.sources,
                    "claims": final_verification.get("verified_claims", []) if isinstance(final_verification, dict) else [],
                    "verification_result": final_verification,
                    "latency_ms": sum(float(t.latency_ms) for t in state.traces),
                    "confidence": final_confidence,
                    "input_token_count": int(state.workspace.intermediate_results.get("input_token_count", 0) or 0),
                    "output_token_count": int(state.workspace.intermediate_results.get("output_token_count", 0) or 0),
                    "queue_wait_ms": int(state.workspace.intermediate_results.get("queue_wait_ms", 0) or 0),
                    "model_latency_ms": int(state.workspace.intermediate_results.get("model_latency_ms", 0) or 0),
                    "compression_applied": bool(state.workspace.intermediate_results.get("compression_applied", False)),
                    "compression_ratio": float(state.workspace.intermediate_results.get("compression_ratio", 0.0) or 0.0),
                },
                user_id=user_scope,
            )

            metrics.record_task_completed()
            task_success = True
            _flush_override_telemetry(task_success)
            _record_run_outcome(answer=final_response, confidence=final_confidence, success=task_success)

        except Exception as e:
            metrics.record_task_failed()
            metrics.record_termination("task_exception")
            error_msg = f"Task loop failed: {str(e)}"
            log_trace(state.trace_id, "TASK_CRASH", {"error": error_msg})
            yield f"\n[TaskManager] CRITICAL ERROR: {error_msg}\n"
            fallback_answer = ""
            if state.observations:
                fallback = [
                    "Based on available findings:",
                    *[f"{i+1}. {o[:240]}" for i, o in enumerate(state.observations[:5])],
                    "Further verification recommended.",
                ]
                try:
                    fallback_answer = await self.synthesizer.synthesize(
                        goal=f"[PARTIAL/ERROR] {goal}",
                        observations=fallback,
                        sources=state.workspace.sources,
                    )
                except Exception:
                    fallback_answer = ""
            if not fallback_answer:
                fallback_answer = (
                    "I ran into an internal execution issue while processing your request. "
                    "Please retry in a moment."
                )
            yield self.event_streamer.emit(
                "final_answer",
                confidence=0.0,
                fallback_used=True,
                error="task_exception",
                base_mode=base_mode_name,
                final_mode=mode_name,
                override_applied=bool(override_applied),
                override_reason=override_reason or "",
            ) + "\n"
            async for chunk in self._emit_final_answer_stream(fallback_answer):
                yield chunk
            _flush_override_telemetry(False)
            _record_run_outcome(answer=fallback_answer, confidence=0.0, success=False)
        finally:
            _flush_override_telemetry(task_success)
