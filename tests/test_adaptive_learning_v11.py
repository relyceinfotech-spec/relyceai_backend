from __future__ import annotations

from dataclasses import replace

from app.agent.task_manager import TaskManager
from app.learning.adaptive_store import adaptive_bucket_key, normalize_for_fingerprint


class _NoopAdaptiveStore:
    def log_case(self, *, user_id: str, case_payload):
        return True

    def upsert_state(self, *, user_id: str, bucket_key: str, state_payload):
        return True

    def get_state(self, *, user_id: str, bucket_key: str):
        return None


class _CountingAdaptiveStore:
    def __init__(self):
        self.get_calls = 0
        self.upsert_calls = 0
        self.states = {}

    def log_case(self, *, user_id: str, case_payload):
        return True

    def upsert_state(self, *, user_id: str, bucket_key: str, state_payload):
        self.upsert_calls += 1
        self.states[f"{user_id}::{bucket_key}"] = dict(state_payload or {})
        return True

    def get_state(self, *, user_id: str, bucket_key: str):
        self.get_calls += 1
        return self.states.get(f"{user_id}::{bucket_key}")


def _new_manager() -> TaskManager:
    tm = TaskManager(client=None)
    tm.adaptive_store = _NoopAdaptiveStore()
    return tm


def test_fingerprint_normalization_strips_urls_numbers_and_spaces():
    raw = "  Latest update: https://example.com/news/2026?id=12   with  345   sources  "
    out = normalize_for_fingerprint(raw)
    assert "https://" not in out
    assert "example.com" not in out
    assert "345" not in out
    assert "  " not in out
    assert "<num>" in out


def test_bucket_key_is_stable_for_equivalent_queries():
    a = adaptive_bucket_key(
        input_text="Latest AI news 2026 please",
        intent="research",
        mode="smart",
    )
    b = adaptive_bucket_key(
        input_text="latest ai news 2027 please",
        intent="research",
        mode="smart",
    )
    assert a.split(":")[1:] == ["research", "smart"]
    assert b.split(":")[1:] == ["research", "smart"]
    assert a.split(":")[0] == b.split(":")[0]


def test_ema_math_alpha_point_two():
    out = TaskManager._ema_update(0.5, 1.0, alpha=0.2)
    assert round(out, 4) == 0.6


def test_adaptive_cooldown_and_step_cap():
    tm = _new_manager()
    tm.debug_config.update(
        {
            "adaptive_enabled": True,
            "adaptive_ema_alpha": 1.0,
            "adaptive_min_cluster_runs": 1,
            "adaptive_low_conf_threshold": 0.1,
            "adaptive_retry_threshold": 0.1,
            "adaptive_cooldown_runs": 3,
            "adaptive_max_bias_delta": 0.25,
            "failure_confidence_threshold": 0.45,
        }
    )
    payload = {
        "query": "latest ai chips in 2026",
        "mode": "smart",
        "detected_intent": "research",
        "retries": 3,
        "confidence": 0.1,
        "success": False,
        "user_id": "u1",
    }

    first = tm._update_adaptive_learning_state(payload=payload)
    first_bias = float(first["snapshot"]["bias_delta"])
    assert round(first_bias, 3) == 0.1

    second = tm._update_adaptive_learning_state(payload=payload)
    second_bias = float(second["snapshot"]["bias_delta"])
    assert second_bias == first_bias

    tm._update_adaptive_learning_state(payload=payload)
    fourth = tm._update_adaptive_learning_state(payload=payload)
    assert round(float(fourth["snapshot"]["bias_delta"]), 3) == 0.2
    assert float(fourth["snapshot"]["bias_delta"]) <= 0.25


def test_final_safety_clamp_preserves_mode_guarantees():
    base = TaskManager._resolve_mode_policy("research_pro")
    candidate = replace(
        base,
        tool_force_for_non_smalltalk=False,
        strict_evidence=False,
        require_multi_source=False,
        require_trusted_sources_only=False,
        min_reliability_budget_ms=max(1000, int(base.min_reliability_budget_ms) - 10000),
        max_reliability_retries=max(1, int(base.max_reliability_retries) - 1),
    )
    clamped, meta = TaskManager._final_safety_clamp(mode_policy=base, candidate_policy=candidate)

    assert clamped.tool_force_for_non_smalltalk is True
    assert clamped.strict_evidence is True
    assert clamped.require_multi_source is True
    assert clamped.require_trusted_sources_only is True
    assert int(clamped.min_reliability_budget_ms) >= int(base.min_reliability_budget_ms)
    assert int(clamped.max_reliability_retries) >= int(base.max_reliability_retries)
    assert meta["clamp_applied"] is True
    assert len(meta["clamp_fields"]) > 0


def test_mode_check_includes_adaptive_and_clamp_payload():
    tm = _new_manager()
    tm.debug_config["adaptive_enabled"] = False
    out = tm.get_mode_check(input_text="latest nvidia news 2026", requested_mode="auto")
    assert "adaptive_bucket_key" in out
    assert "adaptive_snapshot" in out
    assert "final_clamp" in out
    assert "role_policy_meta" in out


def test_mode_check_estimates_outcome_without_history():
    tm = _new_manager()
    out = tm.get_mode_check(input_text="latest ai chip news 2026", requested_mode="agent")
    assert out["outcome_source"] == "estimated"
    assert out["confidence"] in {"low", "medium", "high"}
    assert isinstance(out["retries"], int)
    assert isinstance(out["tool_calls"], int)
    assert isinstance(out["success"], bool)
    assert isinstance(out["role_flow"], list)
    assert len(out["role_flow"]) >= 2
    assert out["role_flow"][-1]["role"] == "synthesizer"


def test_mode_check_prefers_historical_outcome_when_available():
    tm = _new_manager()
    tm.run_history_records.append(
        {
            "query": "latest ai chip news 2026",
            "requested_mode": "agent",
            "confidence": 0.91,
            "retries": 2,
            "tool_count": 3,
            "success": True,
            "role_flow": [
                {"node_id": "P1", "role": "planner", "status": "COMPLETED"},
                {"node_id": "FINAL", "role": "synthesizer", "status": "COMPLETED"},
            ],
            "role_metrics": {"planner": {"runs": 1}, "synthesizer": {"runs": 1}},
        }
    )
    out = tm.get_mode_check(input_text="latest ai chip news 2026", requested_mode="agent")
    assert out["outcome_source"] == "historical"
    assert out["confidence"] == "high"
    assert out["retries"] == 2
    assert out["tool_calls"] == 3
    assert out["success"] is True
    assert out["role_flow"][0]["role"] == "planner"


def test_debug_snapshot_includes_adaptive_learning_block():
    tm = _new_manager()
    tm._record_debug_run(
        {
            "query": "latest ai war update",
            "answer_preview": "preview",
            "requested_mode": "smart",
            "requested_mode_raw": "smart",
            "mode": "smart",
            "detected_intent": "research",
            "override_applied": False,
            "override_reason": "",
            "tool_count": 1,
            "retries": 2,
            "response_time_ms": 1200,
            "confidence": 0.2,
            "success": False,
            "user_id": "u1",
        }
    )
    snap = tm.get_debug_snapshot(range_window="24h", limit=20)
    assert "adaptive_learning" in snap
    adaptive = snap["adaptive_learning"]
    assert "top_failing_buckets" in adaptive
    assert "low_confidence_clusters" in adaptive
    assert "high_retry_queries" in adaptive
    assert "rule_events_recent" in adaptive


def test_adaptive_cache_ttl_avoids_repeated_firestore_reads():
    tm = _new_manager()
    store = _CountingAdaptiveStore()
    tm.adaptive_store = store
    tm.debug_config["adaptive_state_cache_ttl_sec"] = 120

    key = adaptive_bucket_key(input_text="latest ai news 2026", intent="research", mode="smart")
    store.states[f"u1::{key}"] = {"bucket_key": key, "runs": 4}

    first = tm._get_adaptive_bucket_state(user_id="u1", bucket_key=key)
    second = tm._get_adaptive_bucket_state(user_id="u1", bucket_key=key)

    assert int(first.get("runs", 0)) == 4
    assert int(second.get("runs", 0)) == 4
    assert store.get_calls == 1


def test_adaptive_state_write_throttling_skips_non_important_runs():
    tm = _new_manager()
    store = _CountingAdaptiveStore()
    tm.adaptive_store = store
    tm.debug_config.update(
        {
            "adaptive_enabled": True,
            "adaptive_ema_alpha": 0.2,
            "adaptive_state_write_every_runs": 3,
            "failure_confidence_threshold": 0.2,
        }
    )

    payload = {
        "query": "general informative query",
        "mode": "smart",
        "detected_intent": "general",
        "retries": 0,
        "confidence": 0.95,
        "success": True,
        "user_id": "u1",
    }

    out1 = tm._update_adaptive_learning_state(payload=payload)
    out2 = tm._update_adaptive_learning_state(payload=payload)
    out3 = tm._update_adaptive_learning_state(payload=payload)

    assert out1.get("state_persisted") is True
    assert out2.get("state_persisted") is False
    assert out3.get("state_persisted") is True
    assert store.upsert_calls == 2


def test_adaptive_enabled_toggle_takes_effect_immediately():
    tm = _new_manager()
    tm.debug_config["adaptive_enabled"] = False
    out = tm._update_adaptive_learning_state(
        payload={
            "query": "latest ai update",
            "mode": "smart",
            "detected_intent": "research",
            "retries": 2,
            "confidence": 0.1,
            "success": False,
            "user_id": "u1",
        }
    )
    assert out.get("adaptive_enabled") is False


def test_canary_ratio_zero_blocks_adaptive_policy_application():
    tm = _new_manager()
    tm.debug_config["adaptive_enabled"] = True
    tm.debug_config["adaptive_apply_ratio"] = 0.0
    mode_policy = TaskManager._resolve_mode_policy("smart")
    intent = "research"
    goal = "latest ai chip updates"
    bucket_key = tm._adaptive_bucket_key(input_text=goal, intent=intent, mode=mode_policy.mode_name)
    cache_key = tm._adaptive_cache_key(user_id="u1", bucket_key=bucket_key)
    tm.adaptive_learning_state[cache_key] = {
        "bucket_key": bucket_key,
        "policy_delta": {"budget_bonus_ms": 15000, "retry_bonus": 1, "force_tools": True},
    }
    tm.adaptive_state_loaded_keys.add(cache_key)

    out_policy, meta = tm._apply_adaptive_learning_overrides(
        policy=mode_policy,
        dynamic_context={"intent": intent},
        goal=goal,
        user_id="u1",
    )

    assert int(out_policy.min_reliability_budget_ms) == int(mode_policy.min_reliability_budget_ms)
    assert int(out_policy.max_reliability_retries) == int(mode_policy.max_reliability_retries)
    assert meta.get("rollout_applied") is False
    assert meta.get("applied") is False


def test_adaptive_snapshot_and_rollback_restores_state():
    tm = _new_manager()
    tm.adaptive_learning_state["u1::k1"] = {"bucket_key": "k1", "bias_delta": 0.1}
    snap = tm.create_adaptive_snapshot(label="test")
    tm.adaptive_learning_state["u1::k1"] = {"bucket_key": "k1", "bias_delta": 0.9}

    restored = tm.rollback_latest_adaptive_snapshot()

    assert snap.get("snapshot_id")
    assert restored.get("restored") is True
    assert float(tm.adaptive_learning_state["u1::k1"]["bias_delta"]) == 0.1


def test_auto_remediation_reduces_apply_ratio_on_parallel_exception_spike():
    tm = _new_manager()
    tm.debug_config.update(
        {
            "auto_remediation_enabled": True,
            "auto_remediation_min_runs": 1,
            "auto_remediation_cooldown_runs": 1,
            "auto_remediation_step": 0.1,
            "auto_remediation_min_apply_ratio": 0.1,
            "adaptive_apply_ratio": 0.5,
            "slo_parallel_exception_rate_max": 0.2,
        }
    )
    tm._record_debug_run(
        {
            "query": "latest ai chip news",
            "answer_preview": "preview",
            "requested_mode": "smart",
            "requested_mode_raw": "smart",
            "mode": "smart",
            "detected_intent": "research",
            "override_applied": False,
            "override_reason": "",
            "tool_count": 1,
            "retries": 0,
            "response_time_ms": 900,
            "confidence": 0.9,
            "success": True,
            "user_id": "u1",
            "role_metrics": {
                "researcher": {
                    "runs": 1,
                    "success_count": 1,
                    "latency_ms_total": 100,
                    "retries_total": 0,
                    "tool_calls_total": 1,
                    "parallel_exception_count": 1,
                }
            },
        }
    )
    assert float(tm.debug_config.get("adaptive_apply_ratio", 1.0)) == 0.4
    assert len(tm.auto_remediation_events) >= 1


def test_debug_snapshot_includes_canary_compare_and_auto_remediation():
    tm = _new_manager()
    tm.debug_config["auto_remediation_enabled"] = False
    tm._record_debug_run(
        {
            "query": "baseline query",
            "answer_preview": "baseline",
            "requested_mode": "smart",
            "requested_mode_raw": "smart",
            "mode": "smart",
            "detected_intent": "research",
            "override_applied": False,
            "override_reason": "",
            "tool_count": 1,
            "retries": 1,
            "response_time_ms": 1500,
            "confidence": 0.7,
            "success": True,
            "user_id": "u1",
            "adaptive_policy": {
                "bucket_key": "b1",
                "rollout_applied": False,
                "apply_ratio": 0.2,
                "applied": False,
            },
            "role_metrics": {"researcher": {"runs": 1, "parallel_exception_count": 0}},
        }
    )
    tm._record_debug_run(
        {
            "query": "adaptive query",
            "answer_preview": "adaptive",
            "requested_mode": "smart",
            "requested_mode_raw": "smart",
            "mode": "smart",
            "detected_intent": "research",
            "override_applied": False,
            "override_reason": "",
            "tool_count": 1,
            "retries": 0,
            "response_time_ms": 1000,
            "confidence": 0.85,
            "success": True,
            "user_id": "u1",
            "adaptive_policy": {
                "bucket_key": "b2",
                "rollout_applied": True,
                "apply_ratio": 0.2,
                "applied": True,
            },
            "role_metrics": {"researcher": {"runs": 1, "parallel_exception_count": 0}},
        }
    )
    snap = tm.get_debug_snapshot(range_window="24h", limit=20)
    assert "auto_remediation" in snap
    assert "canary_compare" in snap.get("adaptive_learning", {})
    assert int(snap["adaptive_learning"]["canary_compare"]["baseline"]["runs"]) >= 1
    assert int(snap["adaptive_learning"]["canary_compare"]["adaptive"]["runs"]) >= 1


def test_auto_remediation_respects_cooldown_window():
    tm = _new_manager()
    tm.debug_config.update(
        {
            "auto_remediation_enabled": True,
            "auto_remediation_min_runs": 1,
            "auto_remediation_cooldown_runs": 5,
            "auto_remediation_step": 0.1,
            "auto_remediation_min_apply_ratio": 0.1,
            "adaptive_apply_ratio": 0.6,
            "slo_parallel_exception_rate_max": 0.2,
        }
    )

    run_payload = {
        "query": "latest ai chip reliability",
        "answer_preview": "preview",
        "requested_mode": "agent",
        "requested_mode_raw": "agent",
        "mode": "agent",
        "detected_intent": "research",
        "override_applied": False,
        "override_reason": "",
        "tool_count": 1,
        "retries": 0,
        "response_time_ms": 1000,
        "confidence": 0.9,
        "success": True,
        "user_id": "u1",
        "role_metrics": {
            "researcher": {
                "runs": 1,
                "success_count": 1,
                "latency_ms_total": 100,
                "retries_total": 0,
                "tool_calls_total": 1,
                "parallel_exception_count": 1,
            }
        },
    }

    tm._record_debug_run(dict(run_payload))
    first_ratio = float(tm.debug_config.get("adaptive_apply_ratio", 1.0))
    first_events = len(tm.auto_remediation_events)
    assert first_ratio == 0.5
    assert first_events == 1

    tm._record_debug_run(dict(run_payload))
    second_ratio = float(tm.debug_config.get("adaptive_apply_ratio", 1.0))
    second_events = len(tm.auto_remediation_events)
    assert second_ratio == first_ratio
    assert second_events == first_events


def test_auto_remediation_event_is_attached_to_run_payload():
    tm = _new_manager()
    tm.debug_config.update(
        {
            "auto_remediation_enabled": True,
            "auto_remediation_min_runs": 1,
            "auto_remediation_cooldown_runs": 1,
            "auto_remediation_step": 0.1,
            "adaptive_apply_ratio": 0.5,
            "slo_parallel_exception_rate_max": 0.2,
        }
    )
    tm._record_debug_run(
        {
            "query": "latest ai chip reliability",
            "answer_preview": "preview",
            "requested_mode": "smart",
            "requested_mode_raw": "smart",
            "mode": "smart",
            "detected_intent": "research",
            "override_applied": False,
            "override_reason": "",
            "tool_count": 1,
            "retries": 0,
            "response_time_ms": 1000,
            "confidence": 0.9,
            "success": True,
            "user_id": "u1",
            "role_metrics": {
                "researcher": {
                    "runs": 1,
                    "parallel_exception_count": 1,
                }
            },
        }
    )
    latest = dict(list(tm.run_history_records)[-1])
    assert "auto_remediation" in latest
    assert isinstance(latest["auto_remediation"].get("actions", []), list)


def test_update_debug_config_accepts_auto_remediation_knobs():
    tm = _new_manager()
    updated = tm.update_debug_config(
        {
            "auto_remediation_enabled": False,
            "auto_remediation_min_runs": 30,
            "auto_remediation_cooldown_runs": 12,
            "auto_remediation_step": 0.08,
            "auto_remediation_min_apply_ratio": 0.2,
            "auto_remediation_low_conf_trigger": 0.5,
            "auto_remediation_retry_step_cap": 3,
            "auto_remediation_window_runs": 220,
        }
    )
    assert bool(updated.get("auto_remediation_enabled")) is False
    assert int(updated.get("auto_remediation_min_runs", 0)) == 30
    assert int(updated.get("auto_remediation_cooldown_runs", 0)) == 12
    assert round(float(updated.get("auto_remediation_step", 0.0)), 2) == 0.08
    assert round(float(updated.get("auto_remediation_min_apply_ratio", 0.0)), 2) == 0.2
    assert round(float(updated.get("auto_remediation_low_conf_trigger", 0.0)), 2) == 0.5
    assert int(updated.get("auto_remediation_retry_step_cap", 0)) == 3
    assert int(updated.get("auto_remediation_window_runs", 0)) == 220
