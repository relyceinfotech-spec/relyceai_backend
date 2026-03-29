import asyncio

from app.api.routes import ops_admin
from app.platform import get_task_queue

ADMIN_USER = {
    "uid": "u1",
    "claims": {"role": "admin", "admin": True, "superadmin": False},
}


def test_feature_flags_roundtrip():
    out = asyncio.run(ops_admin.update_feature_flags(payload={"flags": {"new_routing": True}}, user_info=ADMIN_USER))
    assert out["success"] is True
    got = asyncio.run(ops_admin.get_feature_flags(user_info=ADMIN_USER))
    assert got["flags"].get("new_routing") is True


def test_queue_pause_resume_controls():
    q = get_task_queue()
    out_pause = asyncio.run(ops_admin.queue_control(payload={"action": "pause"}, user_info=ADMIN_USER))
    assert out_pause["success"] is True
    assert out_pause["queue"]["paused"] is True

    out_resume = asyncio.run(ops_admin.queue_control(payload={"action": "resume"}, user_info=ADMIN_USER))
    assert out_resume["success"] is True
    assert out_resume["queue"]["paused"] is False


def test_export_sign_verify_sha256():
    envelope = {"id": "abc", "ts": 123, "event": "x"}
    import json, hashlib
    canonical = json.dumps(envelope, sort_keys=True, separators=(",", ":"))
    sig = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    out = asyncio.run(ops_admin.export_sign_verify(payload={"envelope": envelope, "signature": sig, "algo": "sha256"}, user_info=ADMIN_USER))
    assert out["success"] is True
    assert out["valid"] is True


def test_policy_lint_runs():
    out = asyncio.run(ops_admin.policy_lint(user_info=ADMIN_USER))
    assert out["success"] is True
    assert isinstance(out["findings"], list)

import pytest
from fastapi import HTTPException


def test_inspect_run_includes_hardening_fields(monkeypatch):
    class _Task:
        status = "completed"
        created_at = 100.0
        started_at = 101.0
        completed_at = 105.0
        request = {"query": "x"}

    class _Queue:
        async def get_task(self, run_id):
            return _Task()

        async def get_events(self, run_id, after_seq=0):
            return [{"seq": 1, "type": "TOOL"}]

    monkeypatch.setattr(ops_admin, "get_task_queue", lambda: _Queue())
    monkeypatch.setattr(
        ops_admin,
        "_load_trace_entries",
        lambda run_id: [
            {"stage": "PLAN_CREATED", "detail": {"selected_tools": ["memory_query"]}},
            {"stage": "TOOL_RESULT", "detail": {"tool": "memory_query", "status": "success"}},
            {"stage": "TASK_COMPLETE", "detail": "done"},
        ],
    )

    out = asyncio.run(ops_admin.inspect_run("run_1", user_info=ADMIN_USER))
    assert out["success"] is True
    assert isinstance(out["tool_selection"], list)
    assert isinstance(out["tool_calls"], list)
    assert isinstance(out["tool_results"], list)
    assert "run_duration_ms" in out
    assert "created_to_start_ms" in out


def test_replay_trace_only_default(monkeypatch):
    class _Queue:
        async def get_task(self, run_id):
            return None

    monkeypatch.setattr(ops_admin, "get_task_queue", lambda: _Queue())
    monkeypatch.setattr(ops_admin, "_load_trace_entries", lambda run_id: [{"stage": "PLAN_CREATED", "detail": {}}])

    out = asyncio.run(ops_admin.replay_run("run_1", payload={}, user_info=ADMIN_USER))
    assert out["success"] is True
    assert out["mode"] == "trace_only"
    assert out["side_effects"] is False
    assert out["replayed"] is False


def test_replay_full_execution_requires_explicit_flag(monkeypatch):
    class _Queue:
        async def get_task(self, run_id):
            return None

    monkeypatch.setattr(ops_admin, "get_task_queue", lambda: _Queue())
    monkeypatch.setattr(ops_admin, "_load_trace_entries", lambda run_id: [])

    with pytest.raises(HTTPException) as exc:
        asyncio.run(ops_admin.replay_run("run_1", payload={"mode": "full_execution"}, user_info=ADMIN_USER))
    assert exc.value.status_code == 403


def test_inspect_run_requires_admin():
    non_admin = {"uid": "u2", "claims": {"role": "user", "admin": False, "superadmin": False}}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ops_admin.inspect_run("run_1", user_info=non_admin))
    assert exc.value.status_code == 403

def test_admin_dry_run_returns_preview(monkeypatch):
    monkeypatch.setattr(ops_admin, "_audit", lambda *args, **kwargs: None)
    out = asyncio.run(
        ops_admin.dry_run_agent(
            payload={"query": "latest nvidia ai news", "mode": "agent", "top_k": 3},
            user_info=ADMIN_USER,
        )
    )
    assert out["success"] is True
    assert out["dry_run"] is True
    assert out["side_effects"] is False
    assert isinstance(out["selected_tools"], list)
    assert "token_estimate" in out
