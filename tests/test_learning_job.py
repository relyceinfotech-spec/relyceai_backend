import os

from app.learning.learning_job import run_learning_job


def test_learning_job_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LEARNING_JOB_ENABLED", raising=False)
    out = run_learning_job()
    assert out["updated"] is False
    assert out["reason"] == "disabled_by_flag"
