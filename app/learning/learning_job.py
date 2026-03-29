from __future__ import annotations

import os
from typing import Dict, Any

from app.agent.daily_learning import run_daily_learning


def run_learning_job() -> Dict[str, Any]:
    enabled = str(os.getenv("LEARNING_JOB_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return {"updated": False, "reason": "disabled_by_flag"}
    try:
        result = run_daily_learning(user_id="global", activate=True)
        return {"updated": bool(result.get("updated")), "reason": "ok", "result": result}
    except Exception as exc:
        return {"updated": False, "reason": f"learning_job_failed:{exc.__class__.__name__}"}
