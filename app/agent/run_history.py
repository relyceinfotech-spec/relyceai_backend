"""
Run History compatibility wrapper.

Canonical implementation lives in app.learning.run_history.
"""
from __future__ import annotations

from app.learning.run_history import RUN_COLLECTION, get_recent_runs, log_run

__all__ = ["RUN_COLLECTION", "log_run", "get_recent_runs"]
