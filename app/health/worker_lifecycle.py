"""
Phase 8: Worker Lifecycle
Tracks task execution count and signals when the worker should recycle.

Design rules:
  - Only signals recycle BETWEEN tasks — never mid-task
  - Recycle = graceful restart opportunity, not forced kill
  - Thread-safe counters
"""
from __future__ import annotations

import time
import threading
from typing import Optional


# Maximum tasks before signaling recycle
DEFAULT_MAX_TASKS = 500

# Maximum uptime before signaling recycle (seconds)
DEFAULT_MAX_UPTIME_SECONDS = 3600  # 1 hour


class WorkerLifecycle:
    """
    Tracks worker state and determines when recycling is needed.

    Usage:
        lifecycle = WorkerLifecycle()
        lifecycle.record_task()
        if lifecycle.should_recycle():
            graceful_restart()
    """

    def __init__(
        self,
        max_tasks: int = DEFAULT_MAX_TASKS,
        max_uptime_seconds: int = DEFAULT_MAX_UPTIME_SECONDS,
    ):
        self._lock = threading.Lock()
        self.max_tasks = max_tasks
        self.max_uptime_seconds = max_uptime_seconds
        self.tasks_processed: int = 0
        self.start_time: float = time.time()
        self.recycle_signaled: bool = False

    def record_task(self) -> None:
        """Record completion of one task."""
        with self._lock:
            self.tasks_processed += 1

    def should_recycle(self) -> bool:
        """
        Check if worker should be recycled.
        Based on task count OR uptime — whichever hits first.
        """
        with self._lock:
            if self.tasks_processed >= self.max_tasks:
                self.recycle_signaled = True
                return True

            uptime = time.time() - self.start_time
            if uptime >= self.max_uptime_seconds:
                self.recycle_signaled = True
                return True

            return False

    def get_state(self) -> dict:
        """Return lifecycle state snapshot."""
        with self._lock:
            return {
                "tasks_processed": self.tasks_processed,
                "uptime_seconds": round(time.time() - self.start_time, 2),
                "max_tasks": self.max_tasks,
                "max_uptime_seconds": self.max_uptime_seconds,
                "recycle_signaled": self.recycle_signaled,
            }

    def reset(self) -> None:
        """Reset counters (after recycle or for testing)."""
        with self._lock:
            self.tasks_processed = 0
            self.start_time = time.time()
            self.recycle_signaled = False


# ============================================
# MODULE SINGLETON
# ============================================

_default_lifecycle: Optional[WorkerLifecycle] = None


def get_worker_lifecycle() -> WorkerLifecycle:
    global _default_lifecycle
    if _default_lifecycle is None:
        _default_lifecycle = WorkerLifecycle()
    return _default_lifecycle
