"""
Phase 7: Execution Event Logger
Append-only in-memory circular buffer for structured telemetry events.

Design rules:
  - NEVER blocks execution
  - NEVER performs disk IO
  - NEVER calls external services
  - Constant-time O(1) append
  - Bounded memory via deque(maxlen)
  - Swappable to Redis/DB later without changing emitter interface
"""
from __future__ import annotations

import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional

from app.observability.event_types import EventType


# Maximum events retained in memory (circular buffer)
MAX_EVENT_BUFFER = 5000


class ExecutionEventLogger:
    """
    Thread-safe, append-only event logger with circular buffer.

    Usage:
        logger = ExecutionEventLogger()
        logger.emit(EventType.NODE_STARTED, {"task_id": "t1", "node_id": "N1"})
        recent = logger.get_recent(limit=50)
    """

    def __init__(self, max_buffer: int = MAX_EVENT_BUFFER):
        self._events: deque = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._max_buffer = max_buffer

    def emit(self, event_type: EventType, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit a structured telemetry event.

        Args:
            event_type: Must be a member of EventType enum.
            payload: Optional dictionary with event-specific data.
        """
        if not isinstance(event_type, EventType):
            raise TypeError(f"event_type must be EventType enum, got {type(event_type).__name__}")

        event = {
            "timestamp": time.time(),
            "event_type": event_type.value,
        }
        if payload:
            event["payload"] = payload

        with self._lock:
            self._events.append(event)

    def get_recent(self, limit: int = 100) -> List[Dict]:
        """Return the most recent N events."""
        with self._lock:
            items = list(self._events)
        return items[-limit:]

    def get_by_type(self, event_type: EventType, limit: int = 50) -> List[Dict]:
        """Filter events by type."""
        with self._lock:
            items = list(self._events)
        filtered = [e for e in items if e["event_type"] == event_type.value]
        return filtered[-limit:]

    def count(self) -> int:
        """Total events currently in buffer."""
        return len(self._events)

    def clear(self) -> None:
        """Flush all events (for testing or session reset)."""
        with self._lock:
            self._events.clear()


# ============================================
# MODULE SINGLETON
# ============================================

_default_logger: Optional[ExecutionEventLogger] = None


def get_event_logger() -> ExecutionEventLogger:
    """Returns the module-level singleton logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = ExecutionEventLogger()
    return _default_logger
