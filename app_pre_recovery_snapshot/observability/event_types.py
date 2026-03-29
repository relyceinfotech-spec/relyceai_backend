"""
Phase 7: Event Types
Centralized typed event schema. No ad-hoc string events allowed.

Every telemetry event in the system MUST use one of these types.
This guarantees:
  - No typos in event names
  - IDE autocomplete
  - Aggregation safety
  - Schema evolution tracking
"""
from enum import Enum


class EventType(str, Enum):
    """Typed execution events for the telemetry layer."""

    # --- Node Lifecycle ---
    NODE_STARTED = "NODE_STARTED"
    NODE_COMPLETED = "NODE_COMPLETED"
    NODE_FAILED = "NODE_FAILED"

    # --- Repair Engine ---
    REPAIR_TRIGGERED = "REPAIR_TRIGGERED"
    REPAIR_EXHAUSTED = "REPAIR_EXHAUSTED"

    # --- Transaction Layer ---
    ROLLBACK_TRIGGERED = "ROLLBACK_TRIGGERED"

    # --- Sandbox Isolation ---
    SANDBOX_STARTED = "SANDBOX_STARTED"
    SANDBOX_COMPLETED = "SANDBOX_COMPLETED"
    SANDBOX_KILLED = "SANDBOX_KILLED"
    SANDBOX_CRASHED = "SANDBOX_CRASHED"

    # --- Strategy Engine ---
    STRATEGY_DRIFT = "STRATEGY_DRIFT"
    STRATEGY_SELECTED = "STRATEGY_SELECTED"

    # --- Memory Layer ---
    MEMORY_UPDATED = "MEMORY_UPDATED"
    MEMORY_DECAYED = "MEMORY_DECAYED"

    # --- Task Lifecycle ---
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_RESUMED = "TASK_RESUMED"

    # --- Anomaly Detection ---
    ANOMALY_ALERT = "ANOMALY_ALERT"

    # --- Research Layer ---
    RESEARCH_EXECUTED = "RESEARCH_EXECUTED"
    RESEARCH_CACHED = "RESEARCH_CACHED"
