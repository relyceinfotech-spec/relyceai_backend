from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


class StreamController:
    """Throttle and filter streaming info events to avoid UI flooding."""

    HIGH_PRIORITY_EVENTS = {
        "task_started",
        "planning_started",
        "planning_complete",
        "verification_started",
        "verification_complete",
        "additional_research_triggered",
        "confidence_update",
        "synthesis_started",
        "final_answer",
        "error",
    }

    HIGH_PRIORITY_STATES = {
        "initializing",
        "planning",
        "researching",
        "using_tool",
        "synthesizing",
        "finalizing",
        "completed",
        "cancelled",
    }

    SENSITIVE_KEYS = {
        "system_prompt",
        "prompt",
        "messages",
        "raw_messages",
        "internal_reasoning",
        "chain_of_thought",
        "cot",
        "developer_instructions",
        "policy_text",
        "full_context",
        "tool_args_raw",
    }

    def __init__(self, min_info_interval_s: float = 0.2):
        self.min_info_interval_s = float(min_info_interval_s)
        self._last_info_emit_ts = 0.0

    def _is_high_priority(self, payload: Dict[str, Any]) -> bool:
        event_name = str(payload.get("event") or "").strip().lower()
        if event_name and event_name in self.HIGH_PRIORITY_EVENTS:
            return True

        agent_state = str(payload.get("agent_state") or "").strip().lower()
        if agent_state and agent_state in self.HIGH_PRIORITY_STATES:
            return True

        if payload.get("completed") is True:
            return True

        if "followups" in payload or "action_chips" in payload:
            return True

        if payload.get("provider_retry") is True:
            return True

        return False

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for key, value in payload.items():
            if str(key).lower() in self.SENSITIVE_KEYS:
                continue
            if isinstance(value, str):
                # Keep streamed metadata compact and avoid accidental prompt leaks.
                safe[key] = value[:500]
            elif isinstance(value, (int, float, bool)) or value is None:
                safe[key] = value
            elif isinstance(value, list):
                safe[key] = value[:20]
            elif isinstance(value, dict):
                # Shallow dicts only for streaming metadata
                safe[key] = {k: (str(v)[:200] if isinstance(v, str) else v) for k, v in list(value.items())[:20]}
            else:
                safe[key] = str(value)[:200]
        return safe

    def filter_info(self, raw_info: str) -> Optional[str]:
        """
        Return info string if it should be emitted, else None.
        Accepts clean [INFO]-stripped payload.
        """
        now = time.monotonic()

        # Keep non-JSON tags like INTEL, RETRIEVED_CONTEXT throttled.
        if not raw_info.startswith("{"):
            if (now - self._last_info_emit_ts) < self.min_info_interval_s:
                return None
            self._last_info_emit_ts = now
            return raw_info

        try:
            payload = json.loads(raw_info)
        except Exception:
            if (now - self._last_info_emit_ts) < self.min_info_interval_s:
                return None
            self._last_info_emit_ts = now
            return raw_info

        safe_payload = self._sanitize_payload(payload)

        if self._is_high_priority(payload):
            self._last_info_emit_ts = now
            return json.dumps(safe_payload)

        if (now - self._last_info_emit_ts) < self.min_info_interval_s:
            return None

        self._last_info_emit_ts = now
        return json.dumps(safe_payload)

