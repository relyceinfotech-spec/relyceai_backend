from __future__ import annotations

import json
from typing import Any, Dict

from app.response_system.schema import ALLOWED_BLOCK_TYPES


class EventStreamer:
    """Formats agent lifecycle events for SSE/WebSocket streaming."""

    def __init__(self, prefix: str = "[INFO]"):
        self.prefix = prefix
        self._seq = 0

    def emit(self, event: str, **payload: Any) -> str:
        self._seq += 1
        body: Dict[str, Any] = {"event": event, "seq": self._seq}
        body.update(payload)
        return f"{self.prefix}{json.dumps(body)}"

    def emit_block_update(self, block: Dict[str, Any]) -> str:
        """Emit incremental block updates with strict block-type validation."""
        btype = str((block or {}).get("type") or "").strip().lower()
        if btype not in ALLOWED_BLOCK_TYPES:
            return self.emit("warning", message=f"invalid_block_type:{btype or 'unknown'}")
        return self.emit("block_update", block=block)
