"""Universal response schema constants and helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

SCHEMA_VERSION = "1.0"
ALLOWED_BLOCK_TYPES = {"text", "list", "table", "timeline", "card"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def base_metadata(extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    meta = dict(extra or {})
    meta["generated_at"] = utc_now_iso()
    return meta
