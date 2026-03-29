from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

_POINTERS: Dict[str, List[Dict[str, Any]]] = {}
_PAYLOADS: Dict[tuple[str, str], Any] = {}


def store_tool_result_pointer(
    *,
    session_id: str,
    tool_name: str,
    status: str,
    summary: str,
    payload: Any,
    source: str,
) -> Dict[str, Any]:
    sid = str(session_id or "default")
    result_id = f"res_{uuid.uuid4().hex[:10]}"
    row = {
        "result_id": result_id,
        "tool": str(tool_name or ""),
        "status": str(status or ""),
        "summary": str(summary or ""),
        "source": str(source or ""),
        "ts": int(time.time()),
    }
    _POINTERS.setdefault(sid, []).append(row)
    _PAYLOADS[(sid, result_id)] = payload
    return row


def list_result_pointers(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    sid = str(session_id or "default")
    rows = list(_POINTERS.get(sid, []))
    return rows[-int(limit) :]


def get_result_payload(session_id: str, result_id: str) -> Optional[Any]:
    sid = str(session_id or "default")
    rid = str(result_id or "")
    return _PAYLOADS.get((sid, rid))

