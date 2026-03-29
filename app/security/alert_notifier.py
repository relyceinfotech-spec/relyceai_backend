from __future__ import annotations

import json
import os
import time
from typing import Any, Dict
from urllib import request


_LAST_ALERT_SENT: Dict[str, float] = {}


def _alert_key(payload: Dict[str, Any]) -> str:
    range_key = str(payload.get("range") or "24h")
    codes = []
    for item in payload.get("alerts") or []:
        if isinstance(item, dict):
            code = str(item.get("code") or "").strip().upper()
            if code:
                codes.append(code)
    if not codes:
        codes = ["GENERIC"]
    return f"{range_key}:{'|'.join(sorted(set(codes)))}"


def send_admin_alert(payload: Dict[str, Any]) -> bool:
    url = str(os.getenv("HS_ALERT_WEBHOOK_URL") or "").strip()
    if not url:
        return False

    cooldown = int(float(os.getenv("HS_ALERT_COOLDOWN_SECONDS", "600") or "600"))
    key = _alert_key(payload or {})
    now = float(time.time())
    last_sent = float(_LAST_ALERT_SENT.get(key, 0.0))
    if (now - last_sent) < max(0, cooldown):
        return False

    body = json.dumps(payload or {}, separators=(",", ":"), default=str).encode("utf-8")
    req = request.Request(url=url, data=body, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10):
        pass

    _LAST_ALERT_SENT[key] = now
    return True

