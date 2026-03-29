from datetime import datetime, timezone
from typing import Dict, Optional

from app.auth import get_firestore_db


def log_routing_decision(user_id: Optional[str], decision: Dict) -> None:
    """
    Best-effort routing decision logger.
    Writes to Firestore if available, otherwise prints to stdout.
    """
    entry = {
        "user_id": user_id or "anonymous",
        "timestamp": datetime.now(timezone.utc),
        "time": datetime.now(timezone.utc).timestamp(),
        **decision
    }

    try:
        db = get_firestore_db()
        if db:
            db.collection("routingLogs").add(entry)
            return
    except Exception as e:
        print(f"[RoutingLog] Firestore write failed: {e}")

    print(f"[RoutingLog] {entry}")
