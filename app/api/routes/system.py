from __future__ import annotations

from typing import Any, Dict

from app.governance.spend_guard import get_spend_guard
from app.platform import get_task_queue


async def capabilities() -> Dict[str, Any]:
    queue = get_task_queue()
    queue_summary = {"paused": False, "queue_depth": 0}
    if hasattr(queue, "summary"):
        try:
            queue_summary = await queue.summary()
        except Exception:
            queue_summary = {"paused": False, "queue_depth": 0}

    spend = get_spend_guard().get_status()
    tools = ["search_web", "search_news", "memory_query", "task_planner", "task_executor_loop"]
    return {
        "status": "ok",
        "tools": {"count": len(tools), "items": tools},
        "modes": ["smart", "agent", "deepsearch", "research_pro"],
        "queue": queue_summary,
        "memory": {"query_enabled": True, "upsert_enabled": True},
        "spend": spend,
    }


async def health() -> Dict[str, str]:
    return {"status": "ok"}

