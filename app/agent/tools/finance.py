from __future__ import annotations

from app.agent.tool_registry import register_tool
from app.agent.tool_executor import _tool_search_currency as _real_search_currency


@register_tool("search_currency")
async def _tool_search_currency(args: str = "", session_id: str = "") -> dict:
    return await _real_search_currency(args=args, session_id=session_id)
