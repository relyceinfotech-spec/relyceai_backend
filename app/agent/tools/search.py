from __future__ import annotations

from app.agent.tool_registry import register_tool
from app.agent.tool_executor import _tool_search_news as _real_search_news
from app.agent.tool_executor import _tool_search_web as _real_search_web


@register_tool("search_web")
async def _tool_search_web(args: str = "", session_id: str = "") -> dict:
    return await _real_search_web(args=args, session_id=session_id)


@register_tool("search_news")
async def _tool_search_news(args: str = "", session_id: str = "") -> dict:
    return await _real_search_news(args=args, session_id=session_id)
