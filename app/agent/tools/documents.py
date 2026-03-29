from __future__ import annotations

from app.agent.tool_registry import register_tool
from app.agent.tool_executor import _tool_search_documents as _real_search_documents
from app.agent.tool_executor import _tool_summarize_url as _real_summarize_url


@register_tool("search_documents")
async def _tool_search_documents(args: str = "", session_id: str = "", user_id: str = "") -> dict:
    return await _real_search_documents(args=args, user_id=user_id, session_id=session_id)


@register_tool("summarize_url")
async def _tool_summarize_url(args: str = "", session_id: str = "") -> dict:
    return await _real_summarize_url(args=args, session_id=session_id)
