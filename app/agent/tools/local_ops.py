from __future__ import annotations

import inspect

from app.agent.tool_registry import register_tool
from app.agent.tool_executor import _tool_calculate as _real_calculate
from app.agent.tool_executor import _tool_execute_code as _real_execute_code
from app.agent.tool_executor import _tool_extract_entities as _real_extract_entities
from app.agent.tool_executor import _tool_generate_tests as _real_generate_tests
from app.agent.tool_executor import _tool_get_current_time as _real_get_current_time
from app.agent.tool_executor import _tool_pdf_maker as _real_pdf_maker
from app.agent.tool_executor import _tool_read_file as _real_read_file
from app.agent.tool_executor import _tool_retrieve_knowledge as _real_retrieve_knowledge
from app.agent.tool_executor import _tool_validate_code as _real_validate_code


async def _call_tool(func, *args, **kwargs):
    """Call sync or async tool impls safely through one wrapper path."""
    out = func(*args, **kwargs)
    if inspect.isawaitable(out):
        return await out
    return out


@register_tool("get_current_time")
async def _tool_get_current_time(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_get_current_time, args=args)


@register_tool("pdf_maker")
async def _tool_pdf_maker(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_pdf_maker, args=args)


@register_tool("extract_entities")
async def _tool_extract_entities(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_extract_entities, args=args)


@register_tool("validate_code")
async def _tool_validate_code(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_validate_code, args=args)


@register_tool("generate_tests")
async def _tool_generate_tests(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_generate_tests, args=args)


@register_tool("execute_code")
async def _tool_execute_code(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_execute_code, args=args)


@register_tool("read_file")
async def _tool_read_file(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_read_file, args=args)


@register_tool("calculate")
async def _tool_calculate(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_calculate, args=args)


@register_tool("retrieve_knowledge")
async def _tool_retrieve_knowledge(args: str = "", session_id: str = "") -> dict:
    return await _call_tool(_real_retrieve_knowledge, args=args)
