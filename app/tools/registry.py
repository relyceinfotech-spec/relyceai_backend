from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional


ToolFn = Callable[..., Awaitable[Dict[str, Any]]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, handler: ToolFn) -> None:
        key = str(name or "").strip()
        if not key:
            raise ValueError("tool name is required")
        self._tools[key] = handler

    def get(self, name: str) -> Optional[ToolFn]:
        return self._tools.get(str(name or "").strip())

    def list(self) -> Dict[str, ToolFn]:
        return dict(self._tools)


registry = ToolRegistry()

