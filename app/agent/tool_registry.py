from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Optional


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Coroutine[Any, Any, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized = str(name or "").strip()
        if not normalized:
            raise ValueError("Tool name cannot be empty")
        existing = self._tools.get(normalized)
        if existing is not None and existing is not func:
            raise ValueError(f"Duplicate tool registration attempted for '{normalized}'")

        self._tools[normalized] = func
        self._metadata[normalized] = metadata or {
            "cost_tier": "low",
            "avg_latency": 1.0,
            "is_sandboxed": False,
        }

    def get_tool(self, name: str) -> Optional[Callable[..., Coroutine[Any, Any, Any]]]:
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self._metadata.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._tools


registry = ToolRegistry()


def register_tool(name: str, **metadata):
    """Decorator to register a tool function with optional metadata."""

    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]):
        registry.register(name, func, metadata)
        return func

    return decorator
