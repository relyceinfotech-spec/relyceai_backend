from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional


@dataclass
class CapabilityRequest:
    user_query: str
    chat_mode: str = "smart"
    context_messages: Optional[List[Dict[str, Any]]] = None
    personality: Optional[Dict[str, Any]] = None
    user_settings: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    resume_graph: Optional[Any] = None
    capability_hint: Optional[str] = None
    file_ids: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityResponse:
    capability: str
    payload: Dict[str, Any]


class CapabilityServiceProtocol:
    name: str = "unknown"

    async def run(self, request: CapabilityRequest) -> CapabilityResponse:
        raise NotImplementedError

    async def run_stream(self, request: CapabilityRequest) -> AsyncGenerator[str, None]:
        raise NotImplementedError
