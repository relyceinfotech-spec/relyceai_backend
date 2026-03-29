from __future__ import annotations

from typing import AsyncGenerator

from app.platform.types import CapabilityRequest, CapabilityResponse, CapabilityServiceProtocol
from app.platform.ai_gateway import AIGateway
from app.chat.mode_mapper import normalize_chat_mode


class BaseCapabilityService(CapabilityServiceProtocol):
    name = "base"
    default_mode = "smart"

    def __init__(self, gateway: AIGateway):
        self.gateway = gateway

    def _resolve_mode(self, request: CapabilityRequest) -> str:
        requested_mode = normalize_chat_mode(str(request.chat_mode or ""))
        if requested_mode and requested_mode != "smart":
            return requested_mode
        return self.default_mode

    async def run(self, request: CapabilityRequest) -> CapabilityResponse:
        result = await self.gateway.generate(
            user_query=request.user_query,
            mode=self._resolve_mode(request),
            context_messages=request.context_messages,
            personality=request.personality,
            user_settings=request.user_settings,
            user_id=request.user_id,
            session_id=request.session_id,
            file_ids=request.file_ids,
        )
        return CapabilityResponse(capability=self.name, payload=result)

    async def run_stream(self, request: CapabilityRequest) -> AsyncGenerator[str, None]:
        async for token in self.gateway.stream(
            user_query=request.user_query,
            mode=self._resolve_mode(request),
            context_messages=request.context_messages,
            personality=request.personality,
            user_settings=request.user_settings,
            user_id=request.user_id,
            session_id=request.session_id,
            task_id=request.task_id,
            resume_graph=request.resume_graph,
            file_ids=request.file_ids,
        ):
            yield token
