from __future__ import annotations

import time
from typing import AsyncGenerator, Dict

from app.capabilities.automation.service import AutomationCapabilityService
from app.capabilities.business.service import BusinessCapabilityService
from app.capabilities.chat.service import ChatCapabilityService
from app.capabilities.coding.service import CodingCapabilityService
from app.capabilities.documents.service import DocumentsCapabilityService
from app.capabilities.research.service import ResearchCapabilityService
from app.observability.metrics_collector import get_metrics_collector
from app.platform.ai_gateway import AIGateway
from app.platform.capability_router import CapabilityRouter
from app.platform.types import CapabilityRequest, CapabilityResponse, CapabilityServiceProtocol


class AIPlatformService:
    """Capability platform orchestrator over shared AI infrastructure."""

    def __init__(self):
        self.gateway = AIGateway()
        self.router = CapabilityRouter()
        self.metrics = get_metrics_collector()
        self._services: Dict[str, CapabilityServiceProtocol] = {
            "chat": ChatCapabilityService(self.gateway),
            "business": BusinessCapabilityService(self.gateway),
            "research": ResearchCapabilityService(self.gateway),
            "documents": DocumentsCapabilityService(self.gateway),
            "coding": CodingCapabilityService(self.gateway),
            "automation": AutomationCapabilityService(self.gateway),
        }

    def get_service(self, capability_name: str) -> CapabilityServiceProtocol | None:
        return self._services.get(str(capability_name or "").strip().lower())

    def _resolve_service(self, request: CapabilityRequest) -> CapabilityServiceProtocol:
        route = self.router.route(
            user_query=request.user_query,
            chat_mode=request.chat_mode,
            capability_hint=request.capability_hint,
        )
        service = self._services.get(route.capability) or self._services["chat"]
        self.metrics.record_capability_route(route.capability)
        return service

    async def run(self, request: CapabilityRequest) -> CapabilityResponse:
        service = self._resolve_service(request)
        started = time.time()
        try:
            response = await service.run(request)
            latency_ms = (time.time() - started) * 1000.0
            self.metrics.record_capability_request(service.name, latency_ms=latency_ms, success=True)
            return response
        except Exception:
            latency_ms = (time.time() - started) * 1000.0
            self.metrics.record_capability_request(service.name, latency_ms=latency_ms, success=False)
            raise

    async def run_stream(self, request: CapabilityRequest) -> AsyncGenerator[str, None]:
        service = self._resolve_service(request)
        started = time.time()
        ok = True
        try:
            async for token in service.run_stream(request):
                yield token
        except Exception:
            ok = False
            raise
        finally:
            latency_ms = (time.time() - started) * 1000.0
            self.metrics.record_capability_request(service.name, latency_ms=latency_ms, success=ok)


_platform_singleton: AIPlatformService | None = None


def get_ai_platform() -> AIPlatformService:
    global _platform_singleton
    if _platform_singleton is None:
        _platform_singleton = AIPlatformService()
    return _platform_singleton
