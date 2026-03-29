from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, Optional

from app.agent.agent_state import AgentState
from app.agent.task_manager import TaskManager
from app.capabilities.base import BaseCapabilityService
from app.chat.mode_mapper import normalize_chat_mode
from app.llm.router import get_openrouter_client
from app.platform.types import CapabilityRequest, CapabilityResponse


class ResearchCapabilityService(BaseCapabilityService):
    name = "research"
    default_mode = "research"

    def __init__(self, gateway):
        super().__init__(gateway)
        self._task_manager: Optional[TaskManager] = None

    def _should_use_task_manager(self, request: CapabilityRequest) -> bool:
        mode = normalize_chat_mode(str(request.chat_mode or ""))
        return mode in {"smart", "agent", "research_pro"}

    def _get_task_manager(self) -> TaskManager:
        if self._task_manager is None:
            self._task_manager = TaskManager(get_openrouter_client())
        return self._task_manager

    def get_debug_snapshot(self, *, range_window: str = "24h", limit: int = 25) -> Dict[str, Any]:
        return self._get_task_manager().get_debug_snapshot(range_window=range_window, limit=limit)

    def update_debug_config(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        return self._get_task_manager().update_debug_config(patch)

    def get_mode_check(self, *, input_text: str, requested_mode: str = "smart") -> Dict[str, Any]:
        return self._get_task_manager().get_mode_check(input_text=input_text, requested_mode=requested_mode)

    @staticmethod
    def _build_initial_state(request: CapabilityRequest) -> AgentState:
        mode = normalize_chat_mode(str(request.chat_mode or ""))
        state = AgentState(
            query=request.user_query,
            goal=request.user_query,
            user_id=request.user_id,
            session_id=request.session_id,
            mode=mode,
            context_messages=list(request.context_messages or []),
        )
        if isinstance(request.metadata, dict) and request.metadata:
            state.workspace.intermediate_results["request_metadata"] = dict(request.metadata)
        if request.file_ids:
            state.workspace.intermediate_results["file_ids"] = list(request.file_ids)
        return state

    @staticmethod
    def _parse_info_payload(chunk: str) -> Optional[Dict[str, Any]]:
        if not chunk.startswith("[INFO]"):
            return None
        raw = chunk[len("[INFO]"):].strip()
        if not raw.startswith("{"):
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    @staticmethod
    def _extract_final_answer(buffer: str) -> str:
        marker = "[FINAL_ANSWER]"
        if marker not in buffer:
            return buffer.strip()
        _, _, tail = buffer.rpartition(marker)
        return tail.strip()

    async def run(self, request: CapabilityRequest) -> CapabilityResponse:
        if not self._should_use_task_manager(request):
            return await super().run(request)

        manager = self._get_task_manager()
        state = self._build_initial_state(request)
        stream_buffer: list[str] = []
        confidence = 0.0

        async for chunk in manager.run_goal(request.user_query, initial_state=state):
            info = self._parse_info_payload(chunk.strip())
            if info and info.get("event") == "final_answer":
                raw_confidence = info.get("confidence")
                if isinstance(raw_confidence, (int, float)):
                    confidence = float(raw_confidence)
                continue
            stream_buffer.append(chunk)

        final_answer = self._extract_final_answer("".join(stream_buffer))
        return CapabilityResponse(
            capability=self.name,
            payload={
                "response": final_answer,
                "confidence": confidence,
                "sources": list(state.workspace.sources),
                "metadata": {
                    "agent_backend": "task_manager",
                    "mode": normalize_chat_mode(str(state.mode or request.chat_mode or "smart")),
                },
            },
        )

    async def run_stream(self, request: CapabilityRequest) -> AsyncGenerator[str, None]:
        if not self._should_use_task_manager(request):
            async for token in super().run_stream(request):
                yield token
            return

        manager = self._get_task_manager()
        in_final_answer = False
        async for chunk in manager.run_goal(request.user_query, initial_state=self._build_initial_state(request)):
            stripped = chunk.strip()
            if not stripped:
                continue
            if stripped.startswith("[INFO]"):
                yield chunk
                continue

            # Do not stream raw internal reasoning/log text to users.
            # Only stream text after the [FINAL_ANSWER] marker.
            if not in_final_answer:
                if "[FINAL_ANSWER]" not in chunk:
                    continue
                in_final_answer = True
                _, _, tail = chunk.partition("[FINAL_ANSWER]")
                tail = tail.lstrip("\r\n")
                if tail:
                    yield tail
                continue

            # Once final answer starts, stream all subsequent text chunks.
            yield chunk
