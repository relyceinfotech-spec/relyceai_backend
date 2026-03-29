from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

from app.llm.processor import llm_processor


class AIGateway:
    """Single entry point for model access used by all capabilities."""

    async def generate(
        self,
        *,
        user_query: str,
        mode: str,
        context_messages: Optional[List[Dict[str, Any]]] = None,
        personality: Optional[Dict[str, Any]] = None,
        user_settings: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        file_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return await llm_processor.process_message(
            user_query=user_query,
            mode=mode,
            context_messages=context_messages,
            personality=personality,
            user_settings=user_settings,
            user_id=user_id,
            session_id=session_id,
            file_ids=file_ids,
        )

    async def stream(
        self,
        *,
        user_query: str,
        mode: str,
        context_messages: Optional[List[Dict[str, Any]]] = None,
        personality: Optional[Dict[str, Any]] = None,
        user_settings: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        resume_graph: Optional[Any] = None,
        file_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        async for token in llm_processor.process_message_stream(
            user_query=user_query,
            mode=mode,
            context_messages=context_messages,
            personality=personality,
            user_settings=user_settings,
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            resume_graph=resume_graph,
            file_ids=file_ids,
        ):
            yield token
