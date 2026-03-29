"""Main non-stream message entry flow for LLMProcessor."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from app.llm.guards import normalize_user_query
from app.llm.router import analyze_and_route_query
from app.llm.processor_helpers import MAX_VERIFY_TIME, _resolve_runtime_intent
from app.llm.emotion_engine import emotion_engine
from app.llm.strategy_memory import strategy_memory
from app.llm.prompt_optimizer import prompt_optimizer
from app.llm.skill_router_runtime import record_skill_outcome, get_last_skill_trace
from app.llm.feedback_engine import feedback_engine
from app.llm.user_profiler import user_profiler
from app.safety.content_policy import classify_nsfw
from app.chat.mode_mapper import normalize_chat_mode


async def process_message_impl(
    self,
    user_query: str,
    mode: str = "smart",
    context_messages: Optional[List[Dict]] = None,
    personality: Optional[Dict] = None,
    user_settings: Optional[Dict] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Main entry point - routes to appropriate handler and returns response dict."""
    user_query = normalize_user_query(user_query)
    mode = normalize_chat_mode(mode)
    analysis = await analyze_and_route_query(user_query, mode, personality=personality)
    routed_intent = analysis.get("intent", "DEEP_SEARCH")
    sub_intent = analysis.get("sub_intent", "general")
    emotions = analysis.get("emotions", [])
    intent = _resolve_runtime_intent(mode, routed_intent, user_query)

    import time as _time

    _start = _time.time()

    emotional_instruction = None
    strategy_instruction = None
    prompt_variant_name = "default"
    prompt_variant_instruction = ""
    skill_level = 0.5

    if session_id:
        state = await emotion_engine.load_state(session_id, user_id=user_id)
        state = emotion_engine.update_state(state, emotions, sub_intent)
        await emotion_engine.save_state(session_id, state, user_id=user_id)
        emotional_instruction = emotion_engine.get_instruction(state)
        skill_level = state.skill_level

    if user_id:
        user_strategy = await strategy_memory.load_strategy(user_id)
        user_strategy = strategy_memory.update_from_query(user_strategy, user_query)
        await strategy_memory.save_strategy(user_id, user_strategy)
        strategy_instruction = strategy_memory.get_instruction(user_strategy)

    prompt_variant_name, prompt_variant_instruction = prompt_optimizer.select_variant(sub_intent, session_id or "")

    tools = []
    if intent == "INTERNAL":
        response_text = await self.process_agent_query(
            user_query,
            personality,
            user_settings,
            mode=mode,
            sub_intent=sub_intent,
            user_id=user_id,
            emotions=emotions,
            emotional_instruction=emotional_instruction,
            session_id=session_id,
        )
        result_response = self._format_user_answer(response_text, mode)
    elif intent == "AGENT":
        _agent_tokens: List[str] = []
        async for token in self._process_agent_mode(
            user_query,
            context_messages,
            personality,
            user_settings,
            user_id,
            session_id,
            start_time=_time.time(),
            mode=mode,
            intent=routed_intent,
            sub_intent=sub_intent,
        ):
            if token.startswith("[INFO]"):
                continue
            _agent_tokens.append(token)
        response_text = "".join(_agent_tokens)
        result_response = self._format_user_answer(response_text, mode)
    else:
        response_text, tools = await self.process_deep_search_query(
            user_query,
            mode,
            context_messages,
            personality,
            user_settings,
            sub_intent,
            user_id,
            emotions=emotions,
            emotional_instruction=emotional_instruction,
            session_id=session_id,
        )
        result_response = self._format_user_answer(response_text, mode)

    if not self.is_structured(result_response):
        result_response = await self._retry_with_format_hint(user_query, result_response, mode)

    if self._should_run_verifier(mode, user_query, result_response):
        try:
            verified = await asyncio.wait_for(
                self._verify_response_quality(user_query, result_response, mode),
                timeout=MAX_VERIFY_TIME,
            )
            if verified:
                result_response = verified
        except asyncio.TimeoutError:
            print(f"[Verifier] Timeout after {MAX_VERIFY_TIME:.1f}s, using draft response")
        except Exception as e:
            print(f"[Verifier] Non-blocking failure: {e}")

    blocked_out, _block_reason = classify_nsfw(result_response)
    if blocked_out:
        result_response = "I can help with educational or safety-oriented discussion, but I can't provide explicit sexual content."

    _latency = (_time.time() - _start) * 1000
    try:
        skill_trace = get_last_skill_trace(session_id)
        record_skill_outcome(
            session_id=session_id,
            success=True,
            latency_ms=_latency,
            fallback_level=int(skill_trace.get("fallback_level", 0)),
        )
    except Exception as e:
        print(f"[SkillRouter] Outcome logging failed: {e}")

    feedback_engine.log_interaction(
        session_id=session_id or "",
        user_id=user_id or "",
        query=user_query,
        intent=intent,
        sub_intent=sub_intent,
        model_used=mode,
        emotions=emotions,
        skill_level=skill_level,
        response_length=len(result_response),
        latency_ms=_latency,
        prompt_variant=prompt_variant_name,
    )

    if user_id:
        try:
            _profile = await user_profiler.load_profile(user_id)
            user_profiler.update_from_interaction(
                _profile,
                sub_intent,
                user_query,
                len(result_response),
                emotions,
                model_used=mode,
                prompt_variant=prompt_variant_name,
            )
        except Exception as e:
            print(f"[UserProfiler] Update failed: {e}")

    return {
        "success": True,
        "response": result_response,
        "mode_used": "agent" if intent == "AGENT" else "deep_search" if intent in ("DEEP_SEARCH", "EXTERNAL") else "internal",
        "tools_activated": tools if intent in ("DEEP_SEARCH", "EXTERNAL") else [],
    }
