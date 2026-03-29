"""Split query-mode handlers for LLMProcessor."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from app.config import CODING_MODEL, UI_MODEL, LLM_MODEL
from app.llm.prompts import MODE_SWITCH_PROMPT_TEMPLATE
from app.llm.router import (
    INTERNAL_SYSTEM_PROMPT,
    TONE_MAP,
    execute_serper_batch,
    get_openrouter_client,
    get_system_prompt_for_mode,
    get_internal_system_prompt_for_personality,
    get_tools_for_mode,
    select_tools_for_mode,
)
from app.llm.routing_log import log_routing_decision
from app.llm.guards import normalize_user_query
from app.chat.mode_mapper import normalize_chat_mode
from app.llm.processor_helpers import (
    LOGIC_CODING_INTENTS,
    UI_SUB_INTENTS,
    _apply_reasoning,
    _get_max_tokens_for_sub_intent,
    get_model_for_intent,
    _resolve_temperature,
    _single_file_html_instruction,
    _wants_single_file_html,
)


async def process_agent_query_impl(
    self,
    user_query: str,
    personality: Optional[Dict] = None,
    user_settings: Optional[Dict] = None,
    mode: str = "smart",
    sub_intent: str = "general",
    user_id: Optional[str] = None,
    emotions: List[str] = [],
    emotional_instruction: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Handle internal queries (greetings, math, code, logic)."""
    user_query = normalize_user_query(user_query)
    mode = normalize_chat_mode(mode)
    if personality:
        system_prompt = get_internal_system_prompt_for_personality(personality, user_settings, mode=mode)
    else:
        if mode == "smart" and sub_intent == "general":
            system_prompt = get_system_prompt_for_mode("smart", user_settings, user_id, user_query)
        else:
            from app.llm.router import _build_user_context_string

            system_prompt = INTERNAL_SYSTEM_PROMPT + _build_user_context_string(user_settings)
            if mode == "smart":
                from app.llm.router import NORMAL_MARKDOWN_POLISH

                system_prompt = f"{system_prompt}\n{NORMAL_MARKDOWN_POLISH}"

    strict_structured_mode = False

    from app.llm.router import INTERNAL_MODE_PROMPTS

    if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
        system_prompt = (
            f"{system_prompt}\n\n"
            + MODE_SWITCH_PROMPT_TEMPLATE.format(
                sub_intent=sub_intent.upper(), specialized_prompt=INTERNAL_MODE_PROMPTS[sub_intent]
            )
        )
    if sub_intent == "ui_demo_html" and _wants_single_file_html(user_query):
        system_prompt = f"{system_prompt}{_single_file_html_instruction()}"

    if not strict_structured_mode:
        if emotional_instruction:
            system_prompt = f"{system_prompt}\n\n{emotional_instruction}"
        else:
            for emotion in emotions:
                if emotion in TONE_MAP:
                    system_prompt = f"{system_prompt}\n\n{TONE_MAP[emotion]}"

    model_to_use = self.model
    if sub_intent in UI_SUB_INTENTS:
        model_to_use = UI_MODEL
    elif sub_intent in LOGIC_CODING_INTENTS:
        model_to_use = CODING_MODEL
    if personality and personality.get("id") == "coding_buddy":
        model_to_use = UI_MODEL if sub_intent in UI_SUB_INTENTS else CODING_MODEL
        print(f"[LLM] Switching to {model_to_use} for Coding Buddy")

    try:
        reason = (
            "ui_override"
            if sub_intent in UI_SUB_INTENTS
            else "coding_override"
            if (sub_intent in LOGIC_CODING_INTENTS or (personality and personality.get("id") == "coding_buddy"))
            else "default_persona_model"
        )
        log_routing_decision(
            user_id,
            {
                "intent": "INTERNAL",
                "sub_intent": sub_intent,
                "mode": mode,
                "forced_model": model_to_use,
                "reason": reason,
            },
        )
    except Exception as e:
        print(f"[RoutingLog] Failed: {e}")

    client = get_openrouter_client() if model_to_use == LLM_MODEL else get_openrouter_client()
    create_kwargs = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    }
    max_tokens = _get_max_tokens_for_sub_intent(sub_intent)
    if max_tokens:
        create_kwargs["max_tokens"] = max_tokens
    temperature = _resolve_temperature(personality, sub_intent)
    if sub_intent in UI_SUB_INTENTS and temperature is not None:
        temperature = min(temperature, 0.2)
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    _apply_reasoning(create_kwargs, model_to_use, sub_intent, user_settings)
    create_kwargs = self._guard_kwargs(create_kwargs, user_query)
    response = await client.chat.completions.create(**create_kwargs)
    out_text = self._sanitize_output_text(response.choices[0].message.content)
    out_text = self._sanitize_output_text(out_text)
    return out_text


async def process_deep_search_query_impl(
    self,
    user_query: str,
    mode: str = "smart",
    context_messages: Optional[List[Dict]] = None,
    personality: Optional[Dict] = None,
    user_settings: Optional[Dict] = None,
    sub_intent: str = "general",
    user_id: Optional[str] = None,
    emotions: List[str] = [],
    emotional_instruction: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Handle external queries that need web search."""
    user_query = normalize_user_query(user_query)
    mode = normalize_chat_mode(mode)
    selected_tools = await select_tools_for_mode(user_query, mode)
    tools_dict = get_tools_for_mode(mode)

    aggregated_context = {}
    for tool in selected_tools:
        if tool in tools_dict:
            endpoint = tools_dict[tool]
            param_key = "url" if tool == "Webpage" else "q"
            result = await execute_serper_batch(endpoint, [user_query], param_key=param_key)
            aggregated_context[tool] = result

    context_str = json.dumps(aggregated_context, indent=2)

    if personality and mode == "smart":
        from app.llm.router import get_system_prompt_for_personality

        system_prompt = get_system_prompt_for_personality(
            personality, user_settings, user_id, user_query, session_id=session_id
        )
    else:
        system_prompt = get_system_prompt_for_mode(
            mode, user_settings, user_id, user_query, session_id=session_id
        )

    from app.llm.router import INTERNAL_MODE_PROMPTS

    if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
        system_prompt = (
            f"{system_prompt}\n\n"
            + MODE_SWITCH_PROMPT_TEMPLATE.format(
                sub_intent=sub_intent.upper(), specialized_prompt=INTERNAL_MODE_PROMPTS[sub_intent]
            )
        )
    if emotional_instruction:
        system_prompt = f"{system_prompt}\n\n{emotional_instruction}"
    else:
        for emotion in emotions:
            if emotion in TONE_MAP:
                system_prompt = f"{system_prompt}\n\n{TONE_MAP[emotion]}"

    messages = [{"role": "system", "content": system_prompt}]
    if context_messages:
        messages.extend(context_messages[-6:])
    messages.append({"role": "user", "content": f"Search Data:\n{context_str}\n\nUser Query: {user_query}"})

    client_type, model_to_use, temperature, _, route_reason = get_model_for_intent(
        mode, sub_intent, personality, user_query
    )

    try:
        log_routing_decision(
            user_id,
            {
                "intent": "EXTERNAL",
                "sub_intent": sub_intent,
                "mode": mode,
                "forced_model": model_to_use,
                "reason": route_reason,
            },
        )
    except Exception as e:
        print(f"[RoutingLog] Failed: {e}")

    client = get_openrouter_client() if client_type == "openai" else get_openrouter_client()
    create_kwargs = {"model": model_to_use, "messages": messages}
    create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    _apply_reasoning(create_kwargs, model_to_use, sub_intent, user_settings)
    create_kwargs = self._guard_kwargs(create_kwargs, user_query)
    response = await client.chat.completions.create(**create_kwargs)
    out_text = self._sanitize_output_text(response.choices[0].message.content)
    out_text = self._sanitize_output_text(out_text)

    return out_text, selected_tools


