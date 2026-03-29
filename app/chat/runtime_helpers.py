"""Shared runtime helpers for HTTP/WS chat flows."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from app.auth import get_firestore_db
from app.chat.context import get_context_for_llm, update_context_with_exchange
from app.chat.history import increment_message_count, save_message_to_firebase
from app.chat.personalities import get_personality_by_id
from app.chat.user_profile import get_session_personality_id
from app.chat.mode_mapper import normalize_chat_mode

def _personality_has_prompt(personality: Optional[Any]) -> bool:
    if not personality:
        return False
    if isinstance(personality, dict):
        return bool(str(personality.get("prompt") or "").strip())
    return bool(str(getattr(personality, "prompt", "") or "").strip())


def resolve_memory_user_id(user_id: str) -> str:
    """Resolve uniqueUserId used by memory layers; fallback to UID."""
    memory_user_id = user_id
    try:
        db = get_firestore_db()
        if db:
            user_doc = db.collection("users").document(user_id).get()
            if user_doc.exists:
                memory_user_id = user_doc.to_dict().get("uniqueUserId", user_id)
    except Exception as e:
        print(f"[RuntimeHelpers] uniqueUserId resolution error (non-blocking): {e}")
    return memory_user_id


def resolve_personality_context(
    *,
    user_id: str,
    session_id: Optional[str],
    chat_mode: str,
    personality: Optional[Any],
    personality_id: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve active personality payload + id consistently across HTTP/WS."""
    normalized_mode = normalize_chat_mode(str(chat_mode or "smart"))
    if normalized_mode != "smart":
        return None, None

    resolved_personality = personality
    resolved_id = personality_id

    if resolved_personality and not resolved_id and hasattr(resolved_personality, "id"):
        resolved_id = resolved_personality.id

    # Backward compatibility: some clients send a lightweight personality object
    # without `prompt`. Rehydrate full persona by id so custom prompts are preserved.
    if resolved_personality and resolved_id and not _personality_has_prompt(resolved_personality):
        p_data = get_personality_by_id(user_id, resolved_id)
        if p_data:
            resolved_personality = p_data

    if not resolved_personality and not resolved_id and session_id:
        saved_id = get_session_personality_id(user_id, session_id)
        if saved_id:
            resolved_id = saved_id

    if not resolved_personality and resolved_id:
        p_data = get_personality_by_id(user_id, resolved_id)
        if p_data:
            resolved_personality = p_data

    if not resolved_personality and normalized_mode == "smart":
        p_data = get_personality_by_id(user_id, "default_relyce")
        if p_data:
            resolved_personality = p_data
            resolved_id = resolved_id or p_data.get("id")

    if hasattr(resolved_personality, "model_dump"):
        resolved_personality = resolved_personality.model_dump()
    elif hasattr(resolved_personality, "dict"):
        resolved_personality = resolved_personality.dict()

    if resolved_personality and not resolved_id:
        resolved_id = resolved_personality.get("id")

    return resolved_personality, resolved_id


def persist_exchange_and_schedule_summary(
    *,
    user_id: str,
    session_id: Optional[str],
    user_text: str,
    assistant_text: str,
    personality_id: Optional[str],
    chat_mode: str = "smart",
    include_message_count: bool = True,
) -> Optional[str]:
    """Persist context + firebase history and schedule background summarization."""
    if not session_id:
        return None

    update_context_with_exchange(user_id, session_id, user_text, assistant_text, personality_id, chat_mode)

    save_message_to_firebase(user_id, session_id, "user", user_text, personality_id)
    msg_id = save_message_to_firebase(user_id, session_id, "assistant", assistant_text, personality_id)

    if include_message_count:
        increment_message_count(user_id)

    try:
        from app.llm.router import get_openrouter_client
        from app.memory.summary_manager import summarize_if_needed

        fresh_context_msgs = get_context_for_llm(user_id, session_id, personality_id, chat_mode)
        asyncio.create_task(
            summarize_if_needed(user_id, session_id, fresh_context_msgs, get_openrouter_client())
        )
    except Exception as sum_e:
        print(f"[RuntimeHelpers] Background summary error: {sum_e}")

    return msg_id
