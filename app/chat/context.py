"""
Relyce AI - Chat Context Manager
Manages conversation context (in-memory with Firestore hydration)
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import asyncio
from app.llm.context_optimizer import context_optimizer
from app.chat.mode_mapper import normalize_chat_mode

# In-memory context storage (Firestore-hydrated structure)
# Format: context_store[user_id][chat_id] = [messages]
_context_store: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
_context_timestamps: Dict[str, Dict[str, datetime]] = defaultdict(dict)
_context_summaries: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(str))
_context_personalities: Dict[str, Dict[str, str]] = defaultdict(dict)
_context_modes: Dict[str, Dict[str, str]] = defaultdict(dict)

# Context settings
# Context settings
MAX_CONTEXT_MESSAGES = 50
SUMMARY_TRIGGER_MESSAGES = 12 # Trigger summarization when we have enough history
KEEP_LAST_MESSAGES = 8        # Strict limit requested by user
CONTEXT_TTL_MINUTES = 60

async def cleanup_expired_contexts():
    """Periodic cleanup of expired contexts to prevent memory leaks."""
    while True:
        await asyncio.sleep(3600)
        now = datetime.now(timezone.utc)
        expired_count = 0
        for user_id in list(_context_timestamps.keys()):
            for chat_id in list(_context_timestamps[user_id].keys()):
                ts = _context_timestamps[user_id][chat_id]
                if ts and (now - ts.replace(tzinfo=timezone.utc) if ts.tzinfo else now - ts) > timedelta(minutes=CONTEXT_TTL_MINUTES):
                    clear_context(user_id, chat_id)
                    expired_count += 1
        if expired_count > 0:
            print(f"[Context] Cleaned up {expired_count} expired contexts")

def get_context_key(user_id: str, chat_id: str) -> str:
    """Generate context key"""
    return f"context:{user_id}:{chat_id}"

def get_raw_message_count(user_id: str, chat_id: str) -> int:
    """Get the raw count of messages in the store (excluding dynamically injected ones)"""
    return len(_context_store[user_id][chat_id])

def _ensure_session_context(user_id: str, chat_id: str, personality_id: Optional[str], chat_mode: Optional[str] = None) -> None:
    """
    If personality or mode changes for a session, clear context to avoid prompt bleed.
    """
    if user_id == "anonymous":
        return
    normalized_mode = normalize_chat_mode(str(chat_mode or "smart"))
    last_mode = _context_modes.get(user_id, {}).get(chat_id)
    if last_mode and last_mode != normalized_mode:
        clear_context(user_id, chat_id)
    _context_modes[user_id][chat_id] = normalized_mode

    if normalized_mode != "smart":
        _context_personalities[user_id][chat_id] = ""
        return

    if not personality_id:
        return
    last_personality = _context_personalities.get(user_id, {}).get(chat_id)
    if last_personality and last_personality != personality_id:
        clear_context(user_id, chat_id)
        _context_modes[user_id][chat_id] = normalized_mode
    _context_personalities[user_id][chat_id] = personality_id


def get_context(user_id: str, chat_id: str, personality_id: Optional[str] = None) -> List[Dict]:
    """
    Get conversation context for a user's chat session.
    Returns ALL current memory messages (processor handles slicing).
    Hydrates from Firestore if memory is empty (Server Restart/TTL).
    Also tries to hydrate the session summary if not present.
    """
    # 0. Hydrate Summary 
    if not _context_summaries.get(user_id, {}).get(chat_id):
        from app.chat.history import get_chat_session
        session_data = get_chat_session(user_id, chat_id)
        if session_data and session_data.get('summary'):
            _context_summaries[user_id][chat_id] = session_data['summary']
    # Check TTL
    if chat_id in _context_timestamps.get(user_id, {}):
        last_update = _context_timestamps[user_id][chat_id]
        if datetime.now() - last_update > timedelta(minutes=CONTEXT_TTL_MINUTES):
            # Context expired, clear it
            _context_store[user_id][chat_id] = []
            if chat_id in _context_summaries.get(user_id, {}):
                 del _context_summaries[user_id][chat_id]
            # Fall through to reload
    
    # If empty, try to load from history (Persistence)
    if not _context_store[user_id][chat_id]:
        try:
            from app.chat.history import load_chat_history
            # Load message limit based on safety cap
            history = load_chat_history(user_id, chat_id, limit=MAX_CONTEXT_MESSAGES, personality_id=personality_id)
            if history:
                # Convert to context format (history has 'createdAt', context needs 'timestamp' string usually, but we can adapt)
                for msg in history:
                     _context_store[user_id][chat_id].append({
                         "role": msg['role'],
                         "content": msg['content'],
                         "timestamp": msg.get('timestamp') or datetime.now().isoformat()
                     })
                _context_timestamps[user_id][chat_id] = datetime.now()
                # print(f"[Context] Hydrated {len(history)} messages from Firestore for {chat_id}") # Optional logging
        except Exception as e:
            print(f"[Context] Failed to hydrate history: {e}")

    return _context_store[user_id][chat_id]

def add_to_context(user_id: str, chat_id: str, role: str, content: str) -> None:
    """
    Add a message to the conversation context.
    Maintains sliding window of last N messages (safety trim).
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    _context_store[user_id][chat_id].append(message)
    _context_timestamps[user_id][chat_id] = datetime.now()
    
    # Safety Trim: If context grows too large (e.g. summarization failed), trim it
    if len(_context_store[user_id][chat_id]) > MAX_CONTEXT_MESSAGES * 2:
        _context_store[user_id][chat_id] = _context_store[user_id][chat_id][-MAX_CONTEXT_MESSAGES:]

def get_session_summary(user_id: str, chat_id: str) -> str:
    """Get the current conversation summary"""
    return _context_summaries[user_id][chat_id]

def update_session_summary(user_id: str, chat_id: str, summary: str) -> None:
    """Update conversation summary"""
    _context_summaries[user_id][chat_id] = summary


def persist_session_summary(user_id: str, chat_id: str) -> None:
    """
    Persist summary hook.
    Intentionally lightweight for tests; runtime persistence is handled in history helpers.
    """
    _ = (user_id, chat_id)

def prune_context_messages(user_id: str, chat_id: str, keep_last_n: int) -> None:
    """Prune context to keep only last N messages (used after summarization)"""
    if user_id in _context_store and chat_id in _context_store[user_id]:
        _context_store[user_id][chat_id] = _context_store[user_id][chat_id][-keep_last_n:]

def clear_context(user_id: str, chat_id: str) -> None:
    """Clear context for a specific chat session"""
    if user_id in _context_store and chat_id in _context_store[user_id]:
        _context_store[user_id][chat_id] = []
        if chat_id in _context_timestamps.get(user_id, {}):
            del _context_timestamps[user_id][chat_id]
        if chat_id in _context_summaries.get(user_id, {}):
            del _context_summaries[user_id][chat_id]
        if chat_id in _context_personalities.get(user_id, {}):
            del _context_personalities[user_id][chat_id]
        if chat_id in _context_modes.get(user_id, {}):
            del _context_modes[user_id][chat_id]

def get_context_for_llm(user_id: str, chat_id: str, personality_id: Optional[str] = None, chat_mode: Optional[str] = "smart") -> List[Dict]:
    """
    Get context formatted for LLM consumption.
    Injects summary as system message if it exists.
    Returns list of {"role": "user"|"assistant"|"system", "content": str}
    """
    context_msgs = []
    _ensure_session_context(user_id, chat_id, personality_id, chat_mode)
    
    # 1. Inject Summary (Long-term memory)
    summary = get_session_summary(user_id, chat_id)
    
    # 2. Add Recent Messages (Short-term memory)
    raw_messages = get_context(user_id, chat_id, personality_id)
    
    # Format messages correctly
    formatted_messages = []
    for msg in raw_messages:
        role = msg["role"]
        if role == "bot":
            role = "assistant"
        formatted_messages.append({"role": role, "content": msg["content"]})
        
    # 3. Assemble Memory Architecture (System -> Summary -> Last N)
    from app.llm.context_builder import build_context
    context_msgs = build_context(
        system_prompt="", # System Prompt is appended at processor level, skip it here.
        summary=summary,
        messages=formatted_messages
    )
    
    # Remove the empty system prompt injected by build_context if we left it blank
    context_msgs = [m for m in context_msgs if m["content"] != ""]
    
    return context_msgs

def update_context_with_exchange(
    user_id: str, 
    chat_id: str, 
    user_message: str, 
    assistant_response: str,
    personality_id: Optional[str] = None,
    chat_mode: Optional[str] = "smart",
) -> None:
    """
    Add a complete exchange (user message + assistant response) to context.
    """
    _ensure_session_context(user_id, chat_id, personality_id, chat_mode)
    add_to_context(user_id, chat_id, "user", user_message)
    add_to_context(user_id, chat_id, "assistant", assistant_response)

    # Rolling summary trigger: summarize recent turns and persist hook.
    # Trigger threshold is in message-pairs, so multiply by 2 raw messages.
    turns = len(_context_store[user_id][chat_id]) // 2
    if turns >= SUMMARY_TRIGGER_MESSAGES:
        recent = _context_store[user_id][chat_id][-KEEP_LAST_MESSAGES:]
        combined = " ".join(str(m.get("content") or "") for m in recent)
        summary = context_optimizer.summarize_history(combined) if hasattr(context_optimizer, "summarize_history") else ""
        summary = str(summary or "").strip()
        if not summary:
            summary = f"Conversation summary: {turns} turns captured."
        update_session_summary(user_id, chat_id, summary[:1200])
        persist_session_summary(user_id, chat_id)
