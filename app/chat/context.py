"""
Relyce AI - Chat Context Manager
Manages conversation context (in-memory for now, Redis-ready)
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict

# In-memory context storage (Redis-ready structure)
# Format: context_store[user_id][chat_id] = [messages]
_context_store: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
_context_timestamps: Dict[str, Dict[str, datetime]] = defaultdict(dict)
_context_summaries: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(str))
_context_personalities: Dict[str, Dict[str, str]] = defaultdict(dict)

# Context settings
MAX_CONTEXT_MESSAGES = 20  # Absolute safety cap (RAM protection)
SUMMARY_TRIGGER_MESSAGES = 12 # Trigger summarization when this many messages exist
KEEP_LAST_MESSAGES = 6 # Keep this many raw messages after summarization
CONTEXT_TTL_MINUTES = 60  # 60 minutes TTL

def get_context_key(user_id: str, chat_id: str) -> str:
    """Generate context key (Redis-compatible format)"""
    return f"context:{user_id}:{chat_id}"

def get_raw_message_count(user_id: str, chat_id: str) -> int:
    """Get the raw count of messages in the store (excluding dynamically injected ones)"""
    return len(_context_store[user_id][chat_id])

def _ensure_personality_context(user_id: str, chat_id: str, personality_id: Optional[str]) -> None:
    """
    If personality changes for a session, clear context to avoid prompt bleed.
    """
    if not personality_id or user_id == "anonymous":
        return
    last_personality = _context_personalities.get(user_id, {}).get(chat_id)
    if last_personality and last_personality != personality_id:
        clear_context(user_id, chat_id)
    _context_personalities[user_id][chat_id] = personality_id


def get_context(user_id: str, chat_id: str, personality_id: Optional[str] = None) -> List[Dict]:
    """
    Get conversation context for a user's chat session.
    Returns ALL current memory messages (processor handles slicing).
    Hydrates from Firestore if memory is empty (Server Restart/TTL).
    """
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

def get_context_for_llm(user_id: str, chat_id: str, personality_id: Optional[str] = None) -> List[Dict]:
    """
    Get context formatted for LLM consumption.
    Injects summary as system message if it exists.
    Returns list of {"role": "user"|"assistant"|"system", "content": str}
    """
    context_msgs = []
    _ensure_personality_context(user_id, chat_id, personality_id)
    
    # 1. Inject Summary (Long-term memory)
    summary = get_session_summary(user_id, chat_id)
    if summary:
        context_msgs.append({
            "role": "system",
            "content": f"Conversation summary so far:\n{summary}"
        })
    
    # 2. Add Recent Messages (Short-term memory)
    raw_messages = get_context(user_id, chat_id, personality_id)
    for msg in raw_messages:
        role = msg["role"]
        if role == "bot":
            role = "assistant"
        context_msgs.append({"role": role, "content": msg["content"]})
    
    return context_msgs

def update_context_with_exchange(
    user_id: str, 
    chat_id: str, 
    user_message: str, 
    assistant_response: str,
    personality_id: Optional[str] = None
) -> None:
    """
    Add a complete exchange (user message + assistant response) to context.
    """
    _ensure_personality_context(user_id, chat_id, personality_id)
    add_to_context(user_id, chat_id, "user", user_message)
    add_to_context(user_id, chat_id, "assistant", assistant_response)
