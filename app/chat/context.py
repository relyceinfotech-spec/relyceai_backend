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

# Context settings
MAX_CONTEXT_MESSAGES = 7  # Keep last 7 messages
CONTEXT_TTL_MINUTES = 60  # 60 minutes TTL

def get_context_key(user_id: str, chat_id: str) -> str:
    """Generate context key (Redis-compatible format)"""
    return f"context:{user_id}:{chat_id}"

def get_context(user_id: str, chat_id: str) -> List[Dict]:
    """
    Get conversation context for a user's chat session.
    Returns last N messages.
    """
    # Check TTL
    if chat_id in _context_timestamps.get(user_id, {}):
        last_update = _context_timestamps[user_id][chat_id]
        if datetime.now() - last_update > timedelta(minutes=CONTEXT_TTL_MINUTES):
            # Context expired, clear it
            _context_store[user_id][chat_id] = []
            return []
    
    return _context_store[user_id][chat_id][-MAX_CONTEXT_MESSAGES:]

def add_to_context(user_id: str, chat_id: str, role: str, content: str) -> None:
    """
    Add a message to the conversation context.
    Maintains sliding window of last N messages.
    """
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    _context_store[user_id][chat_id].append(message)
    _context_timestamps[user_id][chat_id] = datetime.now()
    
    # Trim to max messages
    if len(_context_store[user_id][chat_id]) > MAX_CONTEXT_MESSAGES * 2:
        _context_store[user_id][chat_id] = _context_store[user_id][chat_id][-MAX_CONTEXT_MESSAGES:]

def clear_context(user_id: str, chat_id: str) -> None:
    """Clear context for a specific chat session"""
    if user_id in _context_store and chat_id in _context_store[user_id]:
        _context_store[user_id][chat_id] = []
        if chat_id in _context_timestamps.get(user_id, {}):
            del _context_timestamps[user_id][chat_id]

def get_context_for_llm(user_id: str, chat_id: str) -> List[Dict]:
    """
    Get context formatted for LLM consumption.
    Returns list of {"role": "user"|"assistant", "content": str}
    """
    context = get_context(user_id, chat_id)
    return [{"role": msg["role"], "content": msg["content"]} for msg in context]

def update_context_with_exchange(
    user_id: str, 
    chat_id: str, 
    user_message: str, 
    assistant_response: str
) -> None:
    """
    Add a complete exchange (user message + assistant response) to context.
    """
    add_to_context(user_id, chat_id, "user", user_message)
    add_to_context(user_id, chat_id, "assistant", assistant_response)
