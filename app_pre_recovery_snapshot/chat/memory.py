"""
Relyce AI - User Memory Module
Stores and retrieves user facts (name, preferences, etc.) across sessions.
Facts are persisted in Firestore: users/{uid}/memory/facts
"""
from typing import Dict, Optional, List
from datetime import datetime
import json

from app.auth import get_firestore_db


# In-memory cache to avoid repeated Firestore reads
_user_facts_cache: Dict[str, Dict] = {}


def get_user_facts(user_id: str) -> Dict:
    """
    Get all stored facts about a user.
    Returns dict like: {"name": "Tamizh", "location": "Chennai", ...}
    """
    if user_id == "anonymous":
        return {}
    
    # Check cache first
    if user_id in _user_facts_cache:
        return _user_facts_cache[user_id]
    
    try:
        db = get_firestore_db()
        if not db:
            return {}
        
        doc = db.collection('users').document(user_id).collection('memory').document('facts').get()
        
        if doc.exists:
            facts = doc.to_dict() or {}
            _user_facts_cache[user_id] = facts
            return facts
        
        return {}
    except Exception as e:
        print(f"[Memory] Error reading facts for {user_id}: {e}")
        return {}


def save_user_facts(user_id: str, facts: Dict) -> bool:
    """
    Save/update user facts to Firestore.
    Merges with existing facts (doesn't overwrite completely).
    """
    if user_id == "anonymous" or not facts:
        return False
    
    try:
        db = get_firestore_db()
        if not db:
            return False
        
        doc_ref = db.collection('users').document(user_id).collection('memory').document('facts')
        
        # Merge with existing facts
        existing = get_user_facts(user_id)
        merged = {**existing, **facts, "updated_at": datetime.now().isoformat()}
        
        doc_ref.set(merged)
        
        # Update cache
        _user_facts_cache[user_id] = merged
        
        print(f"[Memory] Saved facts for {user_id}: {list(facts.keys())}")
        return True
        
    except Exception as e:
        print(f"[Memory] Error saving facts for {user_id}: {e}")
        return False


def clear_user_facts(user_id: str) -> bool:
    """Clear all stored facts for a user."""
    if user_id == "anonymous":
        return False
    
    try:
        db = get_firestore_db()
        if not db:
            return False
        
        db.collection('users').document(user_id).collection('memory').document('facts').delete()
        
        # Clear cache
        if user_id in _user_facts_cache:
            del _user_facts_cache[user_id]
        
        return True
    except Exception as e:
        print(f"[Memory] Error clearing facts: {e}")
        return False


def format_facts_for_prompt(user_id: str) -> str:
    """
    Format user facts as a string for injection into system prompt.
    Returns empty string if no facts.
    """
    facts = get_user_facts(user_id)
    
    if not facts:
        return ""
    
    # Filter out metadata
    display_facts = {k: v for k, v in facts.items() if k not in ['updated_at', 'extracted_at']}
    
    if not display_facts:
        return ""
    
    facts_list = []
    for key, value in display_facts.items():
        # Format nicely
        key_display = key.replace('_', ' ').title()
        facts_list.append(f"- {key_display}: {value}")
    
    return f"""
**KNOWN FACTS ABOUT THIS USER (from previous conversations):**
{chr(10).join(facts_list)}
Use this information naturally. Don't repeat it back unless asked.
"""


# Fact extraction patterns (fast regex-based, no LLM needed)
import re
import html

def _sanitize_user_input(value: str, max_length: int = 100) -> str:
    """Sanitize user input to prevent XSS and injection attacks."""
    if not value:
        return ""
    value = str(value).strip()
    value = re.sub(r'[<>"\'`{}[\]\\]', '', value)
    value = html.escape(value)
    return value[:max_length]

FACT_PATTERNS = [
    # Name patterns - case-insensitive for user convenience
    (r"(?:my name is|i[' ]?m|call me|name\'?s)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)?)", "name"),
    (r"(?:en peru|ena peru|en peyar)\s+(\w+)", "name"),  # Tamil: my name is
    (r"(?:mera naam|mera name)\s+(\w+)", "name"),  # Hindi: my name is
    (r"(?:naan|na|naa)\s+(\w+)", "name"),  # Tamil casual: I am [name]
    
    # Location patterns - case-insensitive
    (r"(?:i live in|i[' ]?m from|based in|located in)\s+([A-Za-z][A-Za-z]+(?:[\s,]+[A-Za-z][A-Za-z]+)?)", "location"),
    (r"(?:i[' ]?m in)\s+([A-Za-z][A-Za-z]+)", "location"),
    
    # Profession patterns
    (r"(?:i work as|i[' ]?m a|my job is|my profession is)\s+(?:a\s+)?([a-z]+(?:\s+[a-z]+)?)", "profession"),
    (r"(?:i[' ]?m an?)\s+(engineer|developer|designer|student|teacher|doctor|lawyer|manager|founder|ceo)", "profession"),
    
    # Age patterns
    (r"(?:i[' ]?m|i am)\s+(\d{1,2})\s*(?:years?\s*old)?", "age"),
    
    # Preferences
    (r"(?:i (?:really )?(?:like|love|prefer|enjoy))\s+([a-z]+(?:\s+[a-z]+)?)", "likes"),
    (r"(?:i (?:don[' ]?t|hate|dislike))\s+([a-z]+(?:\s+[a-z]+)?)", "dislikes"),
]


def extract_facts_from_message(message: str) -> Dict:
    """
    Extract user facts from a message using pattern matching.
    Fast (no LLM call needed), runs in <1ms.
    Returns dict of extracted facts, empty if none found.
    """
    facts = {}
    message_lower = message.lower()
    
    for pattern, fact_type in FACT_PATTERNS:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if fact_type in ['name', 'location']:
                value = value.title()
                value = _sanitize_user_input(value, max_length=50)
            facts[fact_type] = value
    
    return facts


def process_and_store_facts(user_id: str, message: str) -> Dict:
    """
    Extract facts from message and store them.
    Called after each user message.
    Returns extracted facts (empty dict if none).
    """
    if user_id == "anonymous":
        return {}
    
    extracted = extract_facts_from_message(message)
    
    if extracted:
        save_user_facts(user_id, extracted)
        print(f"[Memory] Extracted facts from message: {extracted}")
    
    return extracted
