"""
Relyce AI - Smart Memory Engine
GPT-style persistent memory: auto-detects important user facts,
stores with importance scoring, retrieves selectively, decays unused entries.

Storage: Firestore → users/{uid}/memory/entries (collection of structured docs)
"""
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import re
import html
import hashlib
import time

from app.auth import get_firestore_db


# ============================================
# CONSTANTS
# ============================================
MAX_MEMORIES_PER_USER = 100
DECAY_30_DAYS_FACTOR = 0.9
DECAY_90_DAYS_FACTOR = 0.7
AUTO_DELETE_THRESHOLD = 0.1
DEFAULT_IMPORTANCE = 0.6
HIGH_IMPORTANCE = 0.85
SIMILARITY_CATEGORIES = {
    "identity": ["identity"],
    "profession": ["profession"],
    "preference": ["preference"],
    "project": ["project"],
    "context": ["context"],
}

# In-memory cache (per user)
_memory_cache: Dict[str, List[dict]] = {}
_cache_timestamps: Dict[str, float] = {}
CACHE_TTL_SECONDS = 120  # 2 min cache


# ============================================
# DATA STRUCTURES
# ============================================
@dataclass
class MemoryEntry:
    content: str
    category: str  # identity, profession, preference, project, context
    importance: float = DEFAULT_IMPORTANCE
    created_at: str = ""
    last_used: str = ""
    use_count: int = 0
    source: str = "auto"  # auto | imported
    doc_id: str = ""  # Firestore document ID

    def __post_init__(self):
        now = datetime.utcnow().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_used:
            self.last_used = now

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "source": self.source,
        }

    @staticmethod
    def from_dict(data: dict, doc_id: str = "") -> "MemoryEntry":
        return MemoryEntry(
            content=data.get("content", ""),
            category=data.get("category", "context"),
            importance=data.get("importance", DEFAULT_IMPORTANCE),
            created_at=data.get("created_at", ""),
            last_used=data.get("last_used", ""),
            use_count=data.get("use_count", 0),
            source=data.get("source", "auto"),
            doc_id=doc_id,
        )


# ============================================
# MEMORY WORTHINESS CLASSIFIER
# ============================================

# Category → (pattern, importance_boost)
MEMORY_PATTERNS = [
    # --- Identity ---
    (r"(?:my name is|i[' ]?m|call me|name'?s)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)?)", "identity", HIGH_IMPORTANCE),
    (r"(?:en peru|ena peru|en peyar)\s+(\w+)", "identity", HIGH_IMPORTANCE),
    (r"(?:mera naam|mera name)\s+(\w+)", "identity", HIGH_IMPORTANCE),
    (r"(?:i live in|i[' ]?m from|based in|located in)\s+([A-Za-z][A-Za-z]+(?:[\s,]+[A-Za-z][A-Za-z]+)?)", "identity", 0.7),
    (r"(?:i[' ]?m)\s+(\d{1,2})\s*(?:years?\s*old)", "identity", 0.6),
    (r"(?:i speak|my language is|i know)\s+(tamil|hindi|english|telugu|kannada|malayalam|french|spanish|german|japanese|chinese|korean)", "identity", 0.7),

    # --- Profession ---
    (r"(?:i work as|i[' ]?m a|my job is|my profession is|i work at)\s+(?:a\s+)?([a-z]+(?:\s+[a-z]+){0,3})", "profession", HIGH_IMPORTANCE),
    (r"(?:i[' ]?m an?)\s+(engineer|developer|designer|student|teacher|doctor|lawyer|manager|founder|ceo|freelancer|consultant|architect|analyst|researcher|scientist|artist|writer|musician|photographer|marketer|data\s+scientist|product\s+manager|devops|qa|tester)", "profession", HIGH_IMPORTANCE),
    (r"(?:i use|my stack is|i work with|my tech stack|i code in|i develop with)\s+([a-z]+(?:[\s,/+]+[a-z]+){0,5})", "profession", 0.7),
    (r"(?:i[' ]?m learning|i[' ]?m studying|i study)\s+([a-z]+(?:\s+[a-z]+){0,3})", "profession", 0.6),

    # --- Preference ---
    (r"(?:i (?:really )?(?:like|love|prefer|enjoy))\s+([a-z]+(?:\s+[a-z]+){0,3})", "preference", 0.65),
    (r"(?:i (?:don[' ]?t|hate|dislike|can't stand))\s+([a-z]+(?:\s+[a-z]+){0,3})", "preference", 0.65),
    (r"(?:i prefer|i want|i need)\s+(?:you to )?(be\s+(?:concise|brief|detailed|friendly|professional|casual|formal))", "preference", HIGH_IMPORTANCE),
    (r"(?:keep .+? (?:short|concise|brief|detailed|long))", "preference", 0.75),
    (r"(?:don't|do not|never)\s+(?:use|include|add)\s+(emojis?|markdown|code blocks?|bullet points?)", "preference", 0.75),
    (r"(?:always|please)\s+(?:use|include|add)\s+(emojis?|markdown|code blocks?|bullet points?|examples?)", "preference", 0.75),
    (r"(?:respond|reply|answer|talk)\s+(?:in|using)\s+(tanglish|tamil|hindi|english|formal|casual|slang)", "preference", HIGH_IMPORTANCE),

    # --- Project ---
    (r"(?:i[' ]?m building|i am building|i[' ]?m working on|i am working on|my project is|i[' ]?m developing|i am developing|i[' ]?m creating|i am creating)\s+(.+?)(?:\.|$)", "project", HIGH_IMPORTANCE),
    (r"(?:my (?:app|application|website|product|startup|company) is)\s+(.+?)(?:\.|$)", "project", HIGH_IMPORTANCE),
    (r"(?:i[' ]?m using|i am using|my backend is|my frontend is|my database is|i deploy on|i host on)\s+([a-z]+(?:[\s,/+]+[a-z]+){0,5})", "project", 0.7),
    (r"(?:my goal is|i want to|i plan to|i aim to)\s+(.+?)(?:\.|$)", "project", 0.65),

    # --- Context ---
    (r"(?:i have a|i own a)\s+(dog|cat|pet|car|business|company|shop|store|blog|channel|podcast)", "context", 0.55),
    (r"(?:i[' ]?m (?:married|single|engaged|divorced|a parent|a father|a mother))", "context", 0.5),
    (r"(?:my (?:hobby|hobbies|interest|interests) (?:is|are|include))\s+(.+?)(?:\.|$)", "context", 0.55),
]

# Short messages that should NEVER be stored
NOISE_PATTERNS = [
    r"^(hi|hello|hey|ok|okay|sure|yes|no|yep|nope|thanks|thank you|lol|lmao|haha|hmm|hm|nice|cool|great|good|fine|alright|k|ya|yea|yeah|nah|bro|da|macha|dei|seri|ama|illa|test|ping|pong|bye|💀|😂|🔥|👍|❤️|🙏|😏)[\s!?.]*$",
    r"^.{0,3}$",  # 3 chars or less
]

NOISE_COMPILED = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


def _sanitize(value: str, max_length: int = 200) -> str:
    """Sanitize extracted memory content."""
    if not value:
        return ""
    value = str(value).strip()
    value = re.sub(r'[<>"\'`{}\[\]\\]', '', value)
    value = html.escape(value)
    return value[:max_length]


def _content_hash(content: str) -> str:
    """Generate a short hash for duplicate detection."""
    return hashlib.md5(content.lower().strip().encode()).hexdigest()[:12]


def classify_memory(message: str) -> List[MemoryEntry]:
    """
    Classify a user message for memory-worthy content.
    Returns list of MemoryEntry objects to store. Empty if nothing worth remembering.
    Fast: runs in <1ms using regex, no LLM call.
    """
    message = message.strip()

    # Noise filter — skip short/trivial messages
    for noise in NOISE_COMPILED:
        if noise.match(message):
            return []

    # Skip very short messages (less than 8 chars)
    if len(message) < 8:
        return []

    entries = []
    message_lower = message.lower()

    for pattern, category, importance in MEMORY_PATTERNS:
        match = re.search(pattern, message_lower, re.IGNORECASE)
        if match:
            # Extract the captured group or full match
            content = match.group(1).strip() if match.lastindex else match.group(0).strip()
            content = _sanitize(content)

            if len(content) < 2:
                continue

            # Build human-readable memory label
            label = _build_memory_label(message_lower, match, category, content)

            entries.append(MemoryEntry(
                content=label,
                category=category,
                importance=importance,
            ))

    return entries


def _build_memory_label(message: str, match: re.Match, category: str, raw_content: str) -> str:
    """Build a clean, human-readable memory string."""
    # For identity: "Name: Tamizh", "Location: Chennai"
    if category == "identity":
        if "name" in match.re.pattern:
            return f"Name: {raw_content.title()}"
        if "live in" in match.re.pattern or "from" in match.re.pattern or "based in" in match.re.pattern:
            return f"Location: {raw_content.title()}"
        if "years" in match.re.pattern or "old" in match.re.pattern:
            return f"Age: {raw_content}"
        if "speak" in match.re.pattern or "language" in match.re.pattern:
            return f"Language: {raw_content.title()}"
        return raw_content.title()

    if category == "profession":
        if "work as" in match.re.pattern or "i'm a" in match.re.pattern:
            return f"Works as: {raw_content.title()}"
        if "use" in match.re.pattern or "stack" in match.re.pattern or "code in" in match.re.pattern:
            return f"Tech stack: {raw_content}"
        if "learning" in match.re.pattern or "studying" in match.re.pattern:
            return f"Learning: {raw_content.title()}"
        return raw_content.title()

    if category == "preference":
        if "like" in match.re.pattern or "love" in match.re.pattern or "enjoy" in match.re.pattern:
            return f"Likes: {raw_content}"
        if "don't" in match.re.pattern or "hate" in match.re.pattern or "dislike" in match.re.pattern:
            return f"Dislikes: {raw_content}"
        return f"Prefers: {raw_content}"

    if category == "project":
        if "building" in match.re.pattern or "working on" in match.re.pattern:
            return f"Building: {raw_content}"
        if "goal" in match.re.pattern or "want to" in match.re.pattern or "plan to" in match.re.pattern:
            return f"Goal: {raw_content}"
        if "using" in match.re.pattern or "backend" in match.re.pattern or "frontend" in match.re.pattern:
            return f"Uses: {raw_content}"
        return raw_content

    return raw_content


# ============================================
# STORAGE LAYER (Firestore)
# ============================================

def _get_entries_ref(user_id: str):
    """Get Firestore collection reference for user's memory entries."""
    db = get_firestore_db()
    if not db:
        return None
    return db.collection("users").document(user_id).collection("memory").document("smart").collection("entries")


def _invalidate_cache(user_id: str):
    """Clear cached memories for a user."""
    _memory_cache.pop(user_id, None)
    _cache_timestamps.pop(user_id, None)


def load_all_memories(user_id: str, force_refresh: bool = False) -> List[MemoryEntry]:
    """Load all memories for a user from Firestore (cached)."""
    if user_id == "anonymous":
        return []

    # Check cache
    if not force_refresh and user_id in _memory_cache:
        cache_age = time.time() - _cache_timestamps.get(user_id, 0)
        if cache_age < CACHE_TTL_SECONDS:
            return [MemoryEntry.from_dict(d, d.get("_doc_id", "")) for d in _memory_cache[user_id]]

    try:
        ref = _get_entries_ref(user_id)
        if not ref:
            return []

        docs = ref.order_by("importance", direction="DESCENDING").limit(MAX_MEMORIES_PER_USER).stream()
        entries = []
        raw_cache = []
        for doc in docs:
            data = doc.to_dict()
            data["_doc_id"] = doc.id
            raw_cache.append(data)
            entries.append(MemoryEntry.from_dict(data, doc.id))

        _memory_cache[user_id] = raw_cache
        _cache_timestamps[user_id] = time.time()
        return entries

    except Exception as e:
        print(f"[SmartMemory] Error loading memories for {user_id}: {e}")
        return []


def store_memory(user_id: str, entry: MemoryEntry) -> bool:
    """
    Store a memory entry. Handles:
    - Duplicate detection (same content hash → bump importance)
    - Conflict resolution (same category + similar key → replace)
    - Capacity enforcement (evict lowest importance if >MAX)
    """
    if user_id == "anonymous":
        return False

    try:
        ref = _get_entries_ref(user_id)
        if not ref:
            return False

        content_hash = _content_hash(entry.content)
        existing = load_all_memories(user_id)

        # --- Duplicate detection ---
        for mem in existing:
            if _content_hash(mem.content) == content_hash:
                # Same content: bump importance and update timestamp
                if mem.doc_id:
                    ref.document(mem.doc_id).update({
                        "importance": min(1.0, mem.importance + 0.05),
                        "last_used": datetime.utcnow().isoformat(),
                        "use_count": mem.use_count + 1,
                    })
                _invalidate_cache(user_id)
                print(f"[SmartMemory] Duplicate found, bumped: {entry.content[:50]}")
                return True

        # --- Conflict resolution (same category, same key prefix) ---
        entry_key = entry.content.split(":")[0].strip().lower() if ":" in entry.content else ""
        if entry_key:
            for mem in existing:
                mem_key = mem.content.split(":")[0].strip().lower() if ":" in mem.content else ""
                if mem.category == entry.category and mem_key == entry_key and mem.doc_id:
                    # Same key in same category → replace (e.g., "Name: X" → "Name: Y")
                    ref.document(mem.doc_id).update({
                        "content": entry.content,
                        "importance": max(entry.importance, mem.importance),
                        "last_used": datetime.utcnow().isoformat(),
                        "use_count": mem.use_count + 1,
                    })
                    _invalidate_cache(user_id)
                    print(f"[SmartMemory] Conflict resolved, replaced: {mem.content[:30]} → {entry.content[:30]}")
                    return True

        # --- Capacity enforcement ---
        if len(existing) >= MAX_MEMORIES_PER_USER:
            # Find lowest importance entry and evict
            lowest = min(existing, key=lambda m: m.importance)
            if lowest.doc_id and lowest.importance < entry.importance:
                ref.document(lowest.doc_id).delete()
                print(f"[SmartMemory] Evicted: {lowest.content[:40]} (importance={lowest.importance:.2f})")
            else:
                print(f"[SmartMemory] At capacity, new entry not important enough to store")
                return False

        # --- Store new entry ---
        ref.add(entry.to_dict())
        _invalidate_cache(user_id)
        print(f"[SmartMemory] Stored: [{entry.category}] {entry.content[:50]} (importance={entry.importance:.2f})")
        return True

    except Exception as e:
        print(f"[SmartMemory] Error storing memory: {e}")
        return False


def delete_memory(user_id: str, doc_id: str) -> bool:
    """Delete a specific memory by document ID."""
    try:
        ref = _get_entries_ref(user_id)
        if not ref:
            return False
        ref.document(doc_id).delete()
        _invalidate_cache(user_id)
        return True
    except Exception as e:
        print(f"[SmartMemory] Error deleting memory: {e}")
        return False


# ============================================
# SMART RETRIEVAL
# ============================================

# Categories that should always be included (core identity)
ALWAYS_INCLUDE_CATEGORIES = {"identity", "profession"}


def get_relevant_memories(user_id: str, current_query: str = "", limit: int = 15) -> List[MemoryEntry]:
    """
    Retrieve the most relevant memories for the current context.
    
    Scoring: final_score = importance * recency_factor * category_boost
    - recency_factor: 1.0 (< 7 days), 0.9 (< 30 days), 0.75 (< 90 days), 0.5 (> 90 days)
    - category_boost: identity/profession get 1.3x, others get 1.0x
    
    Also applies decay to old unused memories (side effect).
    """
    if user_id == "anonymous":
        return []

    all_memories = load_all_memories(user_id)
    if not all_memories:
        return []

    now = datetime.utcnow()
    scored: List[Tuple[float, MemoryEntry]] = []
    decay_updates = []

    for mem in all_memories:
        # Parse last_used
        try:
            last_used = datetime.fromisoformat(mem.last_used)
        except (ValueError, TypeError):
            last_used = now

        days_since_use = (now - last_used).days

        # --- Recency factor ---
        if days_since_use < 7:
            recency = 1.0
        elif days_since_use < 30:
            recency = 0.9
        elif days_since_use < 90:
            recency = 0.75
        else:
            recency = 0.5

        # --- Memory decay (side effect) ---
        if days_since_use > 90 and mem.importance > AUTO_DELETE_THRESHOLD:
            new_importance = mem.importance * DECAY_90_DAYS_FACTOR
            if new_importance < AUTO_DELETE_THRESHOLD:
                # Schedule deletion
                decay_updates.append(("delete", mem))
            else:
                decay_updates.append(("decay", mem, new_importance))
        elif days_since_use > 30 and mem.importance > AUTO_DELETE_THRESHOLD:
            new_importance = mem.importance * DECAY_30_DAYS_FACTOR
            decay_updates.append(("decay", mem, new_importance))

        # --- Category boost ---
        cat_boost = 1.3 if mem.category in ALWAYS_INCLUDE_CATEGORIES else 1.0

        # --- Query-relevance boost ---
        query_boost = 1.0
        if current_query:
            q = current_query.lower()
            content_lower = mem.content.lower()
            # Simple keyword overlap check
            content_words = set(content_lower.split())
            query_words = set(q.split())
            overlap = content_words & query_words
            if len(overlap) > 0:
                query_boost = 1.0 + min(0.5, len(overlap) * 0.15)

        final_score = mem.importance * recency * cat_boost * query_boost
        scored.append((final_score, mem))

    # Apply decay updates in background (fire and forget)
    if decay_updates:
        _apply_decay_updates(user_id, decay_updates)

    # Sort by score, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for _, mem in scored[:limit]]


def _apply_decay_updates(user_id: str, updates: list):
    """Apply decay/deletion updates to Firestore (non-critical, best-effort)."""
    try:
        ref = _get_entries_ref(user_id)
        if not ref:
            return

        for update in updates:
            if update[0] == "delete" and update[1].doc_id:
                ref.document(update[1].doc_id).delete()
                print(f"[SmartMemory] Decayed → deleted: {update[1].content[:40]}")
            elif update[0] == "decay" and update[1].doc_id:
                ref.document(update[1].doc_id).update({"importance": update[2]})

        _invalidate_cache(user_id)
    except Exception as e:
        print(f"[SmartMemory] Decay update error (non-critical): {e}")


# ============================================
# PROMPT INJECTION
# ============================================

def format_memories_for_prompt(user_id: str, current_query: str = "") -> str:
    """
    Format relevant memories as a string for system prompt injection.
    Only fetches memories if the user explicitly asks about their stored facts or preferences.
    """
    # Strict check: only inject memory if the user explicitly asks for it
    if not current_query:
        return ""
        
    query_lower = current_query.lower()
    memory_triggers = [
        "what do you know about me", "tell me about myself", "my name", "who am i",
        "do you remember", "what is my", "what are my", "my preference", "profile",
        "about me", "my location", "my age", "my language", "-m", "/memory"
    ]
    
    needs_memory = any(trigger in query_lower for trigger in memory_triggers)
    if not needs_memory:
        return ""

    memories = get_relevant_memories(user_id, current_query)
    
    print(f"[SmartMemory] format_memories_for_prompt called for user={user_id}. Found {len(memories)} relevant memories.")

    if not memories:
        return ""

    lines = []
    for mem in memories:
        lines.append(f"- {mem.content}")

    return (
        "\n**VERIFIED FACTS ABOUT THIS USER (stored from previous conversations):**\n"
        + "\n".join(lines)
        + "\n\nIMPORTANT: These facts are VERIFIED and stored by the system. "
        "Use them naturally in conversation. When the user asks what you know about them, "
        "share ONLY these facts accurately — do NOT make up or guess additional details. "
        "If they ask about something not listed here, say you don't have that information yet.\n"
    )


# ============================================
# MAIN ENTRY POINT (called from websocket.py)
# ============================================

def process_message(user_id: str, message: str) -> List[MemoryEntry]:
    """
    Process a user message for memory-worthy content.
    Called after each user message (fire-and-forget from websocket).
    
    1. Classify message for memory worthiness
    2. Store any detected memories (with dedup + conflict resolution)
    3. Return stored entries (for logging)
    """
    if user_id == "anonymous" or not message or len(message.strip()) < 4:
        return []

    entries = classify_memory(message)

    if not entries:
        return []

    stored = []
    for entry in entries:
        if store_memory(user_id, entry):
            stored.append(entry)

    if stored:
        print(f"[SmartMemory] Processed message → {len(stored)} memories stored for {user_id}")

    return stored


# ============================================
# API HELPERS (for endpoints)
# ============================================

def get_all_memories_for_api(user_id: str) -> List[dict]:
    """Get all memories formatted for API response."""
    memories = load_all_memories(user_id, force_refresh=True)
    return [
        {
            "id": mem.doc_id,
            "content": mem.content,
            "category": mem.category,
            "importance": round(mem.importance, 2),
            "created_at": mem.created_at,
            "last_used": mem.last_used,
            "use_count": mem.use_count,
            "source": mem.source,
        }
        for mem in memories
    ]
