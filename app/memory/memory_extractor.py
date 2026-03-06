"""
Memory Extractor — LLM-based semantic fact extraction.
Extracts ≤3 durable facts from a user+assistant exchange.
Runs in background after stream completes (zero latency impact).
"""
import json
import re
import math
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.config import MEMORY_EXTRACTION_MODEL


@dataclass
class ExtractedFact:
    text: str
    memory_type: str  # user_profile | project_context | technical_stack | preferences | temporary_context
    importance: float  # 0.0 - 1.0
    expires_at: Optional[str] = None  # ISO timestamp or None (permanent)


# Memory types and their expiration rules (days or None for permanent)
EXPIRATION_RULES = {
    "temporary_context": 7,
    "project_context": None,
    "technical_stack": None,
    "user_profile": None,
    "preferences": None,
}

EXTRACTION_PROMPT = """Extract durable long-term facts from this conversation.

Store ONLY information that is:
- Stable over time (will still be true next week)
- Useful in future conversations
- Under 20 words per fact

DO NOT store:
- Greetings, small talk, or acknowledgements
- Temporary questions or one-time requests
- Debugging steps or transient error details
- Information that changes frequently

Maximum: 3 facts.
Each fact must be a single concise sentence under 20 words.

Classify each fact as one of:
- user_profile (name, location, language, role)
- project_context (what user is building/working on)
- technical_stack (tools, frameworks, languages used)
- preferences (communication or workflow preferences)
- temporary_context (current debugging/task, expires in 7d)

Rate importance 0.0-1.0 (1.0 = core identity/project info).

User message: {user_message}

Assistant response (summary): {assistant_summary}

Respond ONLY with a JSON array (no markdown, no explanation):
[{{"text": "fact", "type": "memory_type", "importance": 0.8}}]

If nothing worth storing, respond with: []"""


async def extract_facts(
    user_message: str,
    assistant_response: str,
    user_id: str = ""
) -> List[ExtractedFact]:
    """
    Extract semantic facts from a user+assistant exchange using LLM.
    Returns list of ExtractedFact objects.
    """
    # Skip trivial messages
    if not user_message or len(user_message.strip()) < 10:
        return []
    
    # Truncate assistant response for extraction (save tokens)
    assistant_summary = assistant_response[:500] if assistant_response else ""
    
    try:
        from app.llm.router import get_openrouter_client
        client = get_openrouter_client()
        
        prompt = EXTRACTION_PROMPT.format(
            user_message=user_message[:300],
            assistant_summary=assistant_summary
        )
        
        response = await client.chat.completions.create(
            model=MEMORY_EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )
        
        raw = response.choices[0].message.content or "[]"
        facts = _parse_facts(raw)
        
        if facts:
            print(f"[MemoryExtractor] Extracted {len(facts)} facts for {user_id}")
        
        return facts
        
    except Exception as e:
        print(f"[MemoryExtractor] Extraction failed: {e}")
        return []


def _parse_facts(raw_text: str) -> List[ExtractedFact]:
    """Parse the LLM response into ExtractedFact objects with fallback."""
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```\w*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()
    
    try:
        items = json.loads(clean)
        if not isinstance(items, list):
            return []
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []
    
    facts = []
    now = datetime.utcnow()
    
    for item in items[:3]:  # Hard cap at 3
        if not isinstance(item, dict):
            continue
        
        text = item.get("text", "").strip()
        if not text or len(text) < 5:
            continue
        
        # Enforce 20-word limit
        words = text.split()
        if len(words) > 20:
            text = " ".join(words[:20])
        
        memory_type = item.get("type", "temporary_context")
        if memory_type not in EXPIRATION_RULES:
            memory_type = "temporary_context"
        
        importance = min(1.0, max(0.0, float(item.get("importance", 0.5))))
        
        # Calculate expiration
        expire_days = EXPIRATION_RULES.get(memory_type)
        expires_at = None
        if expire_days is not None:
            expires_at = (now + timedelta(days=expire_days)).isoformat()
        
        facts.append(ExtractedFact(
            text=text[:200],
            memory_type=memory_type,
            importance=importance,
            expires_at=expires_at,
        ))
    
    return facts
