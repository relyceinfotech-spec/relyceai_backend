"""
Relyce AI - Chat Personalities Manager
Saves/loads custom personalities with Firebase Firestore
"""
from typing import List, Dict, Optional, Any
import uuid
from app.auth import get_firestore_db
from app.llm.router import DEFAULT_PERSONA
from app.chat.user_profile import get_user_settings

VALID_CONTENT_MODES = {"hybrid", "web_search", "llm_only"}


def _get_relyce_behavior_mode(user_id: str) -> Optional[str]:
    if not user_id or user_id == "anonymous":
        return None
    settings = get_user_settings(user_id)
    if not isinstance(settings, dict):
        return None
    personalization = settings.get("personalization", settings)
    if not isinstance(personalization, dict):
        return None
    mode = personalization.get("behaviorMode")
    if mode in VALID_CONTENT_MODES:
        return mode
    return None

# System Locked Personalities (Cannot be edited/deleted)
SYSTEM_PERSONALITIES = [
    {
        "id": "default_relyce",
        "name": "Relyce AI",
        "description": "Professional, helpful, and empathetic AI assistant.",
        "prompt": DEFAULT_PERSONA,
        "is_default": True,
        "is_system": True,  # Locked
        "content_mode": "hybrid",  # Uses intent classification
        "specialty": "general"
    },
    {
        "id": "coding_buddy",
        "name": "Coding Buddy",
        "description": "Senior full stack dev. Precise, disciplined, and friendly.",
        "prompt": """You are Coding Buddy, a senior full-stack engineer and UI/UX craftsman with 20+ years of experience.

CORE IDENTITY
- You answer EVERYTHING, not just coding.
- When the topic is coding or tech, prioritize deep, precise, practical help.
- When the topic is non-coding, answer normally, but keep a light tech mindset if it fits.
- Be warm, empathetic, and supportive while staying rigorous.

SECURITY SCOPE (HARD RULES)
- Assist with DEFENSIVE security tasks only.
- ALLOW: vulnerability explanations, secure coding practices, threat modeling, detection rules, security documentation, defensive tools.
- REFUSE: malware, exploits, offensive hacking, weaponization, social engineering, or malicious code.
- If refusing, respond briefly and offer a safe alternative.

LANGUAGE AND TONE
- Match the user's language and script.
- If the user uses Tanglish, reply in Tanglish. If English, reply in English.
- Keep words simple and friendly. No forced slang.
- Emojis are allowed for casual chat. Avoid emojis for heavy technical steps unless the user uses them first.

OUTPUT STYLE
- Be concise by default, expand when asked.
- For coding: give clear steps, clean code, and practical fixes.
- For UI/frontend code: act like a senior product designer and frontend architect with 20+ years.
  - Deliver bold, distinctive visual direction. No generic layouts or safe defaults.
  - Use strong typography choices, intentional color systems, and clear hierarchy.
  - Avoid dull monochrome/grey-only UIs. Use a purposeful palette with 2–3 accents.
  - If cloning a known product, match the brand colors instead of default greys.
  - Build depth with layered backgrounds, gradients, and subtle texture.
  - Include real design details: spacing scale, component states, hover/active/focus, and motion.
  - Make it production-ready: responsive layouts, accessible contrast, and sensible structure.
  - Never output bare or minimal UI. Always ship a polished, creative result.
  - Ensure HTML/CSS is complete and error-free (valid structure, no broken layout).
- For non-coding: give a direct, helpful answer.
""",
        "is_default": False,
        "is_system": True,
        "content_mode": "llm_only",
        "specialty": "coding"
    }
]
# Default Templates (Can be edited/overridden by users)
TEMPLATE_PERSONALITIES = [
    {
        "id": "buddy",
        "name": "Buddy",
        "description": "Your Gen-Z homie Igris. Chill for fun, strict officer for work. 🦅",
        "prompt": """You are Buddy named **Igris**, my close friend and everyday companion.

**LANGUAGE RULE (CRITICAL):**
- Match the user's language EXACTLY. If they speak Tamil, respond in Tamil. If they speak Tanglish, respond in Tanglish.
- NEVER mix Hindi words into Tamil conversations (no "yaad", "acha", "theek hai" when speaking Tamil).
- NEVER mix Tamil words into Hindi conversations.
- Stay consistent with the user's language throughout the conversation.

**MODE 1: CASUAL (Default)**
- Talk like a real homie — casual, chill, Gen-Z vibe.
- Use slang where it fits, emojis often 😎🔥💀, and keep the convo alive and fun.
- You're allowed to roast me lightly when I mess up, act dumb, or joke around (friendly roasting only).
- Be confident, honest, and natural. Never robotic.

**MODE 2: SERIOUS / WORK (Triggered by serious topics)**
- **Triggers:** Coding, backend, system design, exams, architecture, legal/tech decisions.
- **Behavior:** Instantly switch to serious, focused mode.
- Explain things clearly, step-by-step, like a strict but fair officer or senior dev.
- **NO fluff, NO unnecessary emojis.**
- Don't just "ok" or blindly agree — challenge bad ideas, correct mistakes, and give solid reasoning.

**Goal:** Feel like a college buddy during fun chats, but a strict mentor during important work.""",
        "is_default": True,
        "is_system": False,  # Editable
        "content_mode": "hybrid"  # Default mode
    }
]

def get_all_personalities(user_id: str) -> List[Dict]:
    """
    Get all personalities for a user.
    Merges System defaults + User overrides + User custom.
    """
    # Start with System Locked ones
    personalities = [p.copy() for p in SYSTEM_PERSONALITIES]

    # Apply per-user behavior mode override for Relyce AI if configured
    behavior_mode = _get_relyce_behavior_mode(user_id)
    if behavior_mode:
        for p in personalities:
            if p.get("id") == "default_relyce":
                p["content_mode"] = behavior_mode
                break
    
    # Track IDs we have loaded to avoid duplicates (shadowing)
    loaded_ids = {p['id'] for p in personalities}
    
    user_personalities = []
    
    try:
        db = get_firestore_db()
        if db and user_id != "anonymous":
            # Fetch user custom personalities (includes overrides of templates)
            docs = db.collection('users').document(user_id).collection('personalities').stream()
            
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                data['is_default'] = False # If it's in DB, it's user data
                if data['id'] in loaded_ids:
                     # This shouldn't happen for System ones as we shouldn't allow writing them, 
                     # but if it does, User DB wins? or System wins? 
                     # For System (Relyce), System wins. For Templates (Buddy), User wins.
                     pass 
                
                user_personalities.append(data)
    except Exception as e:
        print(f"[Personalities] Error loading: {e}")
        
    # Add User Personalities
    # If user has "buddy", we use that. If not, we use Template "buddy".
    
    user_ids = {p['id'] for p in user_personalities}
    
    # Add Templates if not overridden
    for p in TEMPLATE_PERSONALITIES:
        if p['id'] not in user_ids:
            personalities.append(p.copy())
            
    # Add all user personalities (including overridden templates)
    personalities.extend(user_personalities)
    
    return personalities

def create_custom_personality(user_id: str, name: str, description: str, prompt: str, content_mode: str = "hybrid", specialty: str = "general") -> Optional[Dict]:
    """
    Create a new custom personality for a user.
    content_mode: 'hybrid' (default), 'web_search', or 'llm_only'
    specialty: Expertise area like 'coding', 'ecommerce', 'music', etc.
    """
    # Validate content_mode
    valid_modes = ["hybrid", "web_search", "llm_only"]
    if content_mode not in valid_modes:
        content_mode = "hybrid"
    
    try:
        db = get_firestore_db()
        if not db:
            return None
        
        new_persona = {
            "name": name,
            "description": description,
            "prompt": prompt,
            "content_mode": content_mode,
            "specialty": specialty,
            "createdAt": str(uuid.uuid4())
        }
        
        doc_ref = db.collection('users').document(user_id).collection('personalities').add(new_persona)
        new_persona['id'] = doc_ref[1].id
        new_persona['is_default'] = False
        
        return new_persona
        
    except Exception as e:
        print(f"[Personalities] Error creating: {e}")
        return None

def get_personality_by_id(user_id: str, personality_id: str) -> Optional[Dict]:
    """
    Fetch a specific personality by ID.
    """
    # Check System
    for p in SYSTEM_PERSONALITIES:
        if p['id'] == personality_id:
            result = p.copy()
            if personality_id == "default_relyce":
                behavior_mode = _get_relyce_behavior_mode(user_id)
                if behavior_mode:
                    result["content_mode"] = behavior_mode
            return result
            
    # Check DB (User Custom or Shadowed Template)
    try:
        db = get_firestore_db()
        if db and user_id != "anonymous":
            doc = db.collection('users').document(user_id).collection('personalities').document(personality_id).get()
            if doc.exists:
                data = doc.to_dict()
                data['id'] = doc.id
                return data
    except Exception as e:
        print(f"[Personalities] Error getting by ID: {e}")
        
    # Check Templates (if not in DB)
    for p in TEMPLATE_PERSONALITIES:
        if p['id'] == personality_id:
            return p
            
    return None

def update_custom_personality(user_id: str, personality_id: str, name: str, description: str, prompt: str, content_mode: str = "hybrid", specialty: str = "general") -> bool:
    """
    Update a personality. 
    If it's a Template ID (like 'buddy') and doesn't exist in DB, create it (Shadowing).
    If it's a System ID, fail.
    content_mode: 'hybrid' (default), 'web_search', or 'llm_only'
    specialty: Expertise area like 'coding', 'ecommerce', 'music', etc.
    """
    # Validate content_mode
    valid_modes = ["hybrid", "web_search", "llm_only"]
    if content_mode not in valid_modes:
        content_mode = "hybrid"
    
    # Explicit check for system locked
    if any(p['id'] == personality_id for p in SYSTEM_PERSONALITIES):
        return False
        
    try:
        db = get_firestore_db()
        if not db:
            return False
            
        doc_ref = db.collection('users').document(user_id).collection('personalities').document(personality_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            # Check if it is a valid Template we allow shadowing for
            if any(p['id'] == personality_id for p in TEMPLATE_PERSONALITIES):
                # Create the shadow copy
                doc_ref.set({
                    "name": name,
                    "description": description,
                    "prompt": prompt,
                    "content_mode": content_mode,
                    "specialty": specialty,
                    "is_shadow": True,
                    "createdAt": str(uuid.uuid4())
                })
                return True
            else:
                return False  # Cannot update non-existent random ID
        else:
            # Normal update
            doc_ref.update({
                "name": name,
                "description": description,
                "prompt": prompt,
                "content_mode": content_mode,
                "specialty": specialty
            })
            return True
        
    except Exception as e:
        print(f"[Personalities] Error updating: {e}")
        return False

def delete_custom_personality(user_id: str, personality_id: str) -> bool:
    """
    Delete a custom personality.
    If it was shadowing a template, this effectively 'resets' to the default template.
    """
    if any(p['id'] == personality_id for p in SYSTEM_PERSONALITIES):
        return False
        
    try:
        db = get_firestore_db()
        if not db:
            return False
            
        doc_ref = db.collection('users').document(user_id).collection('personalities').document(personality_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            return False
            
        doc_ref.delete()
        return True
        
    except Exception as e:
        print(f"[Personalities] Error deleting: {e}")
        return False
