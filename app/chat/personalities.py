"""
Relyce AI - Chat Personalities Manager
Saves/loads custom personalities with Firebase Firestore
"""
from typing import List, Dict, Optional, Any
import uuid
from app.auth import get_firestore_db
from app.llm.router import DEFAULT_PERSONA

# System Locked Personalities (Cannot be edited/deleted)
SYSTEM_PERSONALITIES = [
    {
        "id": "default_relyce",
        "name": "Relyce AI",
        "description": "Professional, helpful, and empathetic AI assistant.",
        "prompt": DEFAULT_PERSONA,
        "is_default": True,
        "is_system": True # Locked
    }
]

# Default Templates (Can be edited/overridden by users)
TEMPLATE_PERSONALITIES = [
    {
        "id": "buddy",
        "name": "Buddy",
        "description": "A casual, friendly companion who speaks like a close friend.",
        "prompt": "You are a friendly, casual companion named Buddy. Speak like a close friend, use slang where appropriate, and keep it chill. 🤙",
        "is_default": True,
        "is_system": False # Editable
    }
]

def get_all_personalities(user_id: str) -> List[Dict]:
    """
    Get all personalities for a user.
    Merges System defaults + User overrides + User custom.
    """
    # Start with System Locked ones
    personalities = [p.copy() for p in SYSTEM_PERSONALITIES]
    
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

def create_custom_personality(user_id: str, name: str, description: str, prompt: str) -> Optional[Dict]:
    """
    Create a new custom personality for a user.
    """
    try:
        db = get_firestore_db()
        if not db:
            return None
        
        new_persona = {
            "name": name,
            "description": description,
            "prompt": prompt,
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
            return p
            
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

def update_custom_personality(user_id: str, personality_id: str, name: str, description: str, prompt: str) -> bool:
    """
    Update a personality. 
    If it's a Template ID (like 'buddy') and doesn't exist in DB, create it (Shadowing).
    If it's a System ID, fail.
    """
    # explicit check for system locked
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
                    "is_shadow": True,
                    "createdAt": str(uuid.uuid4())
                })
                return True
            else:
                return False # Cannot update non-existent random ID
        else:
            # Normal update
            doc_ref.update({
                "name": name,
                "description": description,
                "prompt": prompt
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


    except Exception as e:
        print(f"[Personalities] Error deleting: {e}")
        return False
