"""
Relyce AI - Chat History Manager
Saves/loads chat history with Firebase Firestore
"""
from typing import List, Dict, Optional
from datetime import datetime
import time
from app.auth import get_firestore_db, initialize_firebase
from firebase_admin import firestore
from app.utils.sanitize import sanitize_message_content


def save_message_to_firebase(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    personality_id: Optional[str] = None
) -> Optional[str]:
    """
    Save a message to Firebase Firestore.
    Path: users/{user_id}/chatSessions/{session_id}/messages/{message_id}
    
    Returns: message_id or None on error
    """
    try:
        db = get_firestore_db()
        if not db:
            print("[History] Firestore not available")
            return None
        
        messages_ref = db.collection('users').document(user_id) \
            .collection('chatSessions').document(session_id) \
            .collection('messages')
        
        safe_content = sanitize_message_content(content)
        payload = {
            'role': role,
            'content': safe_content,
            'timestamp': datetime.now(),
            'createdAt': datetime.now().isoformat()
        }
        if personality_id:
            payload['personalityId'] = personality_id
            
        # Retry logic for saving message
        max_retries = 3
        for attempt in range(max_retries):
            try:
                doc_ref = messages_ref.add(payload)
                return doc_ref[1].id
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 0.5 * (2 ** attempt)
                    print(f"[History] Save failed (attempt {attempt+1}/{max_retries}), retrying... Error: {e}")
                    time.sleep(sleep_time)
                else:
                    raise e
        
    except Exception as e:
        print(f"[History] Error saving message: {e}")
        return None

def load_chat_history(
    user_id: str,
    session_id: str,
    limit: int = 50,
    personality_id: Optional[str] = None
) -> List[Dict]:
    """
    Load chat history from Firebase Firestore.
    Returns list of messages ordered by timestamp.
    """
    try:
        db = get_firestore_db()
        if not db:
            return []
        
        messages_ref = db.collection('users').document(user_id) \
            .collection('chatSessions').document(session_id) \
            .collection('messages')
        
        query = messages_ref.order_by('timestamp').limit(limit)
        
        # Retry logic for loading history
        max_retries = 3
        messages = []
        
        for attempt in range(max_retries):
            try:
                docs = query.stream()
                temp_messages = []
                for doc in docs:
                    data = doc.to_dict()
                    temp_messages.append({
                        'id': doc.id,
                        'role': data.get('role'),
                        'content': data.get('content'),
                        'timestamp': data.get('createdAt'),
                        'personalityId': data.get('personalityId')
                    })
                messages = temp_messages
                break # Success
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = 0.5 * (2 ** attempt)
                    print(f"[History] Load failed (attempt {attempt+1}/{max_retries}), retrying... Error: {e}")
                    time.sleep(sleep_time)
                else:
                    raise e
        
        if personality_id:
            messages = [m for m in messages if m.get('personalityId') == personality_id]
        
        return messages
        
    except Exception as e:
        print(f"[History] Error loading history: {e}")
        return []

def update_session_name(user_id: str, session_id: str, name: str) -> bool:
    """Update the name of a chat session"""
    try:
        db = get_firestore_db()
        if not db:
            return False
        
        session_ref = db.collection('users').document(user_id) \
            .collection('chatSessions').document(session_id)
        
        session_ref.update({'name': name})
        return True
        
    except Exception as e:
        print(f"[History] Error updating session name: {e}")
        return False

def create_chat_session(user_id: str, name: str = "New Chat") -> Optional[str]:
    """Create a new chat session"""
    try:
        db = get_firestore_db()
        if not db:
            return None
        
        sessions_ref = db.collection('users').document(user_id) \
            .collection('chatSessions')
        
        doc_ref = sessions_ref.add({
            'name': name,
            'createdAt': datetime.now(),
            'updatedAt': datetime.now()
        })
        
        return doc_ref[1].id
        
    except Exception as e:
        print(f"[History] Error creating session: {e}")
        return None

def get_user_sessions(user_id: str, limit: int = 50) -> List[Dict]:
    """Get all chat sessions for a user"""
    try:
        db = get_firestore_db()
        if not db:
            return []
        
        sessions_ref = db.collection('users').document(user_id) \
            .collection('chatSessions')
        
        query = sessions_ref.order_by('updatedAt', direction='DESCENDING').limit(limit)
        docs = query.stream()
        
        sessions = []
        for doc in docs:
            data = doc.to_dict()
            sessions.append({
                'id': doc.id,
                'name': data.get('name', 'Untitled'),
                'createdAt': data.get('createdAt'),
                'updatedAt': data.get('updatedAt')
            })
        
        return sessions
        
    except Exception as e:
        print(f"[History] Error getting sessions: {e}")
        return []

def increment_message_count(user_id: str):
    """
    Increment the totalMessages count for a user in Firestore.
    Securely tracked on backend.
    """
    try:
        db = get_firestore_db()
        if not db: return

        user_ref = db.collection('users').document(user_id)
        
        # Retry logic for incrementing count
        max_retries = 3
        for attempt in range(max_retries):
            try:
                user_ref.update({
                    "usage.totalMessages": firestore.Increment(1),
                    "usage.lastActivity": firestore.SERVER_TIMESTAMP
                })
                return
            except Exception as e:
                # If usage map doesn't exist, this will fail with NOT_FOUND-like error, 
                # catch that specific case? 
                # Actually original code had a specific 'except' block for the update to try set/merge.
                # Let's keep the logic simple: verify if it's a network error or logic error.
                # If update fails, we try the fallback block below.
                # But we want to retry on NETWORK errors.
                # Let's wrap the update call.
                if attempt < max_retries - 1:
                     # Simple retry for now
                     time.sleep(0.5 * (2 ** attempt))
                else:
                    raise e

    except Exception as e:
        # If usage map doesn't exist, try set merge
        try:
            db.collection('users').document(user_id).set({
                "usage": {
                    "totalMessages": firestore.Increment(1),
                    "lastActivity": firestore.SERVER_TIMESTAMP
                }
            }, merge=True)
        except Exception as inner_e:
            print(f"[History] Failed to increment message count: {inner_e}")
