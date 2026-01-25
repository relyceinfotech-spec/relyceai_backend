"""
Relyce AI - Chat History Manager
Saves/loads chat history with Firebase Firestore
"""
from typing import List, Dict, Optional
from datetime import datetime
from app.auth import get_firestore_db, initialize_firebase

def save_message_to_firebase(
    user_id: str,
    session_id: str,
    role: str,
    content: str
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
        
        doc_ref = messages_ref.add({
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'createdAt': datetime.now().isoformat()
        })
        
        return doc_ref[1].id
        
    except Exception as e:
        print(f"[History] Error saving message: {e}")
        return None

def load_chat_history(
    user_id: str,
    session_id: str,
    limit: int = 50
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
        docs = query.stream()
        
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                'id': doc.id,
                'role': data.get('role'),
                'content': data.get('content'),
                'timestamp': data.get('createdAt')
            })
        
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
