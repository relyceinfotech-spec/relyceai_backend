"""
User profile helpers: settings and session personality lookup.
"""
from typing import Optional, Dict, Any
from app.auth import get_firestore_db


def get_user_settings(user_id: str) -> Optional[Dict[str, Any]]:
    """Load user settings from Firestore."""
    try:
        db = get_firestore_db()
        if not db:
            return None
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            data = doc.to_dict() or {}
            return data.get("settings")
    except Exception as e:
        print(f"[UserSettings] Failed to load settings for {user_id}: {e}")
    return None


def get_session_personality_id(user_id: str, session_id: str) -> Optional[str]:
    """Load saved personalityId for a chat session."""
    try:
        db = get_firestore_db()
        if not db:
            return None
        doc = (
            db.collection("users")
            .document(user_id)
            .collection("chatSessions")
            .document(session_id)
            .get()
        )
        if doc.exists:
            data = doc.to_dict() or {}
            return data.get("personalityId")
    except Exception as e:
        print(f"[UserSettings] Failed to load session personality for {session_id}: {e}")
    return None


def merge_settings(db_settings: Optional[Dict[str, Any]], request_settings: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Merge DB settings with request settings, preferring request values."""
    if not db_settings and not request_settings:
        return None
    if not db_settings:
        return request_settings
    if not request_settings:
        return db_settings

    merged = {**db_settings, **request_settings}
    db_personalization = db_settings.get("personalization") if isinstance(db_settings, dict) else None
    req_personalization = request_settings.get("personalization") if isinstance(request_settings, dict) else None
    if isinstance(db_personalization, dict) or isinstance(req_personalization, dict):
        merged["personalization"] = {
            **(db_personalization or {}),
            **(req_personalization or {})
        }
    return merged
