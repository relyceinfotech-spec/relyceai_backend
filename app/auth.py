"""
Relyce AI - Firebase Authentication
Token verification for REST API and WebSocket
"""
import firebase_admin
from firebase_admin import credentials, auth, firestore
from typing import Optional, Tuple
from app.config import (
    FIREBASE_PROJECT_ID,
    FIREBASE_PRIVATE_KEY_ID,
    FIREBASE_PRIVATE_KEY,
    FIREBASE_CLIENT_EMAIL,
    FIREBASE_CLIENT_ID,
    FIREBASE_CLIENT_ID,
    FIREBASE_CLIENT_CERT_URL,
    FIREBASE_DATABASE_ID
)
from google.cloud import firestore as google_firestore

# Initialize Firebase Admin SDK
_firebase_initialized = False
_db = None

def initialize_firebase():
    """Initialize Firebase Admin SDK with credentials from environment"""
    global _firebase_initialized, _db
    
    if _firebase_initialized:
        return
    
    try:
        # Build credentials dict from environment variables
        cred_dict = {
            "type": "service_account",
            "project_id": FIREBASE_PROJECT_ID,
            "private_key_id": FIREBASE_PRIVATE_KEY_ID,
            "private_key": FIREBASE_PRIVATE_KEY,
            "client_email": FIREBASE_CLIENT_EMAIL,
            "client_id": FIREBASE_CLIENT_ID,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": FIREBASE_CLIENT_CERT_URL
        }
        
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        if FIREBASE_DATABASE_ID and FIREBASE_DATABASE_ID != "(default)":
            _db = google_firestore.Client(
                credentials=cred.get_credential(),
                project=FIREBASE_PROJECT_ID,
                database=FIREBASE_DATABASE_ID
            )
            print(f"[Firebase] Connected to named DB: {FIREBASE_DATABASE_ID}")
        else:
            _db = firestore.client()
            print("[Firebase] Connected to default DB")
        _firebase_initialized = True
        print("[Firebase] ✅ Initialized successfully")
        
    except Exception as e:
        print(f"[Firebase] ❌ Initialization failed: {e}")
        raise

def get_firestore_db():
    """Get Firestore database client"""
    global _db
    if not _firebase_initialized:
        initialize_firebase()
    return _db

def verify_token(id_token: str) -> Tuple[bool, Optional[dict]]:
    """
    Verify Firebase ID token
    Returns: (is_valid, user_info)
    """
    if not _firebase_initialized:
        initialize_firebase()
    
    try:
        decoded_token = auth.verify_id_token(id_token)
        user_info = {
            "uid": decoded_token.get("uid"),
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),
        }
        return True, user_info
    except auth.InvalidIdTokenError:
        print("[Auth] Invalid ID token")
        return False, None
    except auth.ExpiredIdTokenError:
        print("[Auth] Expired ID token")
        return False, None
    except Exception as e:
        print(f"[Auth] Token verification error: {e}")
        return False, None

def get_user_by_uid(uid: str) -> Optional[dict]:
    """Get user information by UID"""
    if not _firebase_initialized:
        initialize_firebase()
    
    try:
        user = auth.get_user(uid)
        return {
            "uid": user.uid,
            "email": user.email,
            "display_name": user.display_name,
            "photo_url": user.photo_url,
        }
    except Exception as e:
        print(f"[Auth] Get user error: {e}")
        return None

# Auto-initialize on import
try:
    initialize_firebase()
except Exception:
    print("[Firebase] Will initialize on first use")
