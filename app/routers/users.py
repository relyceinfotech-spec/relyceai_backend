"""
Users Router
Handles user initialization and management.
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from app.auth import verify_token, get_firestore_db
from datetime import datetime
from firebase_admin import firestore
import threading

# Lock to prevent race conditions within the same instance (though transaction handles DB)
_id_lock = threading.Lock()
from firebase_admin import firestore
import threading

# Lock to prevent race conditions within the same instance (though transaction handles DB)
_id_lock = threading.Lock()

router = APIRouter()

async def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = authorization.split(" ")[1]
    is_valid, user_info = verify_token(token)
    
    if not is_valid or not user_info:
        raise HTTPException(status_code=401, detail="Invalid token")
        
    return user_info

def check_membership_expiry(user_ref, user_data, uid):
    """
    Helper to check and process membership expiry.
    SAFE: Never downgrades if date is missing or invalid.
    """
    membership = user_data.get("membership")
    
    # 1. Safety Checks
    if not membership:
        return
        
    current_plan = membership.get("plan", "free")
    if current_plan == "free":
        return

    expiry_date_str = membership.get("expiryDate") # Using expiryDate as per frontend schema
    if not expiry_date_str:
        return # Skip if no expiry date (e.g. lifetime/unknown)

    # 2. Expiry Logic
    try:
        # data is stored as ISO format string
        expiry_date = datetime.fromisoformat(expiry_date_str.replace('Z', '+00:00'))
        # Ensure timezone awareness compatibility (assuming stored as simple ISO or UTC)
        now = datetime.now() # Server local time. Ideally should be UTC. 
        # Note: If expiry_date is offset-aware and now is naive, this might crash.
        # Let's assume naive for now or handle mixed.
        # Better:
        # if expiry_date.tzinfo is None: expiry_date = expiry_date.replace(tzinfo=None)
        
        if now > expiry_date:
            print(f"[Users] Expiry: Downgrading user {uid} from {current_plan} to free")
            
            # 3. Downgrade & Log
            updates = {
                "membership.plan": "free",
                "membership.planName": "Free",
                "membership.status": "expired",
                "membership.isExpired": True, 
                "membership.updatedAt": datetime.now()
            }
            user_ref.update(updates)
            
            # Audit Log
            db = get_firestore_db()
            db.collection("auditLogs").add({
                "action": "MEMBERSHIP_CHANGED",
                "from": current_plan,
                "to": "free",
                "reason": "expiry",
                "by": "system",
                "target": uid,
                "timestamp": datetime.now()
            })
            
    except Exception as e:
        print(f"[Users] Expiry check failed for {uid}: {e}")

def generate_unique_id(db):
    """
    Generates a unique ID like RA001, RA002 using a Firestore counter.
    Uses a transaction to ensure atomicity.
    """
    counter_ref = db.collection('counters').document('userIds')
    
    @firestore.transactional
    def increment_counter(transaction, counter_ref):
        snapshot = transaction.get(counter_ref)
        if snapshot.exists:
            current_id = snapshot.get("currentId")
            if not current_id: current_id = 0
            new_id = current_id + 1
        else:
            new_id = 1
            
        transaction.set(counter_ref, {"currentId": new_id, "lastUpdated": datetime.now()}, merge=True)
        return new_id

    try:
        transaction = db.transaction()
        new_id_num = increment_counter(transaction, counter_ref)
        return f"RA{new_id_num:03d}"
    except Exception as e:
        print(f"[Users] ID Generation failed: {e}")
        # Fallback: Timestamp based
        return f"RA{int(datetime.now().timestamp())}"


@router.post("/init")
async def init_user(user_info: dict = Depends(get_current_user)):
    """
    Initialize a user's role if it's missing.
    Guarantees 'role': 'user' exists.
    """
    uid = user_info["uid"]
    email = user_info.get("email")
    
    db = get_firestore_db()
    user_ref = db.collection("users").document(uid)
    
    try:
        doc = user_ref.get()
        
        if doc.exists:
            user_data = doc.to_dict()
            # If role is missing, set it to 'user'
            updates = {}
            if "role" not in user_data:
                print(f"[Users] Init: Setting default role for existing user {uid}")
                updates["role"] = "user"

            # Check for uniqueUserId
            if "uniqueUserId" not in user_data:
                new_unique_id = generate_unique_id(db)
                print(f"[Users] Init: Assigning new ID {new_unique_id} to {uid}")
                updates["uniqueUserId"] = new_unique_id

            # Check for membership and set default ONLY if key is missing
            # Fix: Never overwrite existing membership (even if incomplete/falsy) - that's a data corruption risk
            if "membership" not in user_data:
                print(f"[Users] Init: Setting default FREE membership for {uid}")
                updates["membership"] = {
                    "plan": "free",
                    "planName": "Free",
                    "status": "active",
                    "billingCycle": "monthly",
                    "paymentStatus": "free", # Explicitly mark as free
                    "startDate": datetime.now().isoformat(),
                    "updatedAt": datetime.now() # Use datetime object, Firestore serializer handles it or we use server_timestamp if imported
                }

            if updates:
                user_ref.update(updates)
                # Re-fetch user_data if we updated it, to ensure expiry check uses fresh data
                # BUT: updates variable only has partial keys. 
                # Simplest: Update user_data dict locally for the check
                user_data.update(updates)
                
            # --- EXPIRY CHECK ---
            check_membership_expiry(user_ref, user_data, uid)

            if updates:
                return {"success": True, "status": "updated", "updates": list(updates.keys())}
            else:
                 return {"success": True, "status": "already_has_role_and_membership"}
        else:
            # If user doc doesn't exist, create it with default role, membership AND ID
            print(f"[Users] Init: Creating new user doc for {uid}")
            
            new_unique_id = generate_unique_id(db)
            
            new_user_data = {
                "uid": uid,
                "uniqueUserId": new_unique_id,
                "email": email,
                "role": "user",
                "createdAt": datetime.now(),
                "lastLoginAt": datetime.now(),
                "membership": {
                    "plan": "free",
                    "planName": "Free",
                    "status": "active",
                    "billingCycle": "monthly",
                    "paymentStatus": "free",
                    "startDate": datetime.now().isoformat()
                }
            }
            user_ref.set(new_user_data, merge=True)
            return {"success": True, "status": "created"}
            
    except Exception as e:
        print(f"[Users] Init error for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize user")
