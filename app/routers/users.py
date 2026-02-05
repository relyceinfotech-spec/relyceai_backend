"""
Users Router
Handles user initialization and management.
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from app.auth import verify_token, get_firestore_db, get_current_user
from datetime import datetime, timezone
import threading

router = APIRouter()
_id_lock = threading.Lock()

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

    expiry_value = membership.get("expiryDate") # Using expiryDate as per frontend schema
    if not expiry_value:
        return # Skip if no expiry date (e.g. lifetime/unknown)

    # 2. Expiry Logic
    try:
        # Parse stored date. Accept ISO strings and Firestore Timestamps.
        if hasattr(expiry_value, "to_datetime"):
            expiry_date = expiry_value.to_datetime()
        elif isinstance(expiry_value, datetime):
            expiry_date = expiry_value
        else:
            expiry_date = datetime.fromisoformat(str(expiry_value).replace('Z', '+00:00'))
        
        # If naive (no timezone), force it to UTC
        if expiry_date.tzinfo is None:
             expiry_date = expiry_date.replace(tzinfo=timezone.utc)
        
        # Get current time in UTC
        now = datetime.now(timezone.utc) 
        
        # Debug Log (Helpful to see why users are expiring)
        print(f"[Users] Expiry Check {uid}: Now={now} vs Expiry={expiry_date}")

        if now > expiry_date:
            print(f"[Users] Expiry: Downgrading user {uid} from {current_plan} to free. (Expired at {expiry_date})")
            
            # 3. Downgrade & Log
            updates = {
                "membership.plan": "free",
                "membership.planName": "Free",
                "membership.status": "expired",
                "membership.isExpired": True, 
                "membership.updatedAt": datetime.now(timezone.utc)
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
                "timestamp": datetime.now(timezone.utc)
            })
            
    except Exception as e:
        print(f"[Users] Expiry check failed for {uid}: {e}")


def normalize_role(role_value: str) -> str:
    """Normalize role strings without downgrading."""
    if not role_value:
        return ""
    role = str(role_value).strip().lower()
    if role == "super_admin":
        return "superadmin"
    return role


def infer_plan_from_membership(membership: dict) -> str:
    """Infer plan safely from membership data without guessing."""
    if not membership:
        return ""
    if membership.get("plan"):
        return str(membership.get("plan")).lower()
    plan_name = membership.get("planName")
    if plan_name:
        return str(plan_name).strip().lower()
    payment_history = membership.get("paymentHistory")
    if isinstance(payment_history, dict) and payment_history:
        try:
            # Pick most recent payment by date field if present
            def get_date(item):
                data = item[1] or {}
                return data.get("date") or ""
            latest = max(payment_history.items(), key=get_date)
            plan = latest[1].get("plan")
            if plan:
                return str(plan).lower()
        except Exception:
            pass
    return ""

def generate_unique_id(db):
    """
    Generates a unique ID like RA001, RA002 using a Firestore counter.
    Uses simple get/set with lock instead of transaction (which can fail on named DBs).
    """
    counter_ref = db.collection('counters').document('userIds')
    
    with _id_lock:  # Thread safety within this instance
        try:
            # Get current counter
            counter_doc = counter_ref.get()
            
            if counter_doc.exists:
                current_id = counter_doc.to_dict().get("currentId", 0)
                if not isinstance(current_id, int):
                    current_id = int(current_id) if current_id else 0
                new_id = current_id + 1
            else:
                new_id = 1
            
            # Update counter
            counter_ref.set({
                "currentId": new_id,
                "lastUpdated": datetime.now(timezone.utc)
            }, merge=True)
            
            result_id = f"RA{new_id:03d}"
            print(f"[Users] Generated ID: {result_id}")
            return result_id
            
        except Exception as e:
            print(f"[Users] ID Generation error: {e}")
            # Fallback: Just use a timestamp-based random ID to prevent blocking user creation
            # The previous "scan all users" fallback was too dangerous/slow.
            import random
            fallback_id = f"RA{int(datetime.now().timestamp())}{random.randint(10,99)}"
            print(f"[Users] Using fallback timestamp ID: {fallback_id}")
            return fallback_id


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
            updates = {}
            
            # HARDENING: For existing users, do NOT modify role or membership via /users/init.
            # This prevents unintended downgrades. Only assign uniqueUserId if the field is missing.

            if "uniqueUserId" not in user_data:
                new_unique_id = generate_unique_id(db)
                print(f"[Users] Init: Assigning new ID {new_unique_id} to {uid} (field missing)")
                updates["uniqueUserId"] = new_unique_id

            if updates:
                user_ref.update(updates)
                # Re-fetch user_data if we updated it, to ensure expiry check uses fresh data
                # BUT: updates variable only has partial keys. 
                # Simplest: Update user_data dict locally for the check
                user_data.update(updates)
                
            # --- EXPIRY CHECK ---
            # Disabled in /users/init to avoid accidental downgrades.

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
                "createdAt": datetime.now(timezone.utc),
                "lastLoginAt": datetime.now(timezone.utc),
                "membership": {
                    "plan": "free",
                    "planName": "Free",
                    "status": "active",
                    "billingCycle": "monthly",
                    "paymentStatus": "free",
                    "startDate": datetime.now(timezone.utc).isoformat()
                }
            }
            user_ref.set(new_user_data, merge=True)
            return {"success": True, "status": "created"}
            
    except Exception as e:
        print(f"[Users] Init error for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize user")
