"""
Users Router
Handles user initialization and management.
"""
from fastapi import APIRouter, HTTPException, Depends
from app.auth import get_firestore_db, get_current_user, get_claim_role
from datetime import datetime, timezone
import threading
from firebase_admin import firestore

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

    expiry_value = membership.get("expiryDate")  # Using expiryDate as per frontend schema
    if not expiry_value:
        return  # Skip if no expiry date (e.g. lifetime/unknown)

    def _parse_expiry(value):
        if hasattr(value, "to_datetime"):
            return value.to_datetime()
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    try:
        expiry_date = _parse_expiry(expiry_value)
        if expiry_date.tzinfo is None:
            expiry_date = expiry_date.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        print(f"[Users] Expiry Check {uid}: Now={now} vs Expiry={expiry_date}")

        if now <= expiry_date:
            return

        db = get_firestore_db()
        if not db:
            return

        @firestore.transactional
        def _expire_membership(transaction):
            snap = user_ref.get(transaction=transaction)
            if not snap.exists:
                return False
            data = snap.to_dict() or {}
            membership_current = data.get("membership") or {}
            plan_current = membership_current.get("plan", "free")
            if plan_current == "free":
                return False
            expiry_current = membership_current.get("expiryDate")
            if not expiry_current:
                return False

            try:
                expiry_current_dt = _parse_expiry(expiry_current)
                if expiry_current_dt.tzinfo is None:
                    expiry_current_dt = expiry_current_dt.replace(tzinfo=timezone.utc)
            except Exception:
                return False

            if datetime.now(timezone.utc) <= expiry_current_dt:
                return False

            updates = {
                "membership.plan": "free",
                "membership.planName": "Free",
                "membership.status": "expired",
                "membership.isExpired": True,
                "membership.updatedAt": datetime.now(timezone.utc),
            }
            transaction.update(user_ref, updates)

            audit_ref = db.collection("auditLogs").document()
            transaction.set(audit_ref, {
                "action": "MEMBERSHIP_CHANGED",
                "from": plan_current,
                "to": "free",
                "reason": "expiry",
                "by": "system",
                "target": uid,
                "timestamp": datetime.now(timezone.utc),
            })
            return True

        _expire_membership(db.transaction())
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
    Uses atomic Firestore Increment for race-condition-safe ID generation.
    """
    counter_ref = db.collection('counters').document('userIds')
    
    try:
        @firestore.transactional
        def _increment_counter_transaction(transaction):
            doc = counter_ref.get(transaction=transaction)
            current_id = 0
            if doc.exists:
                current_id = doc.to_dict().get("currentId", 0)
                if not isinstance(current_id, int):
                    current_id = int(current_id) if current_id else 0
            new_id = current_id + 1
            transaction.set(counter_ref, {
                "currentId": new_id,
                "lastUpdated": datetime.now(timezone.utc)
            }, merge=True)
            return new_id
        
        new_id = _increment_counter_transaction(db.transaction())
        result_id = f"RA{new_id:03d}"
        print(f"[Users] Generated ID: {result_id}")
        return result_id
        
    except Exception as e:
        print(f"[Users] ID Generation error: {e}")
        import random
        fallback_id = f"RA{int(datetime.now().timestamp())}{random.randint(10,99)}"
        print(f"[Users] Using fallback timestamp ID: {fallback_id}")
        return fallback_id


@router.post("/init")
async def init_user(user_info: dict = Depends(get_current_user)):
    """
    Initialize a user's role if it's missing.
    Guarantees 'role': 'user' exists.
    ALWAYS returns uniqueUserId.
    """
    uid = user_info["uid"]
    email = user_info.get("email")
    
    db = get_firestore_db()
    user_ref = db.collection("users").document(uid)
    
    try:
        doc = user_ref.get()
        
        claim_role = get_claim_role(user_info)
        if doc.exists:
            user_data = doc.to_dict()
            updates = {}
            
            if not user_data.get("uniqueUserId"):
                new_unique_id = generate_unique_id(db)
                print(f"[Users] Init: Assigning new ID {new_unique_id} to {uid} (field missing)")
                updates["uniqueUserId"] = new_unique_id

            existing_role = normalize_role(user_data.get("role"))
            if not existing_role and not claim_role:
                updates["role"] = "user"
                print(f"[Users] Init: Assigning default role 'user' to {uid}")
            elif claim_role and claim_role != existing_role:
                updates["role"] = claim_role
                print(f"[Users] Init: Syncing role from claims '{claim_role}' for {uid}")

            if not user_data.get("membership") or not user_data.get("membership", {}).get("plan"):
                updates["membership"] = {
                    "plan": "free",
                    "planName": "Free",
                    "status": "active",
                    "billingCycle": "monthly",
                    "paymentStatus": "free",
                    "startDate": datetime.now(timezone.utc).isoformat()
                }
                print(f"[Users] Init: Assigning default free membership to {uid}")

            if updates:
                user_ref.update(updates)
                user_data.update(updates)
                
            final_unique_id = user_data.get("uniqueUserId") or updates.get("uniqueUserId")
            if not final_unique_id:
                new_id = generate_unique_id(db)
                user_ref.update({"uniqueUserId": new_id})
                final_unique_id = new_id
                print(f"[Users] Init: Emergency ID assignment {new_id} to {uid}")

            return {
                "success": True,
                "status": "updated" if updates else "already_initialized",
                "updates": list(updates.keys()) if updates else [],
                "uniqueUserId": final_unique_id
            }
        else:
            print(f"[Users] Init: Creating new user doc for {uid}")
            
            new_unique_id = generate_unique_id(db)
            
            new_user_data = {
                "uid": uid,
                "uniqueUserId": new_unique_id,
                "email": email,
                "role": claim_role or "user",
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
            return {"success": True, "status": "created", "uniqueUserId": new_unique_id}
            
    except Exception as e:
        print(f"[Users] Init error for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize user")

@router.get("/me")
async def get_me(user_info: dict = Depends(get_current_user)):
    """
    Return the authenticated user's profile document.
    """
    uid = user_info["uid"]
    db = get_firestore_db()
    user_ref = db.collection("users").document(uid)
    doc = user_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="User profile not found")
    return {"success": True, "user": doc.to_dict()}


@router.post("/membership/downgrade")
async def downgrade_membership(user_info: dict = Depends(get_current_user)):
    """
    Allow authenticated users to downgrade themselves to the Free plan.
    Paid upgrades must go through the payment flow.
    """
    uid = user_info["uid"]
    db = get_firestore_db()
    user_ref = db.collection("users").document(uid)

    now = datetime.now(timezone.utc)
    updates = {
        "membership.plan": "free",
        "membership.planName": "Free",
        "membership.status": "active",
        "membership.billingCycle": "monthly",
        "membership.paymentStatus": "free",
        "membership.expiryDate": None,
        "membership.isExpired": False,
        "membership.updatedAt": now
    }

    try:
        user_ref.update(updates)
        db.collection("auditLogs").add({
            "action": "MEMBERSHIP_CHANGED",
            "by": uid,
            "target": uid,
            "details": {"plan": "free", "billingCycle": "monthly"},
            "timestamp": now,
            "time": now.timestamp()
        })
        return {"success": True, "message": "Downgraded to Free plan"}
    except Exception as e:
        print(f"[Users] Downgrade error for {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to downgrade membership")
