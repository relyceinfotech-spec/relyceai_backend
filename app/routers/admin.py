"""
Admin Router
Handles administrative actions like role management and audit logging.
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import Optional
import time
from app.auth import get_firestore_db, get_current_user, is_admin_user, is_superadmin_user, normalize_role
from datetime import datetime, timedelta, timezone
from firebase_admin import auth as firebase_auth
# Import reusable expiry logic
from app.routers.users import check_membership_expiry, generate_unique_id

router = APIRouter()

class ChangeRoleRequest(BaseModel):
    target_uid: str
    new_role: str


class UpdateMembershipRequest(BaseModel):
    target_uid: str
    plan: str
    billing_cycle: str = "monthly"
    payment: Optional[dict] = None


def require_admin(user_info: dict = Depends(get_current_user)):
    if not is_admin_user(user_info):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user_info

def require_superadmin(user_info: dict = Depends(get_current_user)):
    if not is_superadmin_user(user_info):
        raise HTTPException(status_code=403, detail="Insufficient permissions: SuperAdmin only")
    return user_info

@router.post("/change-role")
async def change_role(request: ChangeRoleRequest, user_info: dict = Depends(require_superadmin)):
    """
    Change a user's role.
    Only SuperAdmin can perform this action.
    """
    db = get_firestore_db()
    
    # 1. Update Target User Role
    target_ref = db.collection("users").document(request.target_uid)
    target_doc = target_ref.get()
    
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Target user not found")
        
    try:
        # Normalize and validate role
        new_role = normalize_role(request.new_role)
        if new_role not in ["user", "premium", "admin", "superadmin"]:
            raise HTTPException(status_code=400, detail="Invalid role")

        # Update Firebase custom claims (preserve existing claims)
        user_record = firebase_auth.get_user(request.target_uid)
        claims = user_record.custom_claims or {}
        claims.update({
            "role": new_role,
            "admin": new_role in ["admin", "superadmin"],
            "superadmin": new_role == "superadmin"
        })
        firebase_auth.set_custom_user_claims(request.target_uid, claims)
        # Force re-auth so new role takes effect immediately
        firebase_auth.revoke_refresh_tokens(request.target_uid)

        # Mirror role in Firestore for UI display
        target_ref.update({
            "role": new_role,
            "roleChangedAt": datetime.now(timezone.utc)
        })
        
        # 3. Create Audit Log
        audit_entry = {
            "action": "ROLE_CHANGED",
            "by": user_info["uid"],
            "target": request.target_uid,
            "previous_role": target_doc.to_dict().get("role", "unknown"),
            "new_role": new_role,
            "timestamp": datetime.now(timezone.utc), # Python datetime for Firestore
            "time": time.time() # Unix timestamp for easier sorting/filtering if needed
        }
        
        db.collection("auditLogs").add(audit_entry)
        
        return {"success": True, "message": f"Role updated to {request.new_role}"}
        
    except Exception as e:
        print(f"[Admin] Role change error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update role")


@router.post("/membership/update")
async def update_membership(request: UpdateMembershipRequest, user_info: dict = Depends(require_admin)):
    """
    Admin-only membership update through backend.
    """
    db = get_firestore_db()
    target_ref = db.collection("users").document(request.target_uid)
    target_doc = target_ref.get()
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Target user not found")

    now = datetime.now(timezone.utc)
    # Calculate expiry
    if request.billing_cycle == "yearly":
        expiry_date = now.replace(year=now.year + 1)
    else:
        expiry_date = now + timedelta(days=30)

    updates = {
        "membership.plan": request.plan,
        "membership.planName": request.plan.capitalize(),
        "membership.status": "active" if request.plan != "free" else "active",
        "membership.billingCycle": request.billing_cycle,
        "membership.paymentStatus": "paid" if request.plan != "free" else "free",
        "membership.startDate": now.isoformat(),
        "membership.expiryDate": None if request.plan == "free" else expiry_date.isoformat(),
        "membership.updatedAt": now,
    }

    if request.payment and request.plan != "free":
        payment_id = request.payment.get("transactionId") or request.payment.get("paymentId")
        if payment_id:
            updates[f"membership.paymentHistory.{payment_id}"] = {
                "transactionId": payment_id,
                "orderId": request.payment.get("orderId"),
                "amount": request.payment.get("amount"),
                "currency": request.payment.get("currency") or "INR",
                "method": request.payment.get("method") or "manual",
                "plan": request.plan,
                "billingCycle": request.billing_cycle,
                "date": now.isoformat(),
                "verified": True,
                "source": "admin"
            }

    target_ref.update(updates)

    db.collection("auditLogs").add({
        "action": "MEMBERSHIP_CHANGED",
        "by": user_info["uid"],
        "target": request.target_uid,
        "details": {
            "plan": request.plan,
            "billingCycle": request.billing_cycle
        },
        "timestamp": now,
        "time": now.timestamp()
    })

    return {"success": True, "message": "Membership updated"}


@router.delete("/users/{target_uid}")
async def delete_user(target_uid: str, user_info: dict = Depends(require_admin)):
    """
    Admin-only user deletion through backend.
    """
    db = get_firestore_db()
    target_ref = db.collection("users").document(target_uid)
    target_doc = target_ref.get()
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Target user not found")

    # Delete chat sessions and messages
    sessions = db.collection("users").document(target_uid).collection("chatSessions").stream()
    for session in sessions:
        msgs = session.reference.collection("messages").stream()
        for msg in msgs:
            msg.reference.delete()
        session.reference.delete()

    # Delete files metadata
    files = db.collection("users").document(target_uid).collection("files").stream()
    for f in files:
        f.reference.delete()

    # Delete folders
    folders = db.collection("users").document(target_uid).collection("folders").stream()
    for f in folders:
        f.reference.delete()

    # Delete shared chats
    shared = db.collection("sharedChats").where("userId", "==", target_uid).stream()
    for s in shared:
        s.reference.delete()

    target_ref.delete()

    now = datetime.now()
    db.collection("auditLogs").add({
        "action": "USER_DELETED",
        "by": user_info["uid"],
        "target": target_uid,
        "timestamp": now,
        "time": now.timestamp()
    })

    return {"success": True, "message": "User deleted"}

@router.post("/jobs/expire-memberships")
async def run_expiry_job(user_info: dict = Depends(require_superadmin)):
    """
    CRON JOB: Checks all active memberships and expires them if past valid date.
    Triggered by SuperAdmin (or authorized system service).
    """
    print(f"[Job] Expiry job triggered by {user_info['uid']}")
    db = get_firestore_db()

    # 2. Query Candidates: Active Users with Expiry Date < NOW
    # Note: Querying by date requires the Index we just added.
    # To be safe against index propagation delay or slight time diffs, 
    # we query 'active' and potentially filter by date in code if index fails, 
    # BUT standard is to use the index query.
    
    now_iso = datetime.now(timezone.utc).isoformat()
    
    try:
        # Use simple query on status first if index isn't ready, but let's try strict.
        # members = db.collection("users").where("membership.status", "==", "active").where("membership.expiryDate", "<", now_iso).stream()
        
        # Actually, for safety and to reuse our ROBUST logic in check_membership_expiry,
        # let's just fetch all 'active' members. 
        # If the DB grows huge, we MUST use the date filter index. 
        # For now, let's use the Index.
        
        docs = db.collection("users")\
                 .where("membership.status", "==", "active")\
                 .where("membership.expiryDate", "<", now_iso)\
                 .stream()
                 
        count = 0
        for doc in docs:
            # Our helper function does the Double-Check and Atomic Update + Log
            check_membership_expiry(doc.reference, doc.to_dict(), doc.id)
            count += 1
            
        print(f"[Job] Expiry job finished. Processed {count} potential expirations.")
        return {"success": True, "processed": count, "message": "Job completed"}
        
    except Exception as e:
        print(f"[Job] Failed to run expiry job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/backfill-unique-ids")
async def run_unique_id_backfill(user_info: dict = Depends(require_superadmin)):
    """
    CRON/ADMIN JOB: Backfill missing or malformed uniqueUserId for all users.
    Attempts to recover legacy IDs when possible; otherwise generates new RA IDs.
    """
    print(f"[Job] Unique ID backfill triggered by {user_info['uid']}")

    db = get_firestore_db()

    users = db.collection("users").stream()
    updated = 0
    recovered = 0
    converted = 0
    skipped = 0

    legacy_fields = ["legacyUniqueUserId", "publicId", "unique_id", "legacyId"]

    for doc in users:
        data = doc.to_dict() or {}
        current = data.get("uniqueUserId")

        # Normalize if numeric ID slipped in
        if current and str(current).isdigit():
            new_id = f"RA{int(current):03d}"
            doc.reference.update({"uniqueUserId": new_id})
            converted += 1
            updated += 1
            continue

        # Skip if valid
        if isinstance(current, str) and current.startswith("RA"):
            skipped += 1
            continue

        # Try to recover from legacy fields
        legacy_value = None
        for key in legacy_fields:
            val = data.get(key)
            if isinstance(val, str) and val.startswith("RA") and val[2:].isdigit():
                legacy_value = val
                break

        if legacy_value:
            doc.reference.update({"uniqueUserId": legacy_value})
            recovered += 1
            updated += 1
            continue

        # Generate new ID
        new_id = generate_unique_id(db)
        doc.reference.update({"uniqueUserId": new_id})
        updated += 1

    return {
        "success": True,
        "updated": updated,
        "recovered": recovered,
        "converted": converted,
        "skipped": skipped,
        "message": "Unique ID backfill completed"
    }
