"""
Admin Router
Handles administrative actions like role management and audit logging.
"""
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
import time
from app.auth import verify_token, get_firestore_db
from app.auth import verify_token, get_firestore_db
from datetime import datetime
# Import reusable expiry logic
from app.routers.users import check_membership_expiry

router = APIRouter()

class ChangeRoleRequest(BaseModel):
    target_uid: str
    new_role: str

async def get_current_user_uid(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    token = authorization.split(" ")[1]
    is_valid, user_info = verify_token(token)
    
    if not is_valid or not user_info:
        raise HTTPException(status_code=401, detail="Invalid token")
        
    return user_info["uid"]

@router.post("/change-role")
async def change_role(request: ChangeRoleRequest, requester_uid: str = Depends(get_current_user_uid)):
    """
    Change a user's role.
    Only SuperAdmin can perform this action.
    """
    db = get_firestore_db()
    
    # 1. Verify Requester is SuperAdmin
    requester_ref = db.collection("users").document(requester_uid)
    requester_doc = requester_ref.get()
    
    if not requester_doc.exists:
        raise HTTPException(status_code=403, detail="Requester profile not found")
        
    requester_data = requester_doc.to_dict()
    current_role = requester_data.get("role", "user")
    
    if current_role != "superadmin":
        raise HTTPException(status_code=403, detail="Insufficient permissions: Only SuperAdmin can change roles")

    # 2. Update Target User Role
    target_ref = db.collection("users").document(request.target_uid)
    target_doc = target_ref.get()
    
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Target user not found")
        
    try:
        # Update the role
        target_ref.update({"role": request.new_role})
        
        # 3. Create Audit Log
        audit_entry = {
            "action": "ROLE_CHANGED",
            "by": requester_uid,
            "target": request.target_uid,
            "previous_role": target_doc.to_dict().get("role", "unknown"),
            "new_role": request.new_role,
            "timestamp": datetime.now(), # Python datetime for Firestore
            "time": time.time() # Unix timestamp for easier sorting/filtering if needed
        }
        
        db.collection("auditLogs").add(audit_entry)
        
        return {"success": True, "message": f"Role updated to {request.new_role}"}
        
    except Exception as e:
        print(f"[Admin] Role change error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update role")

@router.post("/jobs/expire-memberships")
async def run_expiry_job(requester_uid: str = Depends(get_current_user_uid)):
    """
    CRON JOB: Checks all active memberships and expires them if past valid date.
    Triggered by SuperAdmin (or authorized system service).
    """
    print(f"[Job] Expiry job triggered by {requester_uid}")
    
    # 1. Verify SuperAdmin (Only they can trigger system jobs manually for now)
    # In future, can allow a specific "Service Account" UID.
    db = get_firestore_db()
    requester = db.collection("users").document(requester_uid).get()
    if not requester.exists or requester.to_dict().get("role") != "superadmin":
         raise HTTPException(status_code=403, detail="Only SuperAdmin can trigger jobs")

    # 2. Query Candidates: Active Users with Expiry Date < NOW
    # Note: Querying by date requires the Index we just added.
    # To be safe against index propagation delay or slight time diffs, 
    # we query 'active' and potentially filter by date in code if index fails, 
    # BUT standard is to use the index query.
    
    now_iso = datetime.now().isoformat()
    
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
