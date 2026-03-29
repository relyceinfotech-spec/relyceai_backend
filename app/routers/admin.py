"""
Admin Router
Handles administrative actions like role management and audit logging.
"""
from fastapi import APIRouter, HTTPException, Depends, Body, Query
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import time
from app.auth import get_firestore_db, get_current_user, is_admin_user, is_superadmin_user, normalize_role
from datetime import datetime, timedelta, timezone
from firebase_admin import auth as firebase_auth
from app.platform import get_ai_platform
from app.api.routes import admin as high_stakes_admin
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


class UpdateAgentDebugConfigRequest(BaseModel):
    auto_tuning_enabled: Optional[bool] = None
    auto_tuning_min_runs: Optional[int] = None
    auto_tuning_success_threshold: Optional[float] = None
    auto_tuning_cooldown_runs: Optional[int] = None
    auto_tuning_budget_step_ms: Optional[int] = None
    auto_tuning_retry_step: Optional[int] = None
    auto_tuning_max_budget_delta_ms: Optional[int] = None
    auto_tuning_max_retry_delta: Optional[int] = None
    failure_confidence_threshold: Optional[float] = None
    adaptive_enabled: Optional[bool] = None
    adaptive_ema_alpha: Optional[float] = None
    adaptive_min_cluster_runs: Optional[int] = None
    adaptive_low_conf_threshold: Optional[float] = None
    adaptive_retry_threshold: Optional[float] = None
    adaptive_cooldown_runs: Optional[int] = None
    adaptive_max_bias_delta: Optional[float] = None
    adaptive_state_cache_ttl_sec: Optional[int] = None
    adaptive_state_write_every_runs: Optional[int] = None
    adaptive_apply_ratio: Optional[float] = None
    auto_remediation_enabled: Optional[bool] = None
    auto_remediation_min_runs: Optional[int] = None
    auto_remediation_cooldown_runs: Optional[int] = None
    auto_remediation_step: Optional[float] = None
    auto_remediation_min_apply_ratio: Optional[float] = None
    auto_remediation_low_conf_trigger: Optional[float] = None
    auto_remediation_retry_step_cap: Optional[int] = None
    auto_remediation_window_runs: Optional[int] = None
    slo_p95_latency_ms_max: Optional[float] = None
    slo_low_conf_rate_max: Optional[float] = None
    slo_avg_retries_max: Optional[float] = None
    slo_parallel_exception_rate_max: Optional[float] = None


class AgentModeCheckRequest(BaseModel):
    input: str
    requested_mode: str = "smart"


class AdaptiveSnapshotRequest(BaseModel):
    label: Optional[str] = None


def require_admin(user_info: dict = Depends(get_current_user)):
    if not is_admin_user(user_info):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user_info

def require_superadmin(user_info: dict = Depends(get_current_user)):
    if not is_superadmin_user(user_info):
        raise HTTPException(status_code=403, detail="Insufficient permissions: SuperAdmin only")
    return user_info


def _get_research_service_or_503():
    platform = get_ai_platform()
    service = platform.get_service("research")
    if service is None:
        raise HTTPException(status_code=503, detail="Research service unavailable")
    if not hasattr(service, "get_debug_snapshot"):
        raise HTTPException(status_code=503, detail="Research debug instrumentation unavailable")
    return service


def _build_ops_insights(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    totals = dict(snapshot.get("totals") or {})
    per_mode_stats = list(snapshot.get("per_mode_stats") or [])
    per_role_stats = list(snapshot.get("per_role_stats") or [])
    failure_analyzer = dict(snapshot.get("failure_analyzer") or {})
    confidence_distribution = dict(snapshot.get("confidence_distribution") or {})
    override_recent = list((snapshot.get("override_insights") or {}).get("recent") or [])
    role_flow_recent = list((snapshot.get("role_flow") or {}).get("recent") or [])
    slo = dict(snapshot.get("slo") or {})
    adaptive_learning = dict(snapshot.get("adaptive_learning") or {})
    canary_compare = dict(adaptive_learning.get("canary_compare") or {})
    tool_reliability = dict(snapshot.get("tool_reliability") or {})

    total_runs = int(totals.get("runs", 0) or 0)
    success_rate = float(totals.get("success_rate", 0.0) or 0.0)
    low_conf_rate = float(confidence_distribution.get("low_pct", 0.0) or 0.0)
    p95_latency_ms = float((slo.get("current") or {}).get("p95_latency_ms", 0.0) or 0.0)
    parallel_exception_rate = 0.0
    researcher_stats = next((row for row in per_role_stats if str(row.get("role") or "") == "researcher"), None)
    if researcher_stats:
        parallel_exception_rate = float(researcher_stats.get("parallel_exception_rate", 0.0) or 0.0)
    parallel_exception_rate_max = float((slo.get("targets") or {}).get("parallel_exception_rate_max", 0.30) or 0.30)
    avg_latency_ms = 0.0
    if per_mode_stats:
        avg_latency_ms = sum(float(m.get("avg_latency_ms", 0.0) or 0.0) for m in per_mode_stats) / max(1, len(per_mode_stats))

    failure_items = list(failure_analyzer.get("items") or [])
    low_conf_outputs = [
        {
            "timestamp": str(row.get("timestamp") or ""),
            "query": str(row.get("query") or ""),
            "confidence_level": str(row.get("confidence_level") or "LOW"),
            "retries": int(row.get("retries", 0) or 0),
            "mode": str(row.get("mode") or "smart"),
        }
        for row in failure_items[-20:]
    ]

    recent_chats = []
    for row in failure_items[-30:]:
        recent_chats.append(
            {
                "timestamp": str(row.get("timestamp") or ""),
                "query": str(row.get("query") or ""),
                "mode": str(row.get("mode") or "smart"),
                "success": bool(row.get("success", False)),
                "confidence_level": str(row.get("confidence_level") or "LOW"),
            }
        )

    alerts: List[Dict[str, Any]] = []
    if success_rate < 0.85:
        alerts.append({"severity": "high", "code": "high_failure_rate", "message": f"Failure rate is {round((1.0 - success_rate) * 100, 1)}%."})
    if p95_latency_ms > 12000:
        alerts.append({"severity": "medium", "code": "slow_responses", "message": f"P95 latency is {int(p95_latency_ms)} ms."})
    if low_conf_rate > 0.35:
        alerts.append({"severity": "medium", "code": "low_confidence_spike", "message": f"Low-confidence outputs are {round(low_conf_rate * 100, 1)}%."})
    if parallel_exception_rate > parallel_exception_rate_max:
        alerts.append(
            {
                "severity": "medium",
                "code": "parallel_exception_spike",
                "message": (
                    f"Parallel exception rate is {round(parallel_exception_rate * 100, 1)}% "
                    f"(threshold {round(parallel_exception_rate_max * 100, 1)}%)."
                ),
            }
        )
    tool_failures = sum(1 for item in failure_items if int(item.get("retries", 0) or 0) > 1)
    if tool_failures > 0:
        alerts.append({"severity": "low", "code": "tool_failures", "message": f"{tool_failures} runs required repeated retries."})

    logs: List[Dict[str, Any]] = []
    for row in override_recent[-15:]:
        logs.append(
            {
                "timestamp": str(row.get("at") or ""),
                "level": "info",
                "source": "mode_override",
                "message": f"{row.get('auto_selected')} -> {row.get('overridden_to')} ({row.get('reason')})",
            }
        )
    for row in role_flow_recent[-20:]:
        logs.append(
            {
                "timestamp": str(row.get("at") or ""),
                "level": "info",
                "source": "role_flow",
                "message": f"{row.get('node_id', 'node')} [{row.get('role', 'executor')}] {row.get('status', 'completed')}",
            }
        )
    for row in low_conf_outputs[-10:]:
        logs.append(
            {
                "timestamp": str(row.get("timestamp") or ""),
                "level": "warn",
                "source": "low_confidence",
                "message": f"{row.get('mode')} confidence={row.get('confidence_level')} retries={row.get('retries')}",
            }
        )
    logs.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)

    return {
        "usage": {
            "total_chats": total_runs,
            "active_users": int(total_runs > 0),
            "api_usage": total_runs,
            "response_latency_ms": round(avg_latency_ms, 2),
            "error_rate": round(max(0.0, 1.0 - success_rate), 4),
        },
        "chat_monitoring": {
            "recent_chats": recent_chats[-20:],
            "flagged_responses": int(len(low_conf_outputs)),
            "low_confidence_outputs": low_conf_outputs,
        },
        "alerts": alerts,
        "logs": logs[:50],
        "canary_compare": {
            "baseline": dict(canary_compare.get("baseline") or {}),
            "adaptive": dict(canary_compare.get("adaptive") or {}),
            "delta": dict(canary_compare.get("delta") or {}),
        },
        "tool_reliability": {
            "count": int(tool_reliability.get("count", 0) or 0),
            "top_tools": list(tool_reliability.get("top_tools") or [])[:10],
            "weak_tools": list(tool_reliability.get("weak_tools") or [])[:10],
        },
    }

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


@router.get("/agent-debug")
async def get_agent_debug(
    range: str = Query(default="24h"),
    limit: int = Query(default=25, ge=5, le=100),
    user_info: dict = Depends(require_superadmin),
):
    """
    Admin debug panel data for hybrid/agent behavior:
    per-mode stats, override insights, confidence distribution, failure analyzer.
    """
    _ = user_info
    service = _get_research_service_or_503()
    snapshot = service.get_debug_snapshot(range_window=range, limit=limit)
    return {"success": True, "data": snapshot}


@router.get("/ops-insights")
async def get_ops_insights(
    range: str = Query(default="24h"),
    limit: int = Query(default=25, ge=5, le=100),
    user_info: dict = Depends(require_admin),
):
    """
    Admin-safe operational insights:
    usage dashboard, chat monitoring, alerts, and basic logs.
    """
    _ = user_info
    service = _get_research_service_or_503()
    snapshot = service.get_debug_snapshot(range_window=range, limit=limit)
    return {"success": True, "data": _build_ops_insights(snapshot)}


@router.post("/agent-debug/mode-check")
async def get_agent_mode_check(
    request: AgentModeCheckRequest,
    user_info: dict = Depends(require_superadmin),
):
    """
    One-shot mode/lane/override inspection for a specific input.
    Useful for debugging mode mismatches and slow-path routing.
    """
    _ = user_info
    if not str(request.input or "").strip():
        raise HTTPException(status_code=400, detail="input is required")

    service = _get_research_service_or_503()
    if not hasattr(service, "get_mode_check"):
        raise HTTPException(status_code=503, detail="Mode check instrumentation unavailable")

    result = service.get_mode_check(
        input_text=str(request.input or ""),
        requested_mode=str(request.requested_mode or "smart"),
    )
    return {"success": True, "data": result}


@router.get("/high-stakes-metrics")
async def get_high_stakes_metrics(
    range: str = Query(default="24h"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=5, le=100),
    user_info: dict = Depends(require_admin),
):
    return await high_stakes_admin.high_stakes_metrics(
        range=range,
        page=page,
        page_size=page_size,
        user_info=user_info,
    )


@router.get("/high-stakes-thresholds")
async def get_high_stakes_thresholds(
    user_info: dict = Depends(require_admin),
):
    return await high_stakes_admin.get_thresholds(user_info=user_info)


@router.put("/high-stakes-thresholds")
async def update_high_stakes_thresholds(
    payload: dict = Body(default={}),
    user_info: dict = Depends(require_superadmin),
):
    return await high_stakes_admin.update_thresholds(payload=payload, user_info=user_info)


@router.put("/agent-debug-config")
async def update_agent_debug_config(
    request: UpdateAgentDebugConfigRequest,
    user_info: dict = Depends(require_superadmin),
):
    """
    SuperAdmin-only tuning knobs for auto self-improvement hooks + failure analyzer.
    """
    _ = user_info
    service = _get_research_service_or_503()
    cfg = service.update_debug_config(request.model_dump(exclude_none=True))
    return {"success": True, "config": cfg}


@router.post("/agent-debug/adaptive/snapshot")
async def create_adaptive_snapshot(
    request: AdaptiveSnapshotRequest = Body(default=AdaptiveSnapshotRequest()),
    user_info: dict = Depends(require_superadmin),
):
    _ = user_info
    service = _get_research_service_or_503()
    if not hasattr(service, "create_adaptive_snapshot"):
        raise HTTPException(status_code=503, detail="Adaptive snapshot tooling unavailable")
    result = service.create_adaptive_snapshot(label=str(request.label or "manual"))
    return {"success": True, "data": result}


@router.post("/agent-debug/adaptive/rollback")
async def rollback_adaptive_snapshot(
    user_info: dict = Depends(require_superadmin),
):
    _ = user_info
    service = _get_research_service_or_503()
    if not hasattr(service, "rollback_latest_adaptive_snapshot"):
        raise HTTPException(status_code=503, detail="Adaptive rollback tooling unavailable")
    result = service.rollback_latest_adaptive_snapshot()
    return {"success": True, "data": result}


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

    target_data = target_doc.to_dict() or {}
    membership = target_data.get("membership", {})
    existing_expiry_str = membership.get("expiryDate")
    
    now = datetime.now(timezone.utc)
    base_date = now

    if existing_expiry_str:
        try:
            # If standard isoformat
            existing_expiry = datetime.fromisoformat(existing_expiry_str.replace("Z", "+00:00"))
            if existing_expiry > now:
                base_date = existing_expiry
        except Exception as e:
            print(f"[Admin] Could not parse existing expiry date: {e}")
            pass

    # Calculate expiry
    if request.billing_cycle == "yearly":
        expiry_date = base_date + timedelta(days=365)
    else:
        expiry_date = base_date + timedelta(days=30)

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

@router.get("/users/{target_uid}")
async def get_user(target_uid: str, user_info: dict = Depends(require_admin)):
    """
    Admin-only user fetch (server-side Firestore read).
    """
    db = get_firestore_db()
    target_ref = db.collection("users").document(target_uid)
    target_doc = target_ref.get()
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Target user not found")
        
    user_data = target_doc.to_dict()
    
    # Process membership expiry before returning the profile
    try:
        check_membership_expiry(target_ref, user_data, target_uid)
        # Fetch fresh data if it was downgraded
        target_doc = target_ref.get()
        user_data = target_doc.to_dict()
    except Exception as e:
        print(f"[Admin] Failed to check expiry during /users/{target_uid}: {e}")
        
    return {"success": True, "user": user_data}


@router.delete("/users/{target_uid}")
async def delete_user(target_uid: str, hard: bool = False, user_info: dict = Depends(require_admin)):
    """
    Admin-only user deletion through backend.
    Defaults to soft delete unless hard=true is provided.
    """
    db = get_firestore_db()
    target_ref = db.collection("users").document(target_uid)
    target_doc = target_ref.get()
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Target user not found")

    now = datetime.now(timezone.utc)

    # Disable user in Firebase Auth and revoke tokens
    try:
        firebase_auth.update_user(target_uid, disabled=True)
        firebase_auth.revoke_refresh_tokens(target_uid)
    except Exception as e:
        print(f"[Admin] Failed to disable user {target_uid}: {e}")

    if not hard:
        # Soft delete: keep data, mark deleted
        target_ref.update({
            "isDeleted": True,
            "status": "deleted",
            "deletedAt": now,
            "deletedBy": user_info["uid"],
        })

        db.collection("auditLogs").add({
            "action": "USER_SOFT_DELETED",
            "by": user_info["uid"],
            "target": target_uid,
            "timestamp": now,
            "time": now.timestamp()
        })

        return {"success": True, "message": "User soft-deleted"}

    # Hard delete backup
    try:
        db.collection("deletedUsers").document(target_uid).set({
            "data": target_doc.to_dict(),
            "deletedAt": now,
            "deletedBy": user_info["uid"],
            "mode": "hard"
        }, merge=True)
    except Exception as e:
        print(f"[Admin] Failed to backup user {target_uid}: {e}")

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

    db.collection("auditLogs").add({
        "action": "USER_HARD_DELETED",
        "by": user_info["uid"],
        "target": target_uid,
        "timestamp": now,
        "time": now.timestamp()
    })

    return {"success": True, "message": "User hard-deleted"}

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
