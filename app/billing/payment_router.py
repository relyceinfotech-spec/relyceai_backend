from fastapi import APIRouter, HTTPException, Depends, Body, Request
from app.config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET
from app.auth import get_current_user, get_firestore_db
from firebase_admin import firestore
import razorpay
from app.billing.webhook_handlers import handle_payment_captured, handle_subscription_activated, handle_subscription_charged, handle_subscription_cancelled

router = APIRouter()

# Initialize Razorpay client
client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

DEFAULT_PRICING = {
    "starter": {"monthly": 199, "yearly": 1999},
    "plus": {"monthly": 999, "yearly": 9999},
    "pro": {"monthly": 1999, "yearly": 19999},
    "business": {"monthly": 2499, "yearly": 24999},
}

VALID_BILLING_CYCLES = {"monthly", "yearly"}


def _mask_identifier(value: str) -> str:
    text = str(value or "")
    if len(text) <= 6:
        return "***"
    return f"{text[:3]}***{text[-3:]}"


def _normalize_plan_id(plan_id: str) -> str:
    return str(plan_id).strip().lower() if plan_id else ""

def _load_pricing(db):
    try:
        doc = db.collection("config").document("pricing").get()
        if doc.exists:
            plans = doc.to_dict().get("plans")
            if isinstance(plans, dict) and plans:
                return plans
    except Exception as e:
        print(f"[Payment] Pricing load failed: {e}")
    return DEFAULT_PRICING

def _get_plan_amount(pricing: dict, plan_id: str, billing_cycle: str):
    plan = pricing.get(plan_id)
    if not isinstance(plan, dict):
        return None
    if billing_cycle == "yearly":
        return plan.get("yearly") or plan.get("yearlyPrice")
    return plan.get("monthly") or plan.get("monthlyPrice")

def _infer_plan_from_amount(pricing: dict, amount: float):
    for plan_id, plan in pricing.items():
        if not isinstance(plan, dict):
            continue
        if plan.get("monthly") == amount or plan.get("yearly") == amount:
            return plan_id
        if plan.get("monthlyPrice") == amount or plan.get("yearlyPrice") == amount:
            return plan_id
    return None

def _get_order_meta(db, order_id: str):
    try:
        doc = db.collection("paymentOrders").document(order_id).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"[Payment] Order meta lookup failed: {e}")
        return None

@router.post("/create-order")
async def create_order(
    plan_id: str = Body(...),
    billing_cycle: str = Body("monthly"),
    currency: str = Body("INR"),
    receipt: str = Body(None),
    user_info: dict = Depends(get_current_user)
):
    """
    Create a new Razorpay order.
    """
    try:
        user_id = user_info["uid"]
        plan_id = _normalize_plan_id(plan_id)
        billing_cycle = (billing_cycle or "monthly").lower()
        if billing_cycle not in VALID_BILLING_CYCLES:
            raise HTTPException(status_code=400, detail="Invalid billing cycle")

        db = get_firestore_db()
        pricing = _load_pricing(db)
        amount_rupees = _get_plan_amount(pricing, plan_id, billing_cycle)
        if not amount_rupees:
            raise HTTPException(status_code=400, detail="Invalid plan")

        amount_subunits = int(amount_rupees * 100)
        safe_notes = {
            "user_id": user_id,
            "plan_id": plan_id,
            "billing_cycle": billing_cycle
        }

        data = {
            "amount": amount_subunits,
            "currency": currency,
            "receipt": receipt,
            "notes": safe_notes,
            "payment_capture": 1 # Auto capture
        }
        order = client.order.create(data=data)
        try:
            db.collection("paymentOrders").document(order.get("id")).set({
                "userId": user_id,
                "planId": plan_id,
                "billingCycle": billing_cycle,
                "amount": amount_rupees,
                "currency": currency,
                "createdAt": firestore.SERVER_TIMESTAMP
            }, merge=True)
        except Exception as meta_err:
            print(f"[Payment] Warning: Failed to store order meta: {meta_err}")
        return {"success": True, "order": order, "key_id": RAZORPAY_KEY_ID, "amount": amount_rupees}
    except Exception as e:
        print(f"[Razorpay] Order creation failed type={e.__class__.__name__}")
        raise HTTPException(status_code=500, detail="Order creation failed")

@router.post("/verify")
async def verify_payment(
    razorpay_order_id: str = Body(...),
    razorpay_payment_id: str = Body(...),
    razorpay_signature: str = Body(...),
    plan_id: str = Body(None),
    billing_cycle: str = Body(None),
    user_info: dict = Depends(get_current_user)
):
    """
    Verify Razorpay payment signature and update user membership securely.
    """
    try:
        user_id = user_info["uid"]
        # 1. Verify signature
        params_dict = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        
        client.utility.verify_payment_signature(params_dict)
        
        # 1.5 Fetch payment + order details for validation
        try:
            payment_details = client.payment.fetch(razorpay_payment_id)
            amount_paid = payment_details.get('amount', 0) / 100
        except Exception as fetch_error:
            print(f"[Payment] CRITICAL: Could not fetch payment details: {fetch_error}")
            raise HTTPException(status_code=500, detail="Payment verification unavailable. Please contact support.")

        try:
            order_details = client.order.fetch(razorpay_order_id)
        except Exception as fetch_error:
            print(f"[Payment] Warning: Could not fetch order details: {fetch_error}")
            order_details = {}

        payment_notes = (payment_details or {}).get('notes', {}) or {}
        order_notes = (order_details or {}).get('notes', {}) or {}

        # 1.6 Validate payment notes user_id matches token
        notes_user_id = order_notes.get('user_id') or payment_notes.get('user_id') or payment_notes.get('userId')
        if not notes_user_id:
            print(f"[Payment] Missing user_id in payment notes for {razorpay_payment_id}")
            raise HTTPException(status_code=400, detail="Payment metadata missing user_id")
        if notes_user_id != user_id:
            print(f"[Payment] User mismatch: token={_mask_identifier(user_id)} notes={_mask_identifier(notes_user_id)}")
            raise HTTPException(status_code=403, detail="Payment user mismatch")

        # 1.7 Resolve plan + billing from server-side metadata
        db = get_firestore_db()
        order_meta = _get_order_meta(db, razorpay_order_id) or {}

        resolved_plan_id = _normalize_plan_id(
            order_notes.get('plan_id')
            or order_notes.get('plan')
            or payment_notes.get('plan_id')
            or payment_notes.get('plan')
            or order_meta.get('planId')
        )
        resolved_billing = (
            order_notes.get('billing_cycle')
            or order_notes.get('billingCycle')
            or payment_notes.get('billing_cycle')
            or payment_notes.get('billingCycle')
            or order_meta.get('billingCycle')
            or "monthly"
        ).lower()

        if resolved_billing not in VALID_BILLING_CYCLES:
            raise HTTPException(status_code=400, detail="Invalid billing cycle")
        if not resolved_plan_id:
            raise HTTPException(status_code=400, detail="Missing plan metadata")

        pricing = _load_pricing(db)
        expected_amount = _get_plan_amount(pricing, resolved_plan_id, resolved_billing)
        if not expected_amount:
            raise HTTPException(status_code=400, detail="Invalid plan pricing")

        order_amount = (order_details or {}).get('amount', 0) / 100 if order_details else 0
        if not order_amount and order_meta.get("amount"):
            order_amount = order_meta.get("amount")

        if not order_amount and not amount_paid:
            raise HTTPException(status_code=400, detail="Unable to verify payment amount")

        if order_amount and int(order_amount) != int(expected_amount):
            raise HTTPException(status_code=400, detail="Order amount mismatch")
        if amount_paid and int(amount_paid) != int(expected_amount):
            raise HTTPException(status_code=400, detail="Payment amount mismatch")

        # 2. Payment Verified - Calculate Expiry
        from datetime import datetime, timedelta, timezone

        # Calculate expiry date in UTC
        now = datetime.now(timezone.utc)
        if resolved_billing == 'yearly':
            expiry_date = now + timedelta(days=365)
        else:
            expiry_date = now + timedelta(days=30)
            
        # 3. Update Firestore (Secure Backend Write)
        user_ref = db.collection('users').document(user_id)
        
        amount_to_store = amount_paid if amount_paid else expected_amount
        update_data = {
            "membership.plan": resolved_plan_id,
            "membership.planName": resolved_plan_id.capitalize(), # Or fetch from config
            "membership.status": "active",
            "membership.startDate": now.isoformat(),
            "membership.expiryDate": expiry_date.isoformat(),
            "membership.billingCycle": resolved_billing,
            "membership.paymentStatus": "paid",
            "membership.updatedAt": firestore.SERVER_TIMESTAMP,
            f"membership.paymentHistory.{razorpay_payment_id}": {
                "transactionId": razorpay_payment_id,
                "orderId": razorpay_order_id,
                "amount": amount_to_store,
                "plan": resolved_plan_id,
                "date": now.isoformat(),
                "verified": True
            }
        }
        
        # Merge update using update() to correctly handle dot notation for nested fields
        try:
            user_ref.update(update_data)
        except Exception as e:
            # Fallback for if doc doesn't exist (unlikely for existing user)
            print(f"[Payment] Update failed, trying set with nested dict: {e}")
            # Construct nested dict for set
            nested_data = {
                "membership": {
                    "plan": resolved_plan_id,
                    "planName": resolved_plan_id.capitalize(),
                    "status": "active",
                    "startDate": now.isoformat(),
                    "expiryDate": expiry_date.isoformat(),
                    "billingCycle": resolved_billing,
                    "paymentStatus": "paid",
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                    "paymentHistory": {
                        razorpay_payment_id: {
                            "transactionId": razorpay_payment_id,
                            "orderId": razorpay_order_id,
                            "amount": amount_to_store,
                            "plan": resolved_plan_id,
                            "date": now.isoformat(),
                            "verified": True
                        }
                    }
                }
            }
            user_ref.set(nested_data, merge=True)
        
        # Log payment separately
        payment_ref = db.collection('payments').document(razorpay_payment_id)
        payment_ref.set({
            "userId": user_id,
            "orderId": razorpay_order_id,
            "paymentId": razorpay_payment_id,
            "planId": resolved_plan_id,
            "plan": resolved_plan_id,
            "billingCycle": resolved_billing,
            "amount": amount_to_store,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "verified": True
        })
        
        print(f"[Payment] Verified and updated for user={_mask_identifier(user_id)} plan={resolved_plan_id}")
        return {"success": True, "message": "Payment verified and membership updated"}
        
    except razorpay.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Signature verification failed")
    except Exception as e:
        print(f"[Razorpay] Verification error: {e}")
        raise HTTPException(status_code=500, detail="Payment verification failed")

# ==========================================
# Admin Payment Reconciliation Endpoints
# ==========================================

async def require_admin(user_info: dict = Depends(get_current_user)):
    db = get_firestore_db()
    doc = db.collection("users").document(user_info["uid"]).get()
    if not doc.exists:
        raise HTTPException(status_code=403, detail="Admin profile not found")
    role = doc.to_dict().get("role", "user")
    if role not in ["admin", "superadmin"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return user_info


@router.get("/admin/check-payment/{payment_id}")
async def check_payment_status(payment_id: str, user_info: dict = Depends(require_admin)):
    """
    Fetch payment details directly from Razorpay to verify status.
    Intended for Admin Dashboard reconciliation.
    """
    try:
        payment = client.payment.fetch(payment_id)
        
        # Extract relevant info for display
        notes = payment.get('notes', {})
        email = payment.get('email')
        
        suggested_user_id = None
        
        # Try to find user by email if we have one
        if email:
            try:
                db = get_firestore_db()
                # Query users collection by email
                users_ref = db.collection('users')
                query = users_ref.where('email', '==', email).limit(1)
                results = query.stream()
                
                for doc in results:
                    suggested_user_id = doc.id
                    break
            except Exception as db_e:
                print(f"[Admin Payment Check] DB Lookup Error: {db_e}")

        # Try to infer plan from amount if missing in notes
        inferred_plan = None
        amount_in_rupees = payment.get('amount', 0) / 100
        try:
            pricing = _load_pricing(get_firestore_db())
            inferred_plan = _infer_plan_from_amount(pricing, amount_in_rupees)
        except Exception as infer_err:
            print(f"[Admin Payment Check] Plan inference error: {infer_err}")
        
        return {
            "success": True,
            "payment": {
                "id": payment.get('id'),
                "status": payment.get('status'),
                "amount": amount_in_rupees,
                "currency": payment.get('currency'),
                "email": email,
                "contact": payment.get('contact'),
                "created_at": payment.get('created_at'),
                "method": payment.get('method'),
                "notes": {
                    "user_id": notes.get('user_id') or notes.get('userId') or 'N/A',
                    "plan_id": notes.get('plan_id') or notes.get('plan') or 'N/A',
                    "billing_cycle": notes.get('billing_cycle') or notes.get('billingCycle') or 'monthly'
                },
                "suggested_user_id": suggested_user_id,
                "inferred_plan": inferred_plan
            }
        }
    except Exception as e:
        print(f"[Admin Payment Check] Error: {e}")
        raise HTTPException(status_code=404, detail="Payment not found or error fetching")

@router.post("/admin/sync-payment/{payment_id}")
async def sync_payment_manual(payment_id: str, user_id: str = Body(..., embed=True), plan_id: str = Body(..., embed=True), user_info: dict = Depends(require_admin)):
    """
    Manually sync a payment to activate a user's plan.
    This behaves like the webhook but is triggered manually by admin.
    """
    try:
        # Re-fetch payment to be absolutely sure of status
        payment = client.payment.fetch(payment_id)
        
        if payment.get('status') != 'captured':
             raise HTTPException(status_code=400, detail=f"Payment is in '{payment.get('status')}' state, not 'captured'. Cannot sync.")

        plan_id = _normalize_plan_id(plan_id)
        if not plan_id:
            raise HTTPException(status_code=400, detail="Invalid plan")

        # Prepare payload wrapper to reuse existing logic
        # We construct a synthetic payload that matches what handle_payment_captured expects
        
        # NOTE: We allow admin to override user_id/plan_id in case notes were missing/wrong
        # So we inject the admin-provided values into the entity notes for the handler
        
        # Fetch existing notes to preserve other data
        existing_notes = payment.get('notes', {})
        existing_notes['user_id'] = user_id
        existing_notes['plan_id'] = plan_id
        
        # Ensure billing cycle is present
        if 'billing_cycle' not in existing_notes and 'billingCycle' not in existing_notes:
             existing_notes['billing_cycle'] = 'monthly' # Default

        payload = {
            "payload": {
                "payment": {
                    "entity": {
                        "id": payment_id,
                        "order_id": payment.get('order_id'),
                        "amount": payment.get('amount'),
                        "currency": payment.get('currency'),
                        "status": payment.get('status'),
                        "method": payment.get('method'),
                        "email": payment.get('email'),
                        "contact": payment.get('contact'),
                        "notes": existing_notes,
                    }
                }
            }
        }
        
        db = get_firestore_db()
        
        # reuse the logic
        success = await handle_payment_captured(db, payload)
        if not success:
            raise HTTPException(status_code=400, detail="Payment sync failed validation")
        
        return {"success": True, "message": f"Payment {payment_id} synced and plan {plan_id} activated for user {user_id}"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Admin Payment Sync] Error type={e.__class__.__name__}")
        raise HTTPException(status_code=500, detail="Payment sync failed")

async def handle_razorpay_webhook(request: Request):
    """
    Handle Razorpay Webhooks securely.
    Verifies signature and processes subscription events.
    """
    try:
        # 1. Get raw body and signature
        payload_body = await request.body()
        signature = request.headers.get("X-Razorpay-Signature")

        if not signature:
            print("[Razorpay Webhook] Missing signature")
            raise HTTPException(status_code=400, detail="Missing signature")

        # 2. Verify Signature
        try:
            client.utility.verify_webhook_signature(
                payload_body.decode('utf-8'),
                signature,
                RAZORPAY_WEBHOOK_SECRET
            )
        except Exception as e:
            print(f"[Razorpay Webhook] Signature verification failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")

        # 3. Parse Event
        payload = await request.json()
        event_type = payload.get("event")
        # Get processed ID from different possible locations depending on event type
        # Ideally use the request header X-Razorpay-Event-Id if available for uniqueness
        unique_event_id = request.headers.get("X-Razorpay-Event-Id")
        
        if not unique_event_id:
             unique_event_id = payload.get("payload", {}).get("payment", {}).get("entity", {}).get("id") or payload.get("payload", {}).get("subscription", {}).get("entity", {}).get("id") or payload.get("event") + "_" + str(payload.get("created_at"))

        print(f"[Razorpay Webhook] Received event={event_type} id={_mask_identifier(unique_event_id)}")

        # 4. Idempotency Check
        db = get_firestore_db()
        
        webhook_ref = db.collection('processed_webhooks').document(unique_event_id)
        if webhook_ref.get().exists:
            print(f"[Razorpay Webhook] Event id={_mask_identifier(unique_event_id)} already processed. Skipping.")
            return {"status": "ok", "message": "Already processed"}

        # 5. Handle Events
        if event_type == "subscription.activated":
            await handle_subscription_activated(db, payload)
        
        elif event_type == "subscription.charged":
            await handle_subscription_charged(db, payload)
            
        elif event_type == "subscription.cancelled":
            await handle_subscription_cancelled(db, payload)
            
        elif event_type == "payment.captured":
            await handle_payment_captured(db, payload)

        elif event_type == "payment.failed":
            failure_id = payload.get("payload", {}).get("payment", {}).get("entity", {}).get("id") or "unknown"
            print(f"[Razorpay Webhook] Payment failed (payment_id={_mask_identifier(failure_id)})")
        
        # 6. Mark as processed in DB
        webhook_ref.set({
            "event": event_type,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "entity_id": payload.get("payload", {}).get("payment", {}).get("entity", {}).get("id")
                or payload.get("payload", {}).get("subscription", {}).get("entity", {}).get("id"),
            "created_at": payload.get("created_at")
        })

        return {"status": "ok"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Razorpay Webhook] Error processing event: {e}")
        # Return 200 to prevent Razorpay from retrying indefinitely on logic errors
        return {"status": "error", "message": "Webhook processing error"}


@router.post("/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay Webhook endpoint.
    """
    if not RAZORPAY_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    return await handle_razorpay_webhook(request)




