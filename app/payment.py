from fastapi import APIRouter, HTTPException, Depends, Body, Request
from app.config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET
from app.auth import get_current_user, get_firestore_db
import razorpay
import hmac
import hashlib

router = APIRouter()

# Initialize Razorpay client
client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

@router.post("/create-order")
async def create_order(
    amount: int = Body(..., description="Amount in currency subunits (e.g. 100 paise = 1 INR)"),
    currency: str = Body("INR"),
    receipt: str = Body(None),
    notes: dict = Body(None),
    user_info: dict = Depends(get_current_user)
):
    """
    Create a new Razorpay order.
    Amount should be in subunits (e.g., paise for INR).
    """
    try:
        user_id = user_info["uid"]
        safe_notes = notes.copy() if isinstance(notes, dict) else {}
        safe_notes["user_id"] = user_id

        data = {
            "amount": amount,
            "currency": currency,
            "receipt": receipt,
            "notes": safe_notes,
            "payment_capture": 1 # Auto capture
        }
        order = client.order.create(data=data)
        return {"success": True, "order": order, "key_id": RAZORPAY_KEY_ID}
    except Exception as e:
        print(f"[Razorpay] Order creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify")
async def verify_payment(
    razorpay_order_id: str = Body(...),
    razorpay_payment_id: str = Body(...),
    razorpay_signature: str = Body(...),
    plan_id: str = Body(...),
    billing_cycle: str = Body("monthly"),
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
        
        # 1.5 Fetch payment details to get actual amount (including coupons)
        try:
            payment_details = client.payment.fetch(razorpay_payment_id)
            amount_paid = payment_details.get('amount', 0) / 100 # Convert to Rupees
            print(f"[Payment] Fetched actual amount: {amount_paid}")
        except Exception as fetch_error:
            print(f"[Payment] Warning: Could not fetch payment details: {fetch_error}")
            amount_paid = 0
            payment_details = {}

        # 1.6 Validate payment notes user_id matches token
        notes = (payment_details or {}).get('notes', {}) or {}
        notes_user_id = notes.get('user_id') or notes.get('userId')
        if not notes_user_id:
            print(f"[Payment] Missing user_id in payment notes for {razorpay_payment_id}")
            raise HTTPException(status_code=400, detail="Payment metadata missing user_id")
        if notes_user_id != user_id:
            print(f"[Payment] User mismatch: token={user_id} notes={notes_user_id}")
            raise HTTPException(status_code=403, detail="Payment user mismatch")

        # 2. Payment Verified - Calculate Expiry
        from datetime import datetime, timedelta, timezone
        import firebase_admin
        from firebase_admin import firestore
        
        db = firestore.client()
        
        # Calculate expiry date in UTC
        now = datetime.now(timezone.utc)
        if billing_cycle == 'yearly':
            expiry_date = now + timedelta(days=365)
        else:
            expiry_date = now + timedelta(days=30)
            
        # 3. Update Firestore (Secure Backend Write)
        user_ref = db.collection('users').document(user_id)
        
        update_data = {
            "membership.plan": plan_id,
            "membership.planName": plan_id.capitalize(), # Or fetch from config
            "membership.status": "active",
            "membership.startDate": now.isoformat(),
            "membership.expiryDate": expiry_date.isoformat(),
            "membership.billingCycle": billing_cycle,
            "membership.paymentStatus": "paid",
            "membership.updatedAt": firestore.SERVER_TIMESTAMP,
            f"membership.paymentHistory.{razorpay_payment_id}": {
                "transactionId": razorpay_payment_id,
                "orderId": razorpay_order_id,
                "amount": amount_paid,
                "plan": plan_id,
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
                    "plan": plan_id,
                    "planName": plan_id.capitalize(),
                    "status": "active",
                    "startDate": now.isoformat(),
                    "expiryDate": expiry_date.isoformat(),
                    "billingCycle": billing_cycle,
                    "paymentStatus": "paid",
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                    "paymentHistory": {
                        razorpay_payment_id: {
                            "transactionId": razorpay_payment_id,
                            "orderId": razorpay_order_id,
                            "amount": amount_paid,
                            "plan": plan_id,
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
            "planId": plan_id,
            "billingCycle": billing_cycle,
            "amount": amount_paid,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "verified": True
        })
        
        print(f"[Payment] Verified and updated for user {user_id} - Plan: {plan_id} - Amount: {amount_paid}")
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
                from firebase_admin import firestore
                db = firestore.client()
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
        # This is a heuristic based on known pricing
        inferred_plan = None
        amount_in_rupees = payment.get('amount', 0) / 100
        
        # Simple heuristic mapping (you might want to move this to config if prices change often)
        if amount_in_rupees in [199, 1999]: inferred_plan = 'starter'
        elif amount_in_rupees in [999, 9999]: inferred_plan = 'plus'
        elif amount_in_rupees in [2499, 24999]: inferred_plan = 'business'
        
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
        raise HTTPException(status_code=404, detail=f"Payment not found or error fetching: {str(e)}")

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
        
        from firebase_admin import firestore
        db = firestore.client()
        
        # reuse the logic
        await handle_payment_captured(db, payload)
        
        return {"success": True, "message": f"Payment {payment_id} synced and plan {plan_id} activated for user {user_id}"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Admin Payment Sync] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

        print(f"[Razorpay Webhook] Received event: {event_type} (ID: {unique_event_id})")

        # 4. Idempotency Check
        from firebase_admin import firestore
        db = firestore.client()
        
        webhook_ref = db.collection('processed_webhooks').document(unique_event_id)
        if webhook_ref.get().exists:
            print(f"[Razorpay Webhook] Event {unique_event_id} already processed. Skipping.")
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
            print(f"[Razorpay Webhook] Payment failed: {payload}")
        
        # 6. Mark as processed in DB
        webhook_ref.set({
            "event": event_type,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "payload": payload
        })

        return {"status": "ok"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"[Razorpay Webhook] Error processing event: {e}")
        # Return 200 to prevent Razorpay from retrying indefinitely on logic errors
        return {"status": "error", "message": str(e)}


@router.post("/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay Webhook endpoint.
    """
    if not RAZORPAY_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    return await handle_razorpay_webhook(request)

async def handle_payment_captured(db, payload):
    """
    Handle one-time payments (like UPI scans) that don't go through subscription flow.
    """
    try:
        entity = payload['payload']['payment']['entity']
        payment_id = entity['id']
        order_id = entity.get('order_id')
        
        # Extract metadata from notes
        notes = entity.get('notes', {})
        user_id = notes.get('user_id') or notes.get('userId')
        plan_id = notes.get('plan_id')
        billing_cycle = notes.get('billing_cycle') or notes.get('billingCycle') or 'monthly'

        if not user_id or not plan_id:
            print(f"[Webhook] Missing user_id or plan_id in payment notes for {payment_id}")
            return

        print(f"[Webhook] Processing captured payment {payment_id} for user {user_id}, plan {plan_id}")

        # Note: We skip idempotency check for manual sync if calling directly, but this function checks DB.
        # Ideally, manual sync might want to FORCE override. But let's assume if it exists, it's done.
        # Actually, for manual sync to work if webhook failed partway (e.g. user update worked but log didn't?)
        # safer to allow overwrite or check specifically.
        # For now, let's keep idempotency but maybe log a warning if it exists.
        
        payment_doc_ref = db.collection('payments').document(payment_id)
        # We REMOVE the strict existing check here to allow Manual Sync to "repair" a record if needed
        # OR we just update the user logic.
        # Let's trust the Caller (Webhook or Admin) knows what they are doing.
        # But to be safe against double-counting, we should check `membership.paymentHistory`.

        # Calculate expiry
        from datetime import datetime, timedelta, timezone
        import firebase_admin
        from firebase_admin import firestore

        now = datetime.now(timezone.utc)
        if billing_cycle == 'yearly':
            expiry_date = now + timedelta(days=365)
        else:
            expiry_date = now + timedelta(days=30)
            
        # Update User Membership
        user_ref = db.collection('users').document(user_id)
        
        # Convert amount to Rupees for DB consistency if desired, or keep raw.
        # Standardize: Store raw in amount, but maybe amountPaid in Rupees?
        # Let's stick to storing meaningful amount in Rupees for the 'amount' field if that's what frontend expects.
        # Screenshot showed '9,999' which matches Plan Price. Razorpay sends 999900.
        amount_val = entity.get('amount', 0)
        amount_in_rupees = amount_val / 100 if amount_val > 0 else 0

        update_data = {
            "membership.plan": plan_id,
            "membership.planName": plan_id.capitalize(),
            "membership.status": "active",
            "membership.startDate": now.isoformat(),
            "membership.expiryDate": expiry_date.isoformat(),
            "membership.billingCycle": billing_cycle,
            "membership.paymentStatus": "paid",
            "membership.updatedAt": firestore.SERVER_TIMESTAMP,
            f"membership.paymentHistory.{payment_id}": {
                "transactionId": payment_id,
                "orderId": order_id,
                "amount": amount_in_rupees, # Store as Rupees
                "plan": plan_id,
                "date": now.isoformat(),
                "verified": True,
                "source": "webhook" 
            }
        }
        
        # AUDIT LOG: Membership Auto-Change (Payment)
        try:
            db.collection("auditLogs").add({
                "action": "MEMBERSHIP_AUTO_CHANGED",
                "reason": "payment",
                "by": "system",
                "target": user_id,
                "details": {
                    "plan": plan_id,
                    "paymentId": payment_id,
                    "amount": amount_in_rupees
                },
                "timestamp": firestore.SERVER_TIMESTAMP,
                "time": now.timestamp()
            })
        except Exception as audit_err:
            print(f"[Webhook] Audit Log Error: {audit_err}")

        try:
            user_ref.update(update_data)
        except Exception as e:
            # Fallback for set if needed
            print(f"[Webhook] Update failed, attempting set merge: {e}")
            user_ref.set({
                 "membership": {
                    "plan": plan_id,
                    "planName": plan_id.capitalize(),
                    "status": "active",
                    "startDate": now.isoformat(),
                    "expiryDate": expiry_date.isoformat(),
                    "billingCycle": billing_cycle,
                    "paymentStatus": "paid",
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                    "paymentHistory": {
                        payment_id: {
                            "transactionId": payment_id,
                            "orderId": order_id,
                            "amount": amount_in_rupees,
                            "plan": plan_id,
                            "date": now.isoformat(),
                            "verified": True,
                            "source": "webhook"
                        }
                    }
                }
            }, merge=True)

        # Log to payments collection
        # Ensure we write ALL fields needed by frontend
        payment_doc_ref.set({
            "userId": user_id,
            "orderId": order_id,
            "paymentId": payment_id,
            "planId": plan_id,
            "plan": plan_id, # Redundant but safe for frontend that might use .plan
            "billingCycle": billing_cycle,
            "amount": amount_in_rupees, # Store as Rupees
            "currency": entity.get('currency', 'INR'),
            "method": entity.get('method', 'unknown'),
            "email": entity.get('email'),
            "contact": entity.get('contact'),
            "status": "captured",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "verified": True,
            "source": "webhook"
        }, merge=True) # Merge to allow updating existing botched records

        print(f"[Webhook] Successfully activated plan {plan_id} for user {user_id}")

    except Exception as e:
        print(f"[Webhook] Error in handle_payment_captured: {e}") 


async def handle_subscription_activated(db, payload):
    try:
        entity = payload['payload']['subscription']['entity']
        sub_id = entity['id']
        
        notes = entity.get('notes', {})
        user_id = notes.get('userId') or notes.get('user_id')
        plan_id = notes.get('plan_id') or notes.get('plan') or notes.get('planId')
        
        if not user_id:
            print(f"[Webhook] No user_id found in notes for subscription {sub_id}")
            return

        print(f"[Webhook] Activating subscription {sub_id} for user {user_id}")
        
        # Update user
        user_ref = db.collection('users').document(user_id)
        
        from datetime import datetime, timedelta, timezone
        start_ts = entity.get('start_at') or entity.get('created_at')
        
        if start_ts:
            start_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        else:
            start_date = datetime.now(timezone.utc).isoformat()
            
        billing_cycle = notes.get('billingCycle', 'monthly')
        
        now = datetime.now(timezone.utc)
        if billing_cycle == 'yearly':
            expiry_date = now + timedelta(days=365)
        else:
            expiry_date = now + timedelta(days=30)
            
        update_data = {
            "membership.status": "active",
            "membership.subscriptionId": sub_id,
            "membership.startDate": start_date,
            "membership.expiryDate": expiry_date.isoformat(),
            "membership.paymentStatus": "paid",
            "membership.billingCycle": billing_cycle,
            "membership.updatedAt": firestore.SERVER_TIMESTAMP
        }
        if plan_id:
            update_data["membership.plan"] = plan_id
            update_data["membership.planName"] = str(plan_id).capitalize()

        # AUDIT LOG: Subscription Activated
        try:
             db.collection("auditLogs").add({
                "action": "MEMBERSHIP_AUTO_CHANGED",
                "reason": "subscription_activated",
                "by": "system",
                "target": user_id,
                "details": {
                    "subscriptionId": sub_id,
                    "billingCycle": billing_cycle
                },
                "timestamp": firestore.SERVER_TIMESTAMP,
                "time": datetime.now(timezone.utc).timestamp()
            })
        except Exception as audit_err:
             print(f"[Webhook] Audit Log Error: {audit_err}")
        
        user_ref.update(update_data)
        print(f"[Webhook] User {user_id} activated.")
    except Exception as e:
        print(f"[Webhook] Error in activation: {e}")

async def handle_subscription_charged(db, payload):
    try:
        entity = payload['payload']['subscription']['entity']
        sub_id = entity['id']
        
        # Try notes first
        notes = entity.get('notes', {})
        user_id = notes.get('userId') or notes.get('user_id')
        plan_id = notes.get('plan_id') or notes.get('plan') or notes.get('planId')
        
        user_ref = None
        user_doc = None
        
        if user_id:
            user_ref = db.collection('users').document(user_id)
            user_doc = user_ref.get()
        
        if not user_doc or not user_doc.exists:
            # Fallback check
            users = db.collection('users').where('membership.subscriptionId', '==', sub_id).limit(1).get()
            if users:
                user_doc = users[0]
                user_ref = user_doc.reference
                user_id = user_doc.id
            else:
                print(f"[Webhook] User not found for subscription {sub_id}")
                return

        current_data = user_doc.to_dict().get('membership', {})
        current_expiry = current_data.get('expiryDate')
        
        from datetime import datetime, timedelta, timezone
        
        if current_expiry:
            try:
                # Handle existing timestamps that might have Z or be naive
                old_end = datetime.fromisoformat(current_expiry.replace('Z', '+00:00'))
                if old_end.tzinfo is None:
                    old_end = old_end.replace(tzinfo=timezone.utc)
            except:
                old_end = datetime.now(timezone.utc)
        else:
            old_end = datetime.now(timezone.utc)
            
        # Ensure we don't extend from a long-expired date
        base_date = max(old_end, datetime.now(timezone.utc))
        
        billing_cycle = current_data.get('billingCycle', 'monthly')
        
        if billing_cycle == 'yearly':
            new_end = base_date + timedelta(days=365)
        else:
            new_end = base_date + timedelta(days=30)
            
        update_data = {
            "membership.expiryDate": new_end.isoformat(),
            "membership.lastBillingAt": firestore.SERVER_TIMESTAMP,
            "membership.status": "active"
        }
        if plan_id:
            update_data["membership.plan"] = plan_id
            update_data["membership.planName"] = str(plan_id).capitalize()
        user_ref.update(update_data)
        
        # AUDIT LOG: Subscription Charged (Renewal)
        try:
             db.collection("auditLogs").add({
                "action": "MEMBERSHIP_AUTO_CHANGED",
                "reason": "subscription_charged",
                "by": "system",
                "target": user_id,
                "details": {
                    "subscriptionId": sub_id,
                    "newExpiry": new_end.isoformat()
                },
                "timestamp": firestore.SERVER_TIMESTAMP,
                "time": datetime.now(timezone.utc).timestamp()
            })
        except Exception as audit_err:
             print(f"[Webhook] Audit Log Error: {audit_err}")
        print(f"[Webhook] User {user_id} extended to {new_end}")
    except Exception as e:
        print(f"[Webhook] Error in charged: {e}")

async def handle_subscription_cancelled(db, payload):
    try:
        entity = payload['payload']['subscription']['entity']
        sub_id = entity['id']
        
        users = db.collection('users').where('membership.subscriptionId', '==', sub_id).limit(1).get()
        if users:
            user_ref = users[0].reference
            user_ref.update({
                "membership.status": "cancelled",
                "membership.autoRenew": False
            })

            # AUDIT LOG: Subscription Cancelled
            try:
                db.collection("auditLogs").add({
                    "action": "MEMBERSHIP_AUTO_CHANGED",
                    "reason": "subscription_cancelled",
                    "by": "system",
                    "target": users[0].id,
                    "details": {
                        "subscriptionId": sub_id
                    },
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "time": datetime.now(timezone.utc).timestamp()
                })
            except Exception as audit_err:
                print(f"[Webhook] Audit Log Error: {audit_err}")
            print(f"[Webhook] Subscription {sub_id} cancelled (User {users[0].id})")
        else:
            print(f"[Webhook] User not found for cancellation {sub_id}")
    except Exception as e:
        print(f"[Webhook] Error in cancelled: {e}")
