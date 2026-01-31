from fastapi import APIRouter, HTTPException, Depends, Body, Request
from app.config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET
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
    notes: dict = Body(None)
):
    """
    Create a new Razorpay order.
    Amount should be in subunits (e.g., paise for INR).
    """
    try:
        data = {
            "amount": amount,
            "currency": currency,
            "receipt": receipt,
            "notes": notes,
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
    user_id: str = Body(...)
):
    """
    Verify Razorpay payment signature and update user membership securely.
    """
    try:
        # 1. Verify signature
        params_dict = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        
        client.utility.verify_payment_signature(params_dict)
        
        # 2. Payment Verified - Calculate Expiry
        from datetime import datetime, timedelta
        import firebase_admin
        from firebase_admin import firestore
        
        db = firestore.client()
        
        # Calculate expiry date
        now = datetime.now()
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
            "membership.paymentHistory": {
                razorpay_payment_id: {
                    "transactionId": razorpay_payment_id,
                    "orderId": razorpay_order_id,
                    "amount": 0, # Should ideally be fetched from order/payment details
                    "plan": plan_id,
                    "date": now.isoformat(),
                    "verified": True
                }
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
                            "amount": 0,
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
            "timestamp": firestore.SERVER_TIMESTAMP,
            "verified": True
        })
        
        print(f"[Payment] Verified and updated for user {user_id} - Plan: {plan_id}")
        return {"success": True, "message": "Payment verified and membership updated"}
        
    except razorpay.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Signature verification failed")
    except Exception as e:
        print(f"[Razorpay] Verification error: {e}")
        raise HTTPException(status_code=500, detail="Payment verification failed")

@router.post("/webhook/razorpay")
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

async def handle_subscription_activated(db, payload):
    try:
        entity = payload['payload']['subscription']['entity']
        sub_id = entity['id']
        
        notes = entity.get('notes', {})
        user_id = notes.get('userId') or notes.get('user_id')
        
        if not user_id:
            print(f"[Webhook] No user_id found in notes for subscription {sub_id}")
            return

        print(f"[Webhook] Activating subscription {sub_id} for user {user_id}")
        
        # Update user
        user_ref = db.collection('users').document(user_id)
        
        from datetime import datetime, timedelta
        start_ts = entity.get('start_at') or entity.get('created_at')
        
        if start_ts:
            start_date = datetime.fromtimestamp(start_ts).isoformat()
        else:
            start_date = datetime.now().isoformat()
            
        billing_cycle = notes.get('billingCycle', 'monthly')
        
        now = datetime.now()
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
        
        from datetime import datetime, timedelta
        
        if current_expiry:
            try:
                old_end = datetime.fromisoformat(current_expiry)
            except:
                old_end = datetime.now()
        else:
            old_end = datetime.now()
            
        # Ensure we don't extend from a long-expired date
        base_date = max(old_end, datetime.now())
        
        billing_cycle = current_data.get('billingCycle', 'monthly')
        
        if billing_cycle == 'yearly':
            new_end = base_date + timedelta(days=365)
        else:
            new_end = base_date + timedelta(days=30)
            
        user_ref.update({
            "membership.expiryDate": new_end.isoformat(),
            "membership.lastBillingAt": firestore.SERVER_TIMESTAMP,
            "membership.status": "active"
        })
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
            print(f"[Webhook] Subscription {sub_id} cancelled (User {users[0].id})")
        else:
            print(f"[Webhook] User not found for cancellation {sub_id}")
    except Exception as e:
        print(f"[Webhook] Error in cancelled: {e}")
