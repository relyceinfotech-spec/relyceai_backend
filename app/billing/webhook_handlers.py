"""
Razorpay webhook event handlers.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from firebase_admin import firestore


DEFAULT_PRICING = {
    "starter": {"monthly": 199, "yearly": 1999},
    "plus": {"monthly": 999, "yearly": 9999},
    "pro": {"monthly": 1999, "yearly": 19999},
    "business": {"monthly": 2499, "yearly": 24999},
}

VALID_BILLING_CYCLES = {"monthly", "yearly"}


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


async def handle_payment_captured(db, payload):
    """Handle one-time payments (like UPI scans) that do not go through subscription flow."""
    try:
        entity = payload["payload"]["payment"]["entity"]
        payment_id = entity["id"]
        order_id = entity.get("order_id")

        notes = entity.get("notes", {})
        user_id = notes.get("user_id") or notes.get("userId")
        plan_id = _normalize_plan_id(notes.get("plan_id") or notes.get("plan") or notes.get("planId"))
        billing_cycle = (notes.get("billing_cycle") or notes.get("billingCycle") or "monthly").lower()

        if not user_id or not plan_id:
            print(f"[Webhook] Missing user_id or plan_id in payment notes for {payment_id}")
            return False

        if billing_cycle not in VALID_BILLING_CYCLES:
            print(f"[Webhook] Invalid billing cycle '{billing_cycle}' for {payment_id}")
            return False

        print(f"[Webhook] Processing captured payment {payment_id} for user {user_id}, plan {plan_id}")

        payment_doc_ref = db.collection("payments").document(payment_id)

        now = datetime.now(timezone.utc)
        expiry_date = now + timedelta(days=365 if billing_cycle == "yearly" else 30)

        user_ref = db.collection("users").document(user_id)

        amount_val = entity.get("amount", 0)
        amount_in_rupees = amount_val / 100 if amount_val > 0 else 0

        pricing = _load_pricing(db)
        expected_amount = _get_plan_amount(pricing, plan_id, billing_cycle)
        if not expected_amount:
            print(f"[Webhook] Invalid plan pricing for {plan_id}")
            return False
        if amount_in_rupees and int(amount_in_rupees) != int(expected_amount):
            print(f"[Webhook] Amount mismatch for {payment_id}: expected {expected_amount}, got {amount_in_rupees}")
            return False

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
                "amount": amount_in_rupees,
                "plan": plan_id,
                "date": now.isoformat(),
                "verified": True,
                "source": "webhook",
            },
        }

        try:
            db.collection("auditLogs").add({
                "action": "MEMBERSHIP_AUTO_CHANGED",
                "reason": "payment",
                "by": "system",
                "target": user_id,
                "details": {"plan": plan_id, "paymentId": payment_id, "amount": amount_in_rupees},
                "timestamp": firestore.SERVER_TIMESTAMP,
                "time": now.timestamp(),
            })
        except Exception as audit_err:
            print(f"[Webhook] Audit Log Error: {audit_err}")

        try:
            user_ref.update(update_data)
        except Exception as e:
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
                            "source": "webhook",
                        }
                    },
                }
            }, merge=True)

        payment_doc_ref.set({
            "userId": user_id,
            "orderId": order_id,
            "paymentId": payment_id,
            "planId": plan_id,
            "plan": plan_id,
            "billingCycle": billing_cycle,
            "amount": amount_in_rupees,
            "currency": entity.get("currency", "INR"),
            "method": entity.get("method", "unknown"),
            "email": entity.get("email"),
            "contact": entity.get("contact"),
            "status": "captured",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "verified": True,
            "source": "webhook",
        }, merge=True)

        print(f"[Webhook] Successfully activated plan {plan_id} for user {user_id}")
        return True
    except Exception as e:
        print(f"[Webhook] Error in handle_payment_captured: {e}")
        return False


async def handle_subscription_activated(db, payload):
    try:
        entity = payload["payload"]["subscription"]["entity"]
        sub_id = entity["id"]

        notes = entity.get("notes", {})
        user_id = notes.get("userId") or notes.get("user_id")
        plan_id = notes.get("plan_id") or notes.get("plan") or notes.get("planId")

        if not user_id:
            print(f"[Webhook] No user_id found in notes for subscription {sub_id}")
            return

        print(f"[Webhook] Activating subscription {sub_id} for user {user_id}")
        user_ref = db.collection("users").document(user_id)

        start_ts = entity.get("start_at") or entity.get("created_at")
        if start_ts:
            start_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat()
        else:
            start_date = datetime.now(timezone.utc).isoformat()

        billing_cycle = notes.get("billingCycle", "monthly")
        now = datetime.now(timezone.utc)
        expiry_date = now + timedelta(days=365 if billing_cycle == "yearly" else 30)

        update_data = {
            "membership.status": "active",
            "membership.subscriptionId": sub_id,
            "membership.startDate": start_date,
            "membership.expiryDate": expiry_date.isoformat(),
            "membership.paymentStatus": "paid",
            "membership.billingCycle": billing_cycle,
            "membership.updatedAt": firestore.SERVER_TIMESTAMP,
        }
        if plan_id:
            update_data["membership.plan"] = plan_id
            update_data["membership.planName"] = str(plan_id).capitalize()

        try:
            db.collection("auditLogs").add({
                "action": "MEMBERSHIP_AUTO_CHANGED",
                "reason": "subscription_activated",
                "by": "system",
                "target": user_id,
                "details": {"subscriptionId": sub_id, "billingCycle": billing_cycle},
                "timestamp": firestore.SERVER_TIMESTAMP,
                "time": datetime.now(timezone.utc).timestamp(),
            })
        except Exception as audit_err:
            print(f"[Webhook] Audit Log Error: {audit_err}")

        user_ref.update(update_data)
        print(f"[Webhook] User {user_id} activated.")
    except Exception as e:
        print(f"[Webhook] Error in activation: {e}")


async def handle_subscription_charged(db, payload):
    try:
        entity = payload["payload"]["subscription"]["entity"]
        sub_id = entity["id"]

        notes = entity.get("notes", {})
        user_id = notes.get("userId") or notes.get("user_id")
        plan_id = notes.get("plan_id") or notes.get("plan") or notes.get("planId")

        user_ref = None
        user_doc = None

        if user_id:
            user_ref = db.collection("users").document(user_id)
            user_doc = user_ref.get()

        if not user_doc or not user_doc.exists:
            users = db.collection("users").where("membership.subscriptionId", "==", sub_id).limit(1).get()
            if users:
                user_doc = users[0]
                user_ref = user_doc.reference
                user_id = user_doc.id
            else:
                print(f"[Webhook] User not found for subscription {sub_id}")
                return

        current_data = user_doc.to_dict().get("membership", {})
        current_expiry = current_data.get("expiryDate")

        if current_expiry:
            try:
                old_end = datetime.fromisoformat(current_expiry.replace("Z", "+00:00"))
                if old_end.tzinfo is None:
                    old_end = old_end.replace(tzinfo=timezone.utc)
            except Exception:
                old_end = datetime.now(timezone.utc)
        else:
            old_end = datetime.now(timezone.utc)

        base_date = max(old_end, datetime.now(timezone.utc))
        billing_cycle = current_data.get("billingCycle", "monthly")
        new_end = base_date + timedelta(days=365 if billing_cycle == "yearly" else 30)

        update_data = {
            "membership.expiryDate": new_end.isoformat(),
            "membership.lastBillingAt": firestore.SERVER_TIMESTAMP,
            "membership.status": "active",
        }
        if plan_id:
            update_data["membership.plan"] = plan_id
            update_data["membership.planName"] = str(plan_id).capitalize()
        user_ref.update(update_data)

        try:
            db.collection("auditLogs").add({
                "action": "MEMBERSHIP_AUTO_CHANGED",
                "reason": "subscription_charged",
                "by": "system",
                "target": user_id,
                "details": {"subscriptionId": sub_id, "newExpiry": new_end.isoformat()},
                "timestamp": firestore.SERVER_TIMESTAMP,
                "time": datetime.now(timezone.utc).timestamp(),
            })
        except Exception as audit_err:
            print(f"[Webhook] Audit Log Error: {audit_err}")
        print(f"[Webhook] User {user_id} extended to {new_end}")
    except Exception as e:
        print(f"[Webhook] Error in charged: {e}")


async def handle_subscription_cancelled(db, payload):
    try:
        entity = payload["payload"]["subscription"]["entity"]
        sub_id = entity["id"]

        users = db.collection("users").where("membership.subscriptionId", "==", sub_id).limit(1).get()
        if users:
            user_ref = users[0].reference
            user_ref.update({"membership.status": "cancelled", "membership.autoRenew": False})

            try:
                db.collection("auditLogs").add({
                    "action": "MEMBERSHIP_AUTO_CHANGED",
                    "reason": "subscription_cancelled",
                    "by": "system",
                    "target": users[0].id,
                    "details": {"subscriptionId": sub_id},
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "time": datetime.now(timezone.utc).timestamp(),
                })
            except Exception as audit_err:
                print(f"[Webhook] Audit Log Error: {audit_err}")
            print(f"[Webhook] Subscription {sub_id} cancelled (User {users[0].id})")
        else:
            print(f"[Webhook] User not found for cancellation {sub_id}")
    except Exception as e:
        print(f"[Webhook] Error in cancelled: {e}")
