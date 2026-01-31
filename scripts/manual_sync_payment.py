
import sys
import os
import firebase_admin
from firebase_admin import credentials, firestore
import razorpay
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir) # Parent of scripts/ is backend/
env_path = os.path.join(backend_dir, ".env")

# Load environment variables
load_dotenv(env_path)

# --- Razorpay Config ---
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    print("Error: RAZORPAY_KEY_ID or RAZORPAY_KEY_SECRET not found in .env")
    sys.exit(1)

# --- Firebase Config ---
# Replicating logic from app/auth.py and app/config.py
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_PRIVATE_KEY_ID = os.getenv("FIREBASE_PRIVATE_KEY_ID")
FIREBASE_PRIVATE_KEY = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")
FIREBASE_CLIENT_EMAIL = os.getenv("FIREBASE_CLIENT_EMAIL")
FIREBASE_CLIENT_ID = os.getenv("FIREBASE_CLIENT_ID")
FIREBASE_CLIENT_CERT_URL = os.getenv("FIREBASE_CLIENT_CERT_URL")

# Initialize Razorpay Client
client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# Initialize Firebase
if not firebase_admin._apps:
    print("Initializing Firebase...")
    try:
        cred_dict = {
            "type": "service_account",
            "project_id": FIREBASE_PROJECT_ID,
            "private_key_id": FIREBASE_PRIVATE_KEY_ID,
            "private_key": FIREBASE_PRIVATE_KEY,
            "client_email": FIREBASE_CLIENT_EMAIL,
            "client_id": FIREBASE_CLIENT_ID,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": FIREBASE_CLIENT_CERT_URL
        }
        
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized successfully.")
    except Exception as e:
        print(f"❌ Firebase initialization failed: {e}")
        # Try ApplicationDefault as fallback
        print("Trying Application Default Credentials...")
        try:
             cred = credentials.ApplicationDefault()
             firebase_admin.initialize_app(cred, {'projectId': FIREBASE_PROJECT_ID})
             print("✅ Firebase initialized with Application Default Credentials.")
        except Exception as e2:
             print(f"❌ Fallback failed: {e2}")
             sys.exit(1)

db = firestore.client()

def sync_payment(payment_id):
    print(f"Fetching payment {payment_id} from Razorpay...")
    
    try:
        payment = client.payment.fetch(payment_id)
    except Exception as e:
        print(f"Error fetching payment from Razorpay: {e}")
        return

    status = payment.get('status')
    if status != 'captured':
        print(f"Payment status is '{status}', not 'captured'. Cannot activate.")
        return

    notes = payment.get('notes', {})
    user_id = notes.get('user_id') or notes.get('userId')
    plan_id = notes.get('plan_id') or notes.get('plan')
    billing_cycle = notes.get('billing_cycle') or notes.get('billingCycle') or 'monthly'
    amount = payment.get('amount')
    order_id = payment.get('order_id')
    email = payment.get('email')

    print(f"Payment Details found:")
    print(f"  User ID: {user_id}")
    print(f"  Plan: {plan_id}")
    print(f"  Amount: {amount/100}")
    print(f"  Email: {email}")

    if not user_id:
        print("⚠️  No user_id found in payment notes!")
        if email:
             # Try to find user by email
             print(f"Searching for user with email: {email}...")
             users = db.collection('users').where('email', '==', email).limit(1).get()
             if users:
                 user_id = users[0].id
                 print(f"✅ Found User ID: {user_id}")
             else:
                 print("❌ No user found with that email.")
        
        if not user_id:
            user_id = input(f"Enter User ID manually: ").strip()
            if not user_id:
                 print("Aborting.")
                 return
    
    if not plan_id:
        print("⚠️  No plan_id found in payment notes!")
        plan_id = input("Enter Plan ID manually (e.g. starter, pro): ").strip()
        if not plan_id:
             print("Aborting.")
             return

    # Confirm with user
    confirm = input(f"Activate {plan_id} plan for user {user_id}? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborting.")
        return

    print("Updating Firestore...")
    
    # Calculate expiry
    now = datetime.now()
    if billing_cycle == 'yearly':
        expiry_date = now + timedelta(days=365)
    else:
        expiry_date = now + timedelta(days=30)

    user_ref = db.collection('users').document(user_id)
    
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
            "amount": amount/100,
            "plan": plan_id,
            "date": now.isoformat(),
            "verified": True,
            "source": "manual_sync_script"
        }
    }

    try:
        user_ref.update(update_data)
        print("✅ User membership updated successfully.")
    except Exception as e:
        print(f"Error updating user (trying set...): {e}")
        # Try set if doc missing
        try:
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
                            "amount": amount/100,
                            "plan": plan_id,
                            "date": now.isoformat(),
                            "verified": True,
                            "source": "manual_sync_script"
                        }
                    }
                }
            }, merge=True)
             print("✅ User membership set successfully.")
        except Exception as e2:
             print(f"❌ Failed to update user: {e2}")

    
    # Update payment log 
    try:
        db.collection('payments').document(payment_id).set({
            "userId": user_id,
            "orderId": order_id,
            "paymentId": payment_id,
            "planId": plan_id,
            "billingCycle": billing_cycle,
            "amount": amount/100,
            "status": "captured",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "verified": True,
            "source": "manual_sync_script"
        })
        print("✅ Payment logged in 'payments' collection.")
    except Exception as e:
        print(f"❌ Failed to log payment: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/manual_sync_payment.py <razorpay_payment_id>")
    else:
        sync_payment(sys.argv[1])
