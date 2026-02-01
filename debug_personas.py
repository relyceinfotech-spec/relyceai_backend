
import firebase_admin
from firebase_admin import credentials, firestore
import os

# Initialize with the specific service account to be sure
cred_path = "d:/finalai/relyceai/backend/serviceAccountKey.json"

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    # We want to check the 'relyceinfotech' database specifically as that's what we are enforcing
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://relyceinfotech-21ded.firebaseio.com',
        'databaseAuthVariableOverride': {'uid': 'admin-listing-script'}
    })

# Connect to the SPECIFIC database ID we are targeting
db = firestore.client(database_id="relyceinfotech")

print(f"--- Listing Personalities from DB: relyceinfotech ---")
try:
    docs = db.collection('personalities').stream()
    count = 0
    for doc in docs:
        data = doc.to_dict()
        print(f"ID: {doc.id} | Name: {data.get('name', 'Unknown')} | Public: {data.get('is_public')} | CreatedBy: {data.get('created_by')}")
        count += 1
    
    if count == 0:
        print("No personalities found in 'relyceinfotech' database.")
except Exception as e:
    print(f"Error accessing relyceinfotech: {e}")

print("\n--- Checking (default) database content for comparison ---")
try:
    # Hacky way to check default without re-init app if possible, or just note to user
    # Firestore client in python is tied to the app init. 
    # We will try to get a client for default if possible, but usually requires separate app.
    print("(Note: To check '(default)' accurately, we might need a separate run or app instance if the above yields nothing.)")
except Exception as e:
    print(e)
