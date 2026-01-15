"""
Unified Backend Server - Mounts all API routes for Relyce AI
- Library Chat API at /api/library
- Chat API at /api/chat
- WebSocket at /ws/chat
"""
import os
import sys

# CRITICAL: Create FastAPI app FIRST before any imports that might fail
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Create main app IMMEDIATELY so port binding happens even if imports fail
app = FastAPI(
    title="Relyce AI Backend",
    version="1.0.0",
    description="Unified API server for Relyce AI - Chat and Library features"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint - always works
@app.get("/")
async def root():
    return {
        "service": "Relyce AI Backend",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "relyce-backend"}

# Now try to load the rest (with error handling)
print("🚀 Starting Relyce AI Backend...")

try:
    import uvicorn
    from dotenv import load_dotenv
    
    # Load environment variables
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(backend_dir, '.env')
    load_dotenv(env_path)
    print("✅ Environment loaded")
except Exception as e:
    print(f"⚠️ Dotenv error: {e}")

# Initialize Firebase
try:
    import firebase_admin
    from firebase_admin import credentials, storage as fb_storage
    import json
    
    if not firebase_admin._apps:
        storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
        firebase_sdk_json = os.getenv('FIREBASE_ADMIN_SDK')
        
        if firebase_sdk_json:
            service_account = json.loads(firebase_sdk_json)
            if 'private_key' in service_account:
                service_account['private_key'] = service_account['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(service_account)
            firebase_admin.initialize_app(cred, {'storageBucket': storage_bucket})
            print(f"🔥 Firebase Admin initialized (bucket: {storage_bucket})")
        else:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
            if service_account_path:
                if not os.path.isabs(service_account_path):
                    service_account_path = os.path.normpath(os.path.join(backend_dir, service_account_path))
                if os.path.exists(service_account_path):
                    cred = credentials.Certificate(service_account_path)
                    firebase_admin.initialize_app(cred, {'storageBucket': storage_bucket})
                    print(f"✅ Firebase Admin initialized from FILE")
                else:
                    print(f"⚠️ Firebase SDK not found at: {service_account_path}")
            else:
                print("⚠️ No Firebase credentials found")
except Exception as e:
    print(f"❌ Firebase init error: {e}")

# Add directories to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.join(backend_dir, 'bucket'))
sys.path.insert(0, os.path.join(backend_dir, 'chat'))

# Import and mount sub-applications
try:
    from library_api import library_app
    app.mount("/api", library_app)
    print("✅ Library API mounted at /api")
except Exception as e:
    print(f"⚠️ Could not import library_api: {e}")

try:
    from chat.server import chat_app
    app.mount("/chat", chat_app)
    print("✅ Chat API mounted at /chat")
except Exception as e:
    print(f"⚠️ Could not import chat.server: {e}")

# WebSocket endpoint
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Main WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    session_id = websocket.query_params.get("sessionId", "default")
    user_id = websocket.query_params.get("userId", "anonymous")
    
    print(f"🔗 WebSocket connected: user={user_id}, session={session_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            message = data.get("message", "")
            chat_mode = data.get("chatMode", "standard")
            
            if message == "STOP_PROCESSING":
                await websocket.send_json({"type": "stopped"})
                continue
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            print(f"📨 WS Message: {message[:50]}... (mode={chat_mode})")
            
            try:
                # Lazy import to avoid startup issues
                from chat.server import get_genric_module, get_business_module
                import concurrent.futures
                
                if chat_mode == "plus":
                    bm = get_business_module()
                    response = bm.generate_final_response(message, "", "")
                else:
                    gm = get_genric_module()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_web = executor.submit(gm.perform_web_search, message, 5)
                        web_text = future_web.result()
                    response = gm.generate_final_response(message, "", web_text)
                
                await websocket.send_json({
                    "type": "bot",
                    "text": response,
                    "messageId": data.get("messageId"),
                    "chatId": data.get("chatId")
                })
                
            except Exception as e:
                print(f"❌ Chat error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "text": f"Error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*50)
    print("   RELYCE AI UNIFIED BACKEND")
    print("="*50)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
