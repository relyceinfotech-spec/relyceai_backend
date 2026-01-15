"""
Unified Backend Server - Mounts all API routes for Relyce AI
- Library Chat API at /api/library
- Chat API at /api/chat
- WebSocket at /ws/chat
"""
import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from backend/.env
backend_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

# Fix Firebase service account path to be absolute
firebase_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
if firebase_path and not os.path.isabs(firebase_path):
    # Convert relative path to absolute based on backend directory
    abs_firebase_path = os.path.normpath(os.path.join(backend_dir, firebase_path))
    os.environ['FIREBASE_SERVICE_ACCOUNT_PATH'] = abs_firebase_path
    print(f"📁 Firebase SDK path: {abs_firebase_path}")

# Initialize Firebase BEFORE importing any modules that use it
import firebase_admin
from firebase_admin import credentials, storage as fb_storage

if not firebase_admin._apps:
    try:
        service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
        storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
        
        if service_account_path and os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred, {'storageBucket': storage_bucket})
            print(f"✅ Firebase Admin initialized (bucket: {storage_bucket})")
        else:
            print(f"⚠️ Firebase SDK not found at: {service_account_path}")
    except Exception as e:
        print(f"❌ Firebase init failed: {e}")

# Add directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bucket'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat'))

# Create main app
app = FastAPI(
    title="Relyce AI Backend",
    version="1.0.0",
    description="Unified API server for Relyce AI - Chat and Library features"
)

# CORS for frontend (Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount sub-applications
try:
    # Import library API - mounted at /api since it already has /library/chat route
    from library_api import library_app
    app.mount("/api", library_app)
    print("✅ Library API mounted at /api (routes: /api/library/chat)")
except ImportError as e:
    print(f"⚠️ Could not import library_api: {e}")

try:
    # Import chat API - mounted at /chat for POST endpoint
    from chat.server import chat_app
    app.mount("/chat", chat_app)
    print("✅ Chat API mounted at /chat")
except ImportError as e:
    print(f"⚠️ Could not import chat.server: {e}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Relyce AI Backend",
        "version": "1.0.0",
        "endpoints": {
            "/api/library/chat": "POST - Chat with library documents",
            "/api/chat/chat": "POST - Generic/Business chat",
            "/ws/chat": "WebSocket - Real-time chat"
        },
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "relyce-backend"}

# WebSocket proxy - forward to chat app's WebSocket
from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket("/ws/chat")
async def websocket_proxy(websocket: WebSocket):
    """Main WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    session_id = websocket.query_params.get("sessionId", "default")
    user_id = websocket.query_params.get("userId", "anonymous")
    
    print(f"🔗 WebSocket connected: user={user_id}, session={session_id}")
    
    try:
        # Lazy import to avoid circular imports
        from chat.server import get_genric_module, get_business_module
        import concurrent.futures
        
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
                # Determine which mode to use
                if chat_mode == "plus":  # Plus = Business mode
                    bm = get_business_module()
                    response = bm.generate_final_response(message, "", "")
                else:
                    gm = get_genric_module()
                    
                    # Web search
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_web = executor.submit(gm.perform_web_search, message, 5)
                        web_text = future_web.result()
                    
                    response = gm.generate_final_response(message, "", web_text)
                
                # Send response
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
    print("\n" + "="*50)
    print("   RELYCE AI UNIFIED BACKEND")
    print("="*50)
    print("\nEndpoints:")
    print("  POST /api/library/chat - Library document chat")
    print("  POST /api/chat/chat    - Generic/Business chat")
    print("  WS   /ws/chat          - WebSocket real-time chat")
    print("\nStarting server on http://0.0.0.0:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
