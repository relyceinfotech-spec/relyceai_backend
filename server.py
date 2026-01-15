"""
Unified Backend Server - Mounts all API routes for Relyce AI
CRITICAL: Minimal imports at top level for fast port binding on Render
"""
import os

# CRITICAL: Only import FastAPI essentials - nothing else at top level!
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Create app IMMEDIATELY - this must happen before any heavy imports
app = FastAPI(
    title="Relyce AI Backend",
    version="1.0.0",
    description="Unified API server for Relyce AI"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple endpoints that work immediately (for health checks)
@app.get("/")
async def root():
    return {"service": "Relyce AI Backend", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "relyce-backend"}

# ============================================================================
# LAZY LOADING: All heavy modules loaded on first request, not at startup
# ============================================================================
_initialized = False
_library_app = None
_chat_app = None

def lazy_init():
    """Initialize heavy modules on first request"""
    global _initialized, _library_app, _chat_app
    
    if _initialized:
        return
    
    import sys
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, backend_dir)
    sys.path.insert(0, os.path.join(backend_dir, 'bucket'))
    sys.path.insert(0, os.path.join(backend_dir, 'chat'))
    
    # Load dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(backend_dir, '.env'))
        print("✅ Environment loaded")
    except Exception as e:
        print(f"⚠️ Dotenv: {e}")
    
    # Firebase
    try:
        import firebase_admin
        from firebase_admin import credentials
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
                print(f"🔥 Firebase initialized")
    except Exception as e:
        print(f"⚠️ Firebase: {e}")
    
    # Mount sub-apps
    try:
        from library_api import library_app
        app.mount("/api", library_app)
        _library_app = library_app
        print("✅ Library API mounted at /api")
    except Exception as e:
        print(f"⚠️ Library API: {e}")
    
    try:
        from chat.server import chat_app
        app.mount("/chat", chat_app)
        _chat_app = chat_app
        print("✅ Chat API mounted at /chat")
    except Exception as e:
        print(f"⚠️ Chat API: {e}")
    
    _initialized = True
    print("✅ All modules loaded!")

# Middleware to lazy-init on first real request
@app.middleware("http")
async def lazy_load_middleware(request: Request, call_next):
    # Skip lazy init for health checks (so Render can detect port fast)
    if request.url.path not in ["/", "/health"]:
        lazy_init()
    response = await call_next(request)
    return response

# Manual init endpoint (optional - call once to warm up)
@app.get("/init")
async def init_modules():
    lazy_init()
    return {"status": "initialized", "modules_loaded": _initialized}

# WebSocket endpoint
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    lazy_init()  # Ensure modules are loaded
    await websocket.accept()
    
    session_id = websocket.query_params.get("sessionId", "default")
    user_id = websocket.query_params.get("userId", "anonymous")
    print(f"🔗 WebSocket: user={user_id}, session={session_id}")
    
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
            
            print(f"📨 WS: {message[:50]}...")
            
            try:
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
                await websocket.send_json({"type": "error", "text": str(e)})
                
    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_id}")

# Only for local development
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Starting on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
