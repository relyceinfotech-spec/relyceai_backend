"""
Chat Server API - FastAPI endpoints for Generic and Business chat modes
Imports functionality from genric.py and business.py modules
"""
import os
import sys
import time
import concurrent.futures
from typing import Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from modules - lazy loading to avoid startup issues
genric_module = None
business_module = None

def get_genric_module():
    global genric_module
    if genric_module is None:
        import genric as gm
        genric_module = gm
    return genric_module

def get_business_module():
    global business_module
    if business_module is None:
        import business as bm
        business_module = bm
    return business_module

# FastAPI app
app = FastAPI(title="Chat API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    userId: Optional[str] = None
    mode: str = "generic"  # generic or business
    k_val: int = 5  # Number of web results to fetch

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    mode: str = "generic"
    processing_time: Optional[float] = None

# Store active WebSocket connections
active_connections: dict = {}

@app.post("/chat", response_model=ChatResponse)
async def chat_generic(request: ChatRequest):
    """Chat endpoint for Generic mode - hybrid RAG + Web search"""
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    mode = request.mode.lower()
    print(f"\n📨 Chat Request | Mode: {mode.upper()}")
    print(f"📝 Query: {request.query[:50]}...")
    
    start_time = time.time()
    
    try:
        if mode == "business":
            # Use business module for business mode
            bm = get_business_module()
            response = bm.generate_final_response(request.query, "", "")
        else:
            # Use genric module for generic mode
            gm = get_genric_module()
            
            # Run parallel search
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_web = executor.submit(gm.perform_web_search, request.query, request.k_val)
                web_text = future_web.result()
            
            # Generate response (no RAG context for now, just web search)
            response = gm.generate_final_response(request.query, "", web_text)
        
        processing_time = time.time() - start_time
        print(f"✅ Response generated in {processing_time:.2f}s")
        
        return ChatResponse(
            response=response,
            status="success",
            mode=mode,
            processing_time=processing_time
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming chat"""
    await websocket.accept()
    
    session_id = websocket.query_params.get("sessionId", "default")
    user_id = websocket.query_params.get("userId", "anonymous")
    
    active_connections[session_id] = websocket
    print(f"🔗 WebSocket connected: {session_id}")
    
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
            
            print(f"📨 WS Message from {user_id}: {message[:50]}...")
            
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
                await websocket.send_json({
                    "type": "error",
                    "text": f"Error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "chat-api"}

@app.get("/")
async def root():
    return {
        "service": "Chat API",
        "modes": ["generic", "business"],
        "endpoints": {
            "POST /chat": "Send chat message",
            "WS /ws/chat": "WebSocket for real-time chat"
        }
    }

# Export app for mounting in unified server
chat_app = app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*40)
    print("   CHAT API SERVER")
    print("="*40)
    print("POST /chat - Generic/Business chat")
    print("WS /ws/chat - WebSocket chat")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8001)
