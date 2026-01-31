"""
Relyce AI - FastAPI Main Application
Production-grade ChatGPT-style API with REST and WebSocket support
"""
import json
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import HOST, PORT, CORS_ORIGINS, CORS_ORIGIN_REGEX
from app.models import (
    ChatRequest, ChatResponse, SearchRequest,
    HealthResponse, WebSocketMessage, Personality
)
from app.auth import verify_token, initialize_firebase
from app.llm.processor import llm_processor
from app.chat.context import get_context_for_llm, update_context_with_exchange
from app.chat.history import save_message_to_firebase, load_chat_history
from app.websocket import manager, handle_websocket_message
from app.payment import router as payment_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("=" * 60)
    print("   [RELYCE-AI] BACKEND - Starting Up")
    print("=" * 60)
    
    try:
        initialize_firebase()
        print("[Startup] - Firebase initialized")
    except Exception as e:
        print(f"[Startup] ! Firebase init failed: {e}")
    
    print(f"[Startup] - Server ready on {HOST}:{PORT}")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("[Shutdown] - Relyce AI Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Relyce AI API",
    description="Production-grade ChatGPT-style backend with multi-device support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# REST API ENDPOINTS
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )


from app.chat.personalities import (
    get_all_personalities, 
    create_custom_personality, 
    get_personality_by_id
)

@app.get("/personalities/{user_id}")
async def get_personalities(user_id: str):
    """Get all available personalities for a user"""
    return {"success": True, "personalities": get_all_personalities(user_id)}

@app.post("/personalities")
async def create_personality(request: Personality, user_id: str = Query(...)):
    """Create a new custom personality"""
    result = create_custom_personality(
        user_id, 
        request.name, 
        request.description, 
        request.prompt,
        request.content_mode  # Pass content_mode
    )
    if result:
        return {"success": True, "personality": result}
    raise HTTPException(status_code=500, detail="Failed to create personality")


@app.put("/personalities/{personality_id}")
async def update_personality(personality_id: str, request: Personality, user_id: str = Query(...)):
    """Update a custom personality"""
    from app.chat.personalities import update_custom_personality
    
    success = update_custom_personality(
        user_id,
        personality_id,
        request.name,
        request.description,
        request.prompt,
        request.content_mode  # Pass content_mode
    )
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to update")


@app.delete("/personalities/{personality_id}")
async def delete_personality(personality_id: str, user_id: str = Query(...)):
    """Delete a custom personality"""
    from app.chat.personalities import delete_custom_personality
    
    success = delete_custom_personality(user_id, personality_id)
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to delete")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint.
    Processes message and returns complete response.
    """
    try:
        # Get context if session exists
        context_messages = []
        if request.user_id and request.session_id:
            context_messages = get_context_for_llm(request.user_id, request.session_id)
        
        # Resolve personality
        personality = request.personality
        if not personality and request.personality_id and request.user_id:
            p_data = get_personality_by_id(request.user_id, request.personality_id)
            if p_data:
                # Convert dict to expected format if needed by processor, 
                # but processor takes dict. Models.py defines Personality as object.
                # However processor.py type hint says Optional[Dict].
                personality = p_data
        
        # Convert Personality object to dict if it is an object
        if hasattr(personality, "dict"):
            personality = personality.dict()
        elif hasattr(personality, "model_dump"):
            personality = personality.model_dump()

        # Process message
        result = await llm_processor.process_message(
            user_query=request.message,
            mode=request.chat_mode,
            context_messages=context_messages,
            personality=personality,
            user_settings=request.user_settings
        )
        
        # Update context
        if request.user_id and request.session_id:
            update_context_with_exchange(
                request.user_id, 
                request.session_id,
                request.message,
                result["response"]
            )
            
            # Save to Firebase
            save_message_to_firebase(
                request.user_id, request.session_id, "user", request.message
            )
            msg_id = save_message_to_firebase(
                request.user_id, request.session_id, "assistant", result["response"]
            )
            result["message_id"] = msg_id
        
        return ChatResponse(**result)
        
    except Exception as e:
        print(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Returns tokens as they're generated.
    """
    async def generate():
        try:
            # Get context if session exists
            context_messages = []
            if request.user_id and request.session_id:
                context_messages = get_context_for_llm(request.user_id, request.session_id)
            
            # Resolve personality
            personality = request.personality
            if not personality and request.personality_id and request.user_id:
                p_data = get_personality_by_id(request.user_id, request.personality_id)
                if p_data:
                    personality = p_data
            
             # Convert Personality object to dict if it is an object
            if hasattr(personality, "dict"):
                personality = personality.dict()
            elif hasattr(personality, "model_dump"):
                personality = personality.model_dump()

            full_response = ""
            
            async for token in llm_processor.process_message_stream(
                request.message,
                mode=request.chat_mode,
                context_messages=context_messages,
                personality=personality,
                user_settings=request.user_settings
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            
            # Update context and save to Firebase
            if request.user_id and request.session_id:
                update_context_with_exchange(
                    request.user_id,
                    request.session_id,
                    request.message,
                    full_response
                )
                save_message_to_firebase(
                    request.user_id, request.session_id, "user", request.message
                )
                save_message_to_firebase(
                    request.user_id, request.session_id, "assistant", full_response
                )

                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/search")
async def web_search(request: SearchRequest):
    """
    Web search endpoint.
    Returns search results from Serper API.
    """
    from app.llm.router import execute_serper_batch, SERPER_TOOLS
    
    try:
        tools = request.tools or ["Search"]
        results = {}
        
        for tool in tools:
            if tool in SERPER_TOOLS:
                endpoint = SERPER_TOOLS[tool]
                param_key = "url" if tool == "Webpage" else "q"
                result = await execute_serper_batch(endpoint, [request.query], param_key=param_key)
                results[tool] = result
        
        return {"success": True, "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user_id}/{session_id}")
async def get_history(user_id: str, session_id: str, limit: int = 50):
    """Get chat history for a session"""
    try:
        messages = load_chat_history(user_id, session_id, limit)
        return {"success": True, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Payment Router
app.include_router(payment_router, prefix="/payment", tags=["Payment"])



# ============================================
# WEBSOCKET ENDPOINT
# ============================================

@app.websocket("/ws/chat")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    chat_id: Optional[str] = Query(None)
):
    """
    WebSocket chat endpoint with multi-device support.
    
    Connection URL: ws://localhost:8000/ws/chat?token=FIREBASE_TOKEN&chat_id=SESSION_ID
    
    Message format (JSON):
    - Send message: {"type": "message", "content": "Hello", "chat_mode": "normal"}
    - Stop generation: {"type": "stop"}
    - Ping: {"type": "ping"}
    
    Response format (JSON):
    - Token: {"type": "token", "content": "..."}
    - Done: {"type": "done", "content": ""}
    - Error: {"type": "error", "content": "..."}
    - Info: {"type": "info", "content": "processing"|"stopped"}
    """
    # Verify token if provided
    user_id = "anonymous"
    if token:
        is_valid, user_info = verify_token(token)
        if is_valid and user_info:
            user_id = user_info.get("uid", "anonymous")
        else:
            # For development, allow connection without valid token
            print("[WS] ! Invalid token, using anonymous mode")
    
    # Use provided chat_id or generate one
    if not chat_id:
        chat_id = f"chat_{datetime.now().timestamp()}"
    
    # Connect
    connection_id = await manager.connect(websocket, chat_id, user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, connection_id, message, manager)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "content": "Invalid JSON"}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        print(f"[WS] Error: {e}")
        manager.disconnect(connection_id)


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
