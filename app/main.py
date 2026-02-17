"""
Relyce AI - FastAPI Main Application
Production-grade ChatGPT-style API with REST and WebSocket support
"""
import json
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.websockets import WebSocketState

from app.config import (
    HOST,
    PORT,
    CORS_ORIGINS,
    CORS_ORIGIN_REGEX,
    CORS_ALLOWED_METHODS,
    CORS_ALLOWED_HEADERS,
    FORCE_HTTPS,
    MAX_CHAT_MESSAGE_CHARS,
    RATE_LIMIT_PER_MINUTE,
    SECURITY_HEADERS_ENABLED,
    SECURITY_CSP,
)
from app.models import (
    ChatRequest, ChatResponse, SearchRequest,
    HealthResponse, WebSocketMessage, Personality
)
from app.auth import verify_token, initialize_firebase, get_current_user
from app.llm.processor import llm_processor
from app.chat.context import get_context_for_llm, update_context_with_exchange
from app.chat.user_profile import get_user_settings, get_session_personality_id, merge_settings
from app.chat.history import save_message_to_firebase, load_chat_history, increment_message_count
from app.rate_limit import check_rate_limit as check_chat_rate_limit
from app.websocket import manager, handle_websocket_message
from app.llm.emotion_engine import emotion_engine
from app.llm.feedback_engine import feedback_engine
from app.llm.prompt_optimizer import prompt_optimizer
from app.llm.user_profiler import user_profiler

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
    
    # Load intent embeddings into memory (for fast routing) in the background
    try:
        import asyncio
        from app.llm.embeddings import load_intent_embeddings

        async def _load_embeddings_background():
            try:
                loaded = await load_intent_embeddings()
                if loaded:
                    print("[Startup] - Intent embeddings loaded into RAM")
                else:
                    print("[Startup] - Intent embeddings not found in Firestore")
            except Exception as e:
                print(f"[Startup] ! Embedding load failed: {e}")

        asyncio.create_task(_load_embeddings_background())

    except Exception as e:
        print(f"[Startup] ! Embedding task init failed: {e}")

    # Load self-learning scores from Firestore (strategy scores + prompt variants)
    try:
        async def _load_learning_scores():
            try:
                await feedback_engine.load_scores()
                print("[Startup] - Feedback engine scores loaded")
            except Exception as e:
                print(f"[Startup] ! Feedback score load failed: {e}")
            try:
                await prompt_optimizer.load_stats()
                print("[Startup] - Prompt optimizer stats loaded")
            except Exception as e:
                print(f"[Startup] ! Prompt optimizer load failed: {e}")

        asyncio.create_task(_load_learning_scores())
    except Exception as e:
        print(f"[Startup] ! Learning score task init failed: {e}")

    print(f"[Startup] - Server ready on {HOST}:{PORT}")
    print("=" * 60)
    
    yield
    
    # Shutdown
    await emotion_engine.shutdown()
    await feedback_engine.shutdown()
    await prompt_optimizer.save_stats()
    await user_profiler.shutdown()
    print("[Shutdown] - Relyce AI Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Relyce AI API",
    description="Production-grade ChatGPT-style backend with multi-device support",
    version="1.0.0",
    lifespan=lifespan
)

# HTTPS redirect & HSTS (enable in production)
if FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

    @app.middleware("http")
    async def add_hsts_header(request, call_next):
        response = await call_next(request)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)

# Security headers (skip CSP on docs/redoc to avoid breaking Swagger UI)
if SECURITY_HEADERS_ENABLED:
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "interest-cohort=()")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        path = request.url.path
        if not (path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi.json")):
            response.headers.setdefault("Content-Security-Policy", SECURITY_CSP)
            response.headers.setdefault("X-Frame-Options", "DENY")
        return response


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


# ============================================
# AUTH RATE LIMITING ENDPOINTS
# ============================================
from fastapi import Request
from pydantic import BaseModel, EmailStr
from app.rate_limiter import check_rate_limit as check_login_rate_limit, record_failed_attempt, clear_attempts

class RateLimitRequest(BaseModel):
    email: EmailStr

@app.post("/auth/check-limit")
async def check_login_limit(request: RateLimitRequest, req: Request):
    """Check if login attempt is allowed for this email+IP"""
    ip = req.client.host if req.client else "unknown"
    result = check_login_rate_limit(request.email, ip)
    return result

@app.post("/auth/record-failure")
async def record_login_failure(request: RateLimitRequest, req: Request):
    """Record a failed login attempt"""
    ip = req.client.host if req.client else "unknown"
    result = record_failed_attempt(request.email, ip)
    return result

@app.post("/auth/clear-attempts")
async def clear_login_attempts(request: RateLimitRequest, req: Request):
    """Clear login attempts on successful login"""
    ip = req.client.host if req.client else "unknown"
    success = clear_attempts(request.email, ip)
    return {"success": success}

from app.chat.personalities import (
    get_all_personalities, 
    create_custom_personality, 
    get_personality_by_id
)

@app.get("/personalities/{user_id}")
async def get_personalities(user_id: str, user_info: dict = Depends(get_current_user)):
    """Get all available personalities for a user"""
    uid = user_info["uid"]
    return {"success": True, "personalities": get_all_personalities(uid)}

@app.post("/personalities")
async def create_personality(request: Personality, user_info: dict = Depends(get_current_user)):
    """Create a new custom personality"""
    user_id = user_info["uid"]
    result = create_custom_personality(
        user_id, 
        request.name, 
        request.description, 
        request.prompt,
        request.content_mode,  # Pass content_mode
        request.specialty  # Pass specialty
    )
    if result:
        return {"success": True, "personality": result}
    raise HTTPException(status_code=500, detail="Failed to create personality")


@app.put("/personalities/{personality_id}")
async def update_personality(personality_id: str, request: Personality, user_info: dict = Depends(get_current_user)):
    """Update a custom personality"""
    from app.chat.personalities import update_custom_personality
    
    user_id = user_info["uid"]
    success = update_custom_personality(
        user_id,
        personality_id,
        request.name,
        request.description,
        request.prompt,
        request.content_mode,  # Pass content_mode
        request.specialty  # Pass specialty
    )
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to update")


@app.delete("/personalities/{personality_id}")
async def delete_personality(personality_id: str, user_info: dict = Depends(get_current_user)):
    """Delete a custom personality"""
    from app.chat.personalities import delete_custom_personality
    
    user_id = user_info["uid"]
    success = delete_custom_personality(user_id, personality_id)
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to delete")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_info: dict = Depends(get_current_user)):
    """
    Non-streaming chat endpoint.
    Processes message and returns complete response.
    """
    try:
        if len(request.message or "") > MAX_CHAT_MESSAGE_CHARS:
            raise HTTPException(status_code=413, detail="Message too long")
        user_id = user_info["uid"]
        if not check_chat_rate_limit(user_id):
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min). Try again later.")
        request.user_id = user_id

        # Resolve personality
        personality = request.personality
        personality_id = request.personality_id
        if personality and not personality_id:
            if hasattr(personality, "id"):
                personality_id = personality.id
        if not personality and not personality_id and request.session_id:
            saved_id = get_session_personality_id(user_id, request.session_id)
            if saved_id:
                personality_id = saved_id
        if not personality and personality_id:
            p_data = get_personality_by_id(user_id, personality_id)
            if p_data:
                personality = p_data
        # Default to Relyce AI for normal mode if none provided
        if not personality and request.chat_mode == "normal":
            p_data = get_personality_by_id(user_id, "default_relyce")
            if p_data:
                personality = p_data
                personality_id = personality_id or p_data.get("id")
        
        # Convert Personality object to dict if it is an object
        if hasattr(personality, "dict"):
            personality = personality.dict()
        elif hasattr(personality, "model_dump"):
            personality = personality.model_dump()
        if personality and not personality_id:
            personality_id = personality.get("id")

        # Get context if session exists (personality-aware)
        context_messages = []
        if request.session_id:
            context_messages = get_context_for_llm(user_id, request.session_id, personality_id)

        # Process message
        effective_settings = merge_settings(get_user_settings(user_id), request.user_settings)
        result = await llm_processor.process_message(
            user_query=request.message,
            mode=request.chat_mode,
            context_messages=context_messages,
            personality=personality,
            user_settings=effective_settings,
            user_id=user_id,
            session_id=request.session_id
        )
        
        # Update context
        if request.session_id:
            update_context_with_exchange(
                user_id, 
                request.session_id,
                request.message,
                result["response"],
                personality_id
            )
            
            # Save to Firebase
            save_message_to_firebase(
                user_id, request.session_id, "user", request.message, personality_id
            )
            msg_id = save_message_to_firebase(
                user_id, request.session_id, "assistant", result["response"], personality_id
            )
            result["message_id"] = msg_id
            
            # Increment Usage
            increment_message_count(user_id)
        
        return ChatResponse(**result)
        
    except Exception as e:
        print(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, user_info: dict = Depends(get_current_user)):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Returns tokens as they're generated.
    """
    if len(request.message or "") > MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="Message too long")
    user_id = user_info["uid"]
    if not check_chat_rate_limit(user_id):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min). Try again later.")

    async def generate():
        try:
            # 🚀 Force flush buffer immediately with padding (1KB)
            # This helps in environments like Vercel/Nginx/Render that buffer responses
            yield ": " + (" " * 1024) + "\n\n"

            request.user_id = user_id

            # Resolve personality
            personality = request.personality
            personality_id = request.personality_id
            if personality and not personality_id:
                if hasattr(personality, "id"):
                    personality_id = personality.id
            if not personality and not personality_id and request.session_id:
                saved_id = get_session_personality_id(user_id, request.session_id)
                if saved_id:
                    personality_id = saved_id
            if not personality and personality_id:
                p_data = get_personality_by_id(user_id, personality_id)
                if p_data:
                    personality = p_data
            # Default to Relyce AI for normal mode if none provided
            if not personality and request.chat_mode == "normal":
                p_data = get_personality_by_id(user_id, "default_relyce")
                if p_data:
                    personality = p_data
                    personality_id = personality_id or p_data.get("id")
            
             # Convert Personality object to dict if it is an object
            if hasattr(personality, "dict"):
                personality = personality.dict()
            elif hasattr(personality, "model_dump"):
                personality = personality.model_dump()
            if personality and not personality_id:
                personality_id = personality.get("id")

            # Get context if session exists (personality-aware)
            context_messages = []
            if request.session_id:
                context_messages = get_context_for_llm(user_id, request.session_id, personality_id)

            full_response = ""
            
            effective_settings = merge_settings(get_user_settings(user_id), request.user_settings)
            async for token in llm_processor.process_message_stream(
                request.message,
                mode=request.chat_mode,
                context_messages=context_messages,
                personality=personality,
                user_settings=effective_settings,
                user_id=user_id, # Pass User ID for facts
                session_id=request.session_id
            ):
                if token.strip().startswith("[INFO]"):
                    clean_info = token.replace("[INFO]", "").strip()
                    yield f"data: {json.dumps({'type': 'info', 'content': clean_info})}\n\n"
                    continue
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            
            # Update context and save to Firebase
            if request.session_id:
                update_context_with_exchange(
                    user_id,
                    request.session_id,
                    request.message,
                    full_response,
                    personality_id
                )
                save_message_to_firebase(
                    user_id, request.session_id, "user", request.message, personality_id
                )
                save_message_to_firebase(
                    user_id, request.session_id, "assistant", full_response, personality_id
                )
                
                # Increment Usage
                increment_message_count(user_id)
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # Critical for Nginx/Vercel
            "Content-Type": "text/event-stream"
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


@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50, user_info: dict = Depends(get_current_user)):
    """Get chat history for a session"""
    try:
        uid = user_info["uid"]
        messages = load_chat_history(uid, session_id, limit)
        return {"success": True, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Payment Router
app.include_router(payment_router, prefix="/payment", tags=["Payment"])

# Admin & User Management
from app.routers import admin, users, files
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(files.router, prefix="", tags=["Files"]) # Mount at root to match /upload expectation or /files



# ============================================
# WEBSOCKET ENDPOINT
# ============================================

@app.websocket("/ws/chat")
async def websocket_endpoint(
    websocket: WebSocket,
    chat_id: Optional[str] = Query(None)
):
    """
    WebSocket chat endpoint with multi-device support.
    
    Connection URL: ws://localhost:8080/ws/chat?chat_id=SESSION_ID
    
    Message format (JSON):
    - Authenticate: {"type": "auth", "token": "FIREBASE_TOKEN", "chat_id": "SESSION_ID"}
    - Send message: {"type": "message", "content": "Hello", "chat_mode": "normal"}
    - Stop generation: {"type": "stop"}
    - Ping: {"type": "ping"}
    
    Response format (JSON):
    - Token: {"type": "token", "content": "..."}
    - Done: {"type": "done", "content": ""}
    - Error: {"type": "error", "content": "..."}
    - Info: {"type": "info", "content": "processing"|"stopped"}
    """

    # Accept connection immediately to handle errors gracefully
    await websocket.accept()

    async def safe_send(payload: dict) -> bool:
        """Send only if the socket is still open."""
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            await websocket.send_text(json.dumps(payload))
            return True
        except Exception:
            return False

    # Authenticate via initial auth message (preferred) or query param (legacy)
    token = None
    resolved_chat_id = chat_id

    try:
        auth_raw = await asyncio.wait_for(websocket.receive_text(), timeout=6)
        auth_msg = json.loads(auth_raw)
        if auth_msg.get("type") == "auth":
            token = auth_msg.get("token")
            if not resolved_chat_id:
                resolved_chat_id = auth_msg.get("chat_id")
        else:
            await safe_send({"type": "error", "content": "Unauthorized: Missing auth"})
            await websocket.close(code=1008)
            return
    except asyncio.TimeoutError:
        await safe_send({"type": "error", "content": "Unauthorized: Auth timeout"})
        await websocket.close(code=1008)
        return
    except Exception:
        await safe_send({"type": "error", "content": "Unauthorized: Invalid auth"})
        await websocket.close(code=1008)
        return

    if token:
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token or len(token) < 10:
            await safe_send({"type": "error", "content": "Invalid token format"})
            await websocket.close(code=1008)
            return

    if not token:
        await safe_send({"type": "error", "content": "Unauthorized: Missing token"})
        await websocket.close(code=1008)
        return

    # Verify token safely
    try:
        is_valid, user_info = verify_token(token)
    except Exception as e:
        print(f"[WS] Auth Error: {e}")
        is_valid, user_info = False, None

    if not is_valid or not user_info:
        await safe_send({"type": "error", "content": "Unauthorized: Invalid token"})
        await websocket.close(code=1008)
        return

    user_id = user_info.get("uid")
    if not user_id:
        await safe_send({"type": "error", "content": "Unauthorized: Invalid user"})
        await websocket.close(code=1008)
        return
    
    # Use provided chat_id or generate one
    if not resolved_chat_id:
        resolved_chat_id = f"chat_{datetime.now().timestamp()}"
    
    # Connect
    try:
        connection_id = await manager.connect(websocket, resolved_chat_id, user_id)
    except Exception as e:
        await safe_send({"type": "error", "content": str(e)})
        await websocket.close(code=1013)
        return

    # Notify client auth success
    await safe_send({"type": "auth_ok"})

    # Pre-warm emotion cache for this session
    try:
        await emotion_engine.prewarm(resolved_chat_id)
    except Exception:
        pass  # Non-critical, don't block connection
    
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
        await manager.disconnect(connection_id)
    except Exception as e:
        print(f"[WS] Error: {e}")
        await manager.disconnect(connection_id)


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
