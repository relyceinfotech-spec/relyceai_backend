"""
Relyce AI Backend - Unified Server
All endpoints in one file with lazy loading for fast startup
"""
print("🚀 Starting Relyce AI Backend...")
import os
import sys
print("✅ Basic imports OK")

# CRITICAL: Only import FastAPI essentials at top level!
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
print("✅ FastAPI imports OK")

# Setup paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.join(backend_dir, 'bucket'))
sys.path.insert(0, os.path.join(backend_dir, 'chat'))
print(f"✅ Paths configured: {backend_dir}")

# ============================================================================
# APP SETUP - Created immediately for fast health check response
# ============================================================================
app = FastAPI(
    title="Relyce AI Backend",
    version="1.0.0",
    description="Unified API server for Relyce AI"
)
print("✅ FastAPI app created")

app.add_middleware(
    CORSMiddleware,
    # SECURITY: Strict CORS - only allow specific origins
    allow_origins=[
        "https://relyceai-frontend.vercel.app",
        "https://relyceai.com",
        "https://www.relyceai.com",
        "http://localhost:5173",  # Dev only
        "http://localhost:3000",  # Dev only
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# SECURITY: Rate Limiting & Timeout
# ============================================================================
import time
import asyncio
from collections import defaultdict

# Rate limiting configuration (configurable via environment variables)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))  # Max requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))      # Window in seconds
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))          # Max seconds for LLM request

# PRODUCTION: Connection limits (configurable - scale based on server resources)
# For 4GB RAM: ~2000 connections is safe, adjust based on your needs
MAX_CONNECTIONS_PER_USER = int(os.getenv("MAX_CONNECTIONS_PER_USER", "5"))    # Max per user
MAX_TOTAL_CONNECTIONS = int(os.getenv("MAX_TOTAL_CONNECTIONS", "2000"))       # 4GB RAM = ~2000
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "5000"))             # Max chars per message

# In-memory rate limiter (for production, use Redis)
user_request_times = defaultdict(list)
user_warnings = defaultdict(int)  # Track warnings per user
user_connections = defaultdict(set)  # Track active connection IDs per user
total_connections = 0  # Global connection counter

def get_connection_stats():
    """Get current connection statistics"""
    return {
        "total_connections": total_connections,
        "unique_users": len(user_connections),
        "connections_by_user": {uid[:8] + "...": len(conns) for uid, conns in user_connections.items()}
    }

def check_rate_limit(user_id: str) -> tuple[bool, int]:
    """
    Check if user is rate limited.
    Returns: (allowed: bool, remaining: int)
    """
    if not user_id or user_id == "anonymous":
        return True, RATE_LIMIT_REQUESTS  # Don't rate limit anonymous
    
    now = time.time()
    # Clean old entries outside the window
    user_request_times[user_id] = [
        t for t in user_request_times[user_id] 
        if now - t < RATE_LIMIT_WINDOW
    ]
    
    current_count = len(user_request_times[user_id])
    remaining = RATE_LIMIT_REQUESTS - current_count
    
    if current_count >= RATE_LIMIT_REQUESTS:
        user_warnings[user_id] += 1
        return False, 0
    
    user_request_times[user_id].append(now)
    return True, remaining - 1

def get_rate_limit_reset(user_id: str) -> int:
    """Get seconds until rate limit resets"""
    if not user_request_times[user_id]:
        return 0
    oldest = min(user_request_times[user_id])
    reset_time = oldest + RATE_LIMIT_WINDOW - time.time()
    return max(0, int(reset_time))

async def run_with_timeout(coro, timeout: int = REQUEST_TIMEOUT):
    """Run a coroutine with timeout protection"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return None

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
class ChatRequest(BaseModel):
    query: str
    userId: Optional[str] = None
    mode: str = "generic"
    k_val: int = 5

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    mode: str = "generic"
    processing_time: Optional[float] = None
    remaining_requests: Optional[int] = None  # Show remaining quota

class LibraryChatRequest(BaseModel):
    userId: str
    query: str
    mode: str = "general"

class LibraryChatResponse(BaseModel):
    response: str
    status: str = "success"
    mode: str = "general"

# ============================================================================
# LAZY LOADING - Heavy modules only loaded on first real request
# ============================================================================
_modules = {
    "env_loaded": False,
    "firebase_loaded": False,
    "genric": None,
    "business": None,
    "bucket": None,
    "business_bucket": None
}

def _init_env():
    if _modules["env_loaded"]:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(backend_dir, '.env'))
        print("✅ Environment loaded")
    except Exception as e:
        print(f"⚠️ Dotenv: {e}")
    _modules["env_loaded"] = True

def _init_firebase():
    if _modules["firebase_loaded"]:
        return
    _init_env()
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
                print("🔥 Firebase initialized")
            else:
                print("⚠️ FIREBASE_ADMIN_SDK not set")
    except Exception as e:
        print(f"⚠️ Firebase: {e}")
    _modules["firebase_loaded"] = True

def _get_genric():
    if _modules["genric"] is None:
        _init_env()
        import genric
        _modules["genric"] = genric
    return _modules["genric"]

def _get_business():
    if _modules["business"] is None:
        _init_env()
        import business
        _modules["business"] = business
    return _modules["business"]

def _get_bucket():
    if _modules["bucket"] is None:
        _init_env()
        import Bucket
        _modules["bucket"] = Bucket
    return _modules["bucket"]

def _get_business_bucket():
    if _modules["business_bucket"] is None:
        _init_env()
        import Busintss_Bucket
        _modules["business_bucket"] = Busintss_Bucket
    return _modules["business_bucket"]

# ============================================================================
# HEALTH & ROOT - Respond immediately without loading heavy modules
# ============================================================================
@app.get("/")
async def root():
    return {"service": "Relyce AI Backend", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "relyce-backend"}

@app.get("/stats")
async def stats():
    """Get server statistics for monitoring (use behind auth in production)"""
    return {
        "status": "running",
        "connections": get_connection_stats(),
        "limits": {
            "max_connections_per_user": MAX_CONNECTIONS_PER_USER,
            "max_total_connections": MAX_TOTAL_CONNECTIONS,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window_seconds": RATE_LIMIT_WINDOW,
            "max_message_length": MAX_MESSAGE_LENGTH
        }
    }

# ============================================================================
# CHAT ENDPOINTS - Web search powered chat
# ============================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with web search - Generic or Business mode"""
    import time
    import concurrent.futures
    
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    mode = request.mode.lower()
    print(f"\n📨 Chat | Mode: {mode.upper()}")
    print(f"📝 Query: {request.query[:50]}...")
    
    start_time = time.time()
    
    try:
        if mode == "business":
            bm = _get_business()
            response = bm.generate_final_response(request.query, "", "")
        else:
            gm = _get_genric()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_web = executor.submit(gm.perform_web_search, request.query, request.k_val)
                web_text = future_web.result()
            response = gm.generate_final_response(request.query, "", web_text)
        
        processing_time = time.time() - start_time
        print(f"✅ Response in {processing_time:.2f}s")
        
        return ChatResponse(response=response, status="success", mode=mode, processing_time=processing_time)
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WEBSOCKET CHAT - Real-time streaming with security
# ============================================================================
active_connections: dict = {}

# Security: Max message length to prevent DoS
MAX_MESSAGE_LENGTH = 10000

def sanitize_input(text: str) -> str:
    """Sanitize user input - remove control characters, limit length"""
    if not isinstance(text, str):
        return ""
    # Remove control characters except newline/tab
    sanitized = ''.join(c for c in text if c >= ' ' or c in '\n\t')
    # Limit length
    return sanitized[:MAX_MESSAGE_LENGTH]

def validate_user_id(user_id: str) -> bool:
    """Validate userId format (Firebase UID is 28 chars)"""
    if not isinstance(user_id, str):
        return False
    if len(user_id) < 10 or len(user_id) > 128:
        return False
    # Only allow alphanumeric and common UID characters
    return all(c.isalnum() or c in '-_' for c in user_id)

def validate_session_id(session_id: str) -> bool:
    """Validate sessionId format"""
    if not isinstance(session_id, str):
        return False
    if len(session_id) < 5 or len(session_id) > 128:
        return False
    return all(c.isalnum() or c in '-_' for c in session_id)

def verify_firebase_token(token: str) -> str:
    """
    Verify Firebase ID token and extract UID
    Returns: verified UID or None if invalid
    SECURITY: This is the ONLY way to get trusted user identity
    """
    if not token or not isinstance(token, str):
        return None
    
    try:
        _init_firebase()
        from firebase_admin import auth
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token.get('uid')
        if uid and validate_user_id(uid):
            return uid
        return None
    except Exception as e:
        print(f"⚠️ Token verification failed: {e}")
        return None

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with token-based auth"""
    import concurrent.futures
    global total_connections
    
    await websocket.accept()
    connection_id = id(websocket)  # Unique ID for this connection
    
    # SECURITY: Get token and session from query params
    # Token is verified to extract real UID - never trust userId from client
    token = websocket.query_params.get("token", "")
    session_id = websocket.query_params.get("sessionId", "default")
    
    # SECURITY: Verify token and extract real UID
    user_id = verify_firebase_token(token) if token else None
    
    # If no valid token, allow anonymous (limited features)
    if not user_id:
        user_id = "anonymous"
        print(f"🔗 WebSocket: anonymous user, session={session_id[:8]}...")
    else:
        print(f"🔗 WebSocket: verified user={user_id[:8]}..., session={session_id[:8]}...")
    
    # PRODUCTION: Check global connection limit
    if total_connections >= MAX_TOTAL_CONNECTIONS:
        await websocket.send_json({"type": "error", "text": "Server is at capacity. Please try again later."})
        await websocket.close()
        return
    
    # PRODUCTION: Check per-user connection limit
    if len(user_connections[user_id]) >= MAX_CONNECTIONS_PER_USER:
        await websocket.send_json({"type": "error", "text": "Too many active connections. Please close other tabs."})
        await websocket.close()
        return
    
    # Validate session ID
    if not validate_session_id(session_id) and session_id != "default":
        await websocket.send_json({"type": "error", "text": "Invalid session ID"})
        await websocket.close()
        return
    
    # SECURITY: Lock this connection to the verified user/session
    # UID came from verified token - can't be faked
    locked_user_id = user_id
    locked_session_id = session_id
    
    # Track this connection - use connection_id as key to allow multiple devices
    user_connections[user_id].add(connection_id)
    total_connections += 1
    active_connections[connection_id] = websocket  # Use connection_id, not session_id!
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # SECURITY: Validate message structure
            if not isinstance(data, dict):
                await websocket.send_json({"type": "error", "text": "Invalid request"})
                continue
            
            # SECURITY: Check if client is trying to impersonate different user
            msg_user_id = data.get("userId", "")
            if msg_user_id and msg_user_id != locked_user_id:
                await websocket.send_json({"type": "error", "text": "Session mismatch"})
                continue
            
            message = sanitize_input(data.get("message", ""))
            chat_mode = data.get("chatMode", "standard")
            is_web_search = data.get("isWebSearch", False)
            
            # Validate chat_mode
            if chat_mode not in ["standard", "plus"]:
                chat_mode = "standard"
            
            if message == "STOP_PROCESSING":
                await websocket.send_json({"type": "stopped"})
                continue
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            # SECURITY: Reject empty messages
            if not message.strip():
                continue
            
            # PRODUCTION: Reject overly long messages
            if len(message) > MAX_MESSAGE_LENGTH:
                await websocket.send_json({
                    "type": "error", 
                    "text": f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed."
                })
                continue
            
            # SECURITY: Rate limiting check
            allowed, remaining = check_rate_limit(locked_user_id)
            if not allowed:
                reset_time = get_rate_limit_reset(locked_user_id)
                await websocket.send_json({
                    "type": "error",
                    "text": f"Rate limit exceeded. Try again in {reset_time} seconds.",
                    "rateLimited": True,
                    "resetIn": reset_time
                })
                continue
            
            # Log mode info (truncated for privacy)
            mode_label = "Pro" if is_web_search else "Lite"
            print(f"📨 WS [{mode_label}]: {message[:30]}...")
            
            try:
                print(f"   🔄 [Server] Processing chat_mode={chat_mode}, is_web_search={is_web_search}")
                
                # SMART QUERY CLASSIFICATION - Determine if web search is needed
                def needs_web_search(query: str) -> bool:
                    """
                    Classify if a query needs real-time web search or can be answered by LLM directly.
                    Returns: True if web search needed, False for direct LLM response
                    """
                    query_lower = query.lower().strip()
                    
                    # Patterns that DON'T need web search (direct LLM)
                    no_search_patterns = [
                        # Greetings and casual
                        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
                        'how are you', 'what\'s up', 'thanks', 'thank you', 'bye', 'goodbye',
                        'ok', 'okay', 'sure', 'yes', 'no', 'please', 'help',
                        # Simple questions about the AI
                        'who are you', 'what are you', 'what can you do', 'your name',
                        # General knowledge (LLM already knows)
                        'what is', 'explain', 'define', 'meaning of', 'how does', 'why is',
                        'tell me about', 'describe', 'difference between',
                        # Creative/conversational
                        'write a', 'create a', 'make a', 'generate', 'compose',
                        'joke', 'story', 'poem', 'code', 'script', 'function',
                        # Opinions and advice (no real-time data needed)
                        'should i', 'what do you think', 'recommend', 'suggest', 'advice',
                    ]
                    
                    # Patterns that NEED web search (real-time/factual data)
                    search_patterns = [
                        # Current events/news
                        'latest', 'recent', 'news', 'today', 'yesterday', 'this week',
                        'current', 'now', '2024', '2025', '2026',
                        # Real-time data
                        'price', 'stock', 'weather', 'score', 'result', 'live',
                        # Specific lookups
                        'who won', 'how much', 'when did', 'where is', 'statistics',
                        # Research topics
                        'research', 'study', 'report', 'data', 'statistics', 'trends',
                        # Comparisons requiring current info
                        'best', 'top', 'compare', 'vs', 'versus', 'review',
                    ]
                    
                    # Check if it's a simple greeting/short query
                    if len(query_lower.split()) <= 3:
                        for pattern in no_search_patterns[:15]:  # Check greetings first
                            if pattern in query_lower:
                                return False
                    
                    # Check if query explicitly needs search
                    for pattern in search_patterns:
                        if pattern in query_lower:
                            return True
                    
                    # Check if query is conversational/creative (no search needed)
                    for pattern in no_search_patterns:
                        if query_lower.startswith(pattern) or pattern in query_lower:
                            # Exception: if query also has search patterns, still search
                            has_search_term = any(sp in query_lower for sp in search_patterns)
                            if not has_search_term:
                                return False
                    
                    # Default: do web search for longer, complex queries
                    return len(query_lower.split()) > 5
                
                # Determine if we need web search
                should_search = needs_web_search(message) or is_web_search  # Force search if Pro mode
                
                gm = _get_genric()
                message_id = data.get("messageId")
                chat_id = data.get("chatId")
                web_text = ""
                sources = []
                k_results = 10 if is_web_search else 5
                
                if should_search:
                    # Send "searching" status to frontend so it can show skeleton loader
                    await websocket.send_json({
                        "type": "searching",
                        "messageId": message_id,
                        "chatId": chat_id,
                        "query": message[:100],
                        "mode": mode_label
                    })
                    
                    # Perform web search (in thread pool to not block)
                    loop = asyncio.get_event_loop()
                    web_text, sources = await loop.run_in_executor(
                        None, 
                        lambda: gm.perform_web_search(message, k_results)
                    )
                    print(f"   🔄 [Server] Web search done, got {len(web_text) if web_text else 0} chars, {len(sources)} sources")
                    
                    # Send sources to frontend for display
                    await websocket.send_json({
                        "type": "sources",
                        "messageId": message_id,
                        "chatId": chat_id,
                        "sources": sources,
                        "mode": mode_label
                    })
                else:
                    # Skip web search - direct LLM response
                    print(f"   ⚡ [Server] Smart skip - no web search needed for: {message[:30]}...")
                
                # Get the streaming generator based on mode
                if chat_mode == "plus":
                    bm = _get_business()
                    stream_generator = bm.generate_streaming_response(message, "", web_text)
                else:
                    stream_generator = gm.generate_streaming_response(message, "", web_text)
                
                # Stream response chunks to client
                full_response = ""
                
                # Send initial streaming start signal
                await websocket.send_json({
                    "type": "stream_start",
                    "messageId": message_id,
                    "chatId": chat_id,
                    "mode": mode_label,
                    "sources": sources  # Include sources in stream_start too
                })
                
                # Stream each chunk
                for chunk in stream_generator:
                    if chunk:
                        full_response += chunk
                        await websocket.send_json({
                            "type": "stream",
                            "text": chunk,
                            "messageId": message_id,
                            "chatId": chat_id
                        })
                
                # Send stream end signal with full response
                await websocket.send_json({
                    "type": "stream_end",
                    "text": full_response,
                    "messageId": message_id,
                    "chatId": chat_id,
                    "mode": mode_label,
                    "sources": sources,
                    "remaining": remaining
                })
                print(f"   ✅ [Server] Streaming complete ({len(full_response)} chars)")
                
            except Exception as e:
                # SECURITY: Don't expose internal error details
                print(f"❌ Chat error: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"type": "error", "text": "Something went wrong. Please try again."})
                
    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_id[:8]}...")
        # Clean up connection tracking - use connection_id as key
        if connection_id in active_connections:
            del active_connections[connection_id]
        if connection_id in user_connections.get(locked_user_id, set()):
            user_connections[locked_user_id].discard(connection_id)
            if not user_connections[locked_user_id]:
                del user_connections[locked_user_id]
        total_connections = max(0, total_connections - 1)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        # Clean up on any error - use connection_id as key
        if connection_id in active_connections:
            del active_connections[connection_id]
        user_connections.get(locked_user_id, set()).discard(connection_id)
        total_connections = max(0, total_connections - 1)

# ============================================================================
# LIBRARY ENDPOINTS - Document Q&A with RAG
# ============================================================================
def _load_document(filepath):
    """Load a single document and return its content"""
    ext = filepath.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            from langchain_community.document_loaders import PDFPlumberLoader
            loader = PDFPlumberLoader(filepath)
            return "\n\n".join([doc.page_content for doc in loader.load()])
        elif ext == 'docx':
            return _get_bucket().parse_docx(filepath)
        elif ext == 'pptx':
            return _get_bucket().parse_pptx(filepath)
        elif ext == 'xlsx':
            return _get_bucket().parse_excel(filepath)
        elif ext == 'csv':
            return _get_bucket().parse_csv(filepath)
        elif ext in ['html', 'htm']:
            return _get_bucket().parse_html(filepath)
        elif ext in ['txt', 'md', 'py']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

def _download_user_files(user_id: str, temp_dir: str) -> list:
    """Download user's library files from Firebase Storage"""
    _init_firebase()
    downloaded = []
    try:
        from firebase_admin import storage as fb_storage
        bucket = fb_storage.bucket()
        blobs = bucket.list_blobs(prefix=f"library/users/{user_id}/")
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            filename = os.path.basename(blob.name)
            local_path = os.path.join(temp_dir, filename)
            blob.download_to_filename(local_path)
            downloaded.append(local_path)
            print(f"  📄 {filename}")
    except Exception as e:
        print(f"❌ Download error: {e}")
    return downloaded

@app.post("/api/library/chat", response_model=LibraryChatResponse)
async def library_chat(request: LibraryChatRequest):
    """Chat with library documents - General or Business mode"""
    import tempfile
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    
    if not request.userId or not request.query:
        raise HTTPException(status_code=400, detail="userId and query required")
    
    mode = request.mode.lower() if request.mode.lower() in ["general", "business"] else "general"
    print(f"\n📨 Library | User: {request.userId} | Mode: {mode.upper()}")
    print(f"📝 Query: {request.query[:50]}...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = _download_user_files(request.userId, temp_dir)
            
            if not file_paths:
                return LibraryChatResponse(response="No documents in library. Please upload files first.", status="no_files", mode=mode)
            
            # Process documents
            raw_docs = []
            for filepath in file_paths:
                content = _load_document(filepath)
                if content:
                    raw_docs.append(Document(page_content=content, metadata={"source": os.path.basename(filepath)}))
            
            if not raw_docs:
                return LibraryChatResponse(response="Could not read documents.", status="read_error", mode=mode)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
            splits = text_splitter.split_documents(raw_docs)
            
            # Create vectorstore and chain
            bucket = _get_bucket()
            vectorstore = bucket.setup_vectorstore(splits)
            
            if mode == "general":
                chain = bucket.create_advanced_rag_chain(vectorstore, splits)
            else:
                chain = _get_business_bucket().create_advanced_rag_chain(vectorstore, splits)
            
            response = chain.invoke({"input": request.query})
            print(f"✅ Library response generated")
            
            return LibraryChatResponse(response=response, status="success", mode=mode)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Relyce AI Backend on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
