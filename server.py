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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# WEBSOCKET CHAT - Real-time streaming
# ============================================================================
active_connections: dict = {}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    import concurrent.futures
    
    await websocket.accept()
    session_id = websocket.query_params.get("sessionId", "default")
    user_id = websocket.query_params.get("userId", "anonymous")
    active_connections[session_id] = websocket
    print(f"🔗 WebSocket: user={user_id}, session={session_id}")
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            chat_mode = data.get("chatMode", "standard")
            is_web_search = data.get("isWebSearch", False)  # Pro mode flag
            
            if message == "STOP_PROCESSING":
                await websocket.send_json({"type": "stopped"})
                continue
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            # Log mode info
            mode_label = "Pro" if is_web_search else "Lite"
            print(f"📨 WS [{mode_label}]: {message[:50]}...")
            
            try:
                if chat_mode == "plus":
                    # Business mode
                    bm = _get_business()
                    response = bm.generate_final_response(message, "", "")
                else:
                    # Generic mode - Lite or Pro based on isWebSearch
                    gm = _get_genric()
                    
                    # Pro mode: More search results (k=10), Lite mode: Standard (k=5)
                    k_results = 10 if is_web_search else 5
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_web = executor.submit(gm.perform_web_search, message, k_results)
                        web_text = future_web.result()
                    
                    response = gm.generate_final_response(message, "", web_text)
                
                await websocket.send_json({
                    "type": "bot",
                    "text": response,
                    "messageId": data.get("messageId"),
                    "chatId": data.get("chatId"),
                    "mode": mode_label
                })
            except Exception as e:
                print(f"❌ Chat error: {e}")
                await websocket.send_json({"type": "error", "text": str(e)})
                
    except WebSocketDisconnect:
        print(f"🔌 Disconnected: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]

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
