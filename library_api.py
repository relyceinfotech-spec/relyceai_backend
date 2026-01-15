"""
Library Chat API - FastAPI endpoint for document Q&A with General/Business modes
Imports RAG functionality from Bucket.py (General) and Busintss_Bucket.py (Business)
"""
import os
import sys
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, storage as fb_storage
import json

# Initialize Firebase Admin (only once)
if not firebase_admin._apps:
    try:
        storage_bucket = os.getenv('FIREBASE_STORAGE_BUCKET')
        
        # 🔐 Option 1: Use JSON string from env (secure - no file needed)
        firebase_sdk_json = os.getenv('FIREBASE_ADMIN_SDK')
        
        if firebase_sdk_json:
            service_account = json.loads(firebase_sdk_json)
            # Fix newline issue in private_key
            if 'private_key' in service_account:
                service_account['private_key'] = service_account['private_key'].replace('\\n', '\n')
            cred = credentials.Certificate(service_account)
            firebase_admin.initialize_app(cred, {'storageBucket': storage_bucket})
            print("🔥 Firebase Admin initialized from ENV")
        else:
            # 🔐 Option 2: Fallback to file path (legacy)
            service_account_path = os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH')
            if service_account_path and os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred, {'storageBucket': storage_bucket})
                print("✅ Firebase Admin initialized from FILE")
            else:
                print("⚠️ Set FIREBASE_ADMIN_SDK or FIREBASE_SERVICE_ACCOUNT_PATH in .env")
    except Exception as e:
        print(f"❌ Firebase init failed: {e}")

# Import RAG functions from existing files
from Bucket import (
    parse_docx, parse_pptx, parse_excel, parse_csv, parse_html,
    setup_vectorstore, hybrid_retrieval_func, create_advanced_rag_chain as create_general_chain
)
from Busintss_Bucket import (
    create_advanced_rag_chain as create_business_chain
)

# Import document loaders
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

app = FastAPI(title="Library Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    userId: str
    query: str
    mode: str = "general"

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    mode: str = "general"

def load_document(filepath):
    """Load a single document and return its content"""
    ext = filepath.lower().split('.')[-1]
    try:
        if ext == 'pdf':
            loader = PDFPlumberLoader(filepath)
            docs = loader.load()
            return "\n\n".join([doc.page_content for doc in docs])
        elif ext == 'docx':
            return parse_docx(filepath)
        elif ext == 'pptx':
            return parse_pptx(filepath)
        elif ext == 'xlsx':
            return parse_excel(filepath)
        elif ext == 'csv':
            return parse_csv(filepath)
        elif ext in ['html', 'htm']:
            return parse_html(filepath)
        elif ext in ['txt', 'md', 'py']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

def download_user_files(user_id: str, temp_dir: str) -> list:
    """Download user's library files from Firebase Storage"""
    downloaded = []
    try:
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

def process_documents(file_paths: list):
    """Process documents and create splits"""
    raw_documents = []
    for filepath in file_paths:
        content = load_document(filepath)
        if content:
            doc = Document(page_content=content, metadata={"source": os.path.basename(filepath)})
            raw_documents.append(doc)
    
    if not raw_documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(raw_documents)

@app.post("/library/chat", response_model=ChatResponse)
async def library_chat(request: ChatRequest):
    """Chat with library documents - routes to General or Business mode"""
    
    if not request.userId or not request.query:
        raise HTTPException(status_code=400, detail="userId and query required")
    
    mode = request.mode.lower() if request.mode.lower() in ["general", "business"] else "general"
    
    print(f"\n📨 User: {request.userId} | Mode: {mode.upper()}")
    print(f"📝 Query: {request.query[:50]}...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Firebase
            file_paths = download_user_files(request.userId, temp_dir)
            
            if not file_paths:
                return ChatResponse(response="No documents in library. Please upload files first.", status="no_files", mode=mode)
            
            # Process documents
            splits = process_documents(file_paths)
            if not splits:
                return ChatResponse(response="Could not read documents.", status="read_error", mode=mode)
            
            # Create vectorstore and chain based on mode
            vectorstore = setup_vectorstore(splits)
            
            if mode == "general":
                chain = create_general_chain(vectorstore, splits)
            else:
                chain = create_business_chain(vectorstore, splits)
            
            # Get response
            response = chain.invoke({"input": request.query})
            print(f"✅ Response generated")
            
            return ChatResponse(response=response, status="success", mode=mode)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"service": "Library Chat API", "modes": ["general", "business"]}

# Export app for mounting in unified server
library_app = app
