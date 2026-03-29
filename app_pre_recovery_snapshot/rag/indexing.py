"""
RAG Indexing Module
Handles document extraction, chunking, and vector indexing into Weaviate.
"""
import os
import asyncio
import fitz  # PyMuPDF
import zipfile
import xml.etree.ElementTree as ET
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.memory.weaviate_client import get_weaviate_client, generate_embedding

# Collection for personal/shared document RAG
RAG_COLLECTION_NAME = "DocumentRAG"

# Text splitter for chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=125,
)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    text_parts = []
    try:
        pdf = fitz.open(file_path)
        for page in pdf:
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)
        pdf.close()
    except Exception as e:
        print(f"[RAG] PDF extraction failed: {e}")
    return "\n\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using zip + XML parsing."""
    try:
        with zipfile.ZipFile(file_path) as zf:
            with zf.open('word/document.xml') as doc_xml:
                xml_bytes = doc_xml.read()
        root = ET.fromstring(xml_bytes)
        texts = []
        for node in root.iter():
            if node.tag.endswith('}t') and node.text:
                texts.append(node.text)
        return "\n".join(texts)
    except Exception as e:
        print(f"[RAG] DOCX extraction failed: {e}")
        return ""
def extract_text_from_file(file_path: str) -> str:
    """Extract text from a non-PDF file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[RAG] File extraction failed: {e}")
        return ""

async def process_document_for_rag(user_id: str, file_path: str, original_name: str, session_id: str = "general") -> int:
    """
    Extract, chunk, and index a document into Weaviate.
    Returns the number of chunks created and indexed.
    """
    ext = os.path.splitext(original_name)[1].lower()
    
    # 1. Extract Text
    if ext == ".pdf":
        text = await asyncio.to_thread(extract_text_from_pdf, file_path)
    elif ext == ".docx":
        text = await asyncio.to_thread(extract_text_from_docx, file_path)
    else:
        text = await asyncio.to_thread(extract_text_from_file, file_path)
        
    if not text.strip():
        print(f"[RAG] No text extracted from {original_name}")
        return 0
        
    # 2. Chunking
    documents = [Document(page_content=text, metadata={"source": original_name, "user_id": user_id, "session_id": session_id})]
    chunks = text_splitter.split_documents(documents)
    
    if not chunks:
        return 0
        
    # 3. Vectorization & Storage
    try:
        client = await get_weaviate_client()
        if not client:
            print("[RAG] Weaviate client not available")
            return 0
            
        collection = client.collections.get(RAG_COLLECTION_NAME)
        
        # Process chunks in small batches to avoid timeout
        count = 0
        for chunk in chunks:
            embedding = await generate_embedding(chunk.page_content)
            if not embedding:
                continue
                
            collection.data.insert(
                properties={
                    "text": chunk.page_content,
                    "filename": original_name,
                    "user_id": user_id,
                    "session_id": session_id,
                    "source": original_name,
                },
                vector=embedding
            )
            count += 1
            
        print(f"[RAG] ✅ Indexed '{original_name}' — {count} chunks for user {user_id} in session {session_id}")
        return count
        
    except Exception as e:
        print(f"[RAG] Indexing failed for {original_name}: {e}")
        return 0
