import os
import sys
import csv
import time
import requests
import concurrent.futures
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Note: Selenium removed for Railway compatibility (no Chrome browser)

# --- LangChain / RAG Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from docx import Document as DocxDocument
from pptx import Presentation
from openpyxl import load_workbook

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
load_dotenv()

if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("SERPER_API_KEY"):
    print("Warning: Missing API Keys in .env file. Some features may not work.")

DOCS_DIR = "./do"
DB_DIR = "./chroma_db"

llm = ChatOpenAI(
    model_name="openai/gpt-4o-mini",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

# ==========================================
# 2. MODULE: LOCAL RAG (Document Processing)
# ==========================================
def parse_docx(filepath):
    doc = DocxDocument(filepath)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def parse_pptx(filepath):
    prs = Presentation(filepath)
    text = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def parse_excel(filepath):
    wb = load_workbook(filepath, data_only=True)
    text = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for row in ws.iter_rows(values_only=True):
            row_text = [str(cell) for cell in row if cell is not None]
            if row_text:
                text.append(" | ".join(row_text))
    return "\n".join(text)

def load_documents():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"[System] Created {DOCS_DIR}. No documents to load.")
        return []  # Return empty list instead of crashing

    raw_docs = []
    print(f"[RAG] Scanning {DOCS_DIR}...")
    
    for root, dirs, files in os.walk(DOCS_DIR):
        for filename in files:
            filepath = os.path.join(root, filename)
            ext = filename.lower().split('.')[-1]
            try:
                content = ""
                if ext == 'pdf':
                    loader = PDFPlumberLoader(filepath)
                    raw_docs.extend(loader.load())
                    continue
                elif ext == 'docx': content = parse_docx(filepath)
                elif ext == 'pptx': content = parse_pptx(filepath)
                elif ext == 'xlsx': content = parse_excel(filepath)
                elif ext == 'txt': 
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                
                if content:
                    # Clean filename to be used as a "Link"
                    clean_source = filename 
                    raw_docs.append(Document(page_content=content, metadata={"source": clean_source}))
            except Exception as e:
                print(f"  [!] Failed to load {filename}: {e}")
                
    return raw_docs

def setup_rag_system():
    raw_docs = load_documents()
    if not raw_docs:
        print("[RAG] No documents found. Running in Web-Only mode.")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(raw_docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(DB_DIR):
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
    
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return bm25_retriever, chroma_retriever

def retrieve_rag_context(query, bm25, chroma):
    if not bm25 or not chroma:
        return ""
    
    bm25_docs = bm25.invoke(query)
    chroma_docs = chroma.invoke(query)
    
    combined = {doc.page_content: doc for doc in bm25_docs + chroma_docs}
    final_docs = list(combined.values())[:5]
    
    # --- UPDATED: Clean Source Labeling ---
    # We remove "[Local Doc]" brackets so the AI just sees "Source: filename"
    context_text = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in final_docs])
    
    return context_text

# ==========================================
# 3. MODULE: WEB SEARCH (Serper + Selenium)
# ==========================================
# Selenium driver removed for Railway compatibility
# Deep search feature disabled in cloud deployment

def perform_web_search(query, k_val):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": os.getenv("SERPER_API_KEY"), "Content-Type": "application/json"}
    payload = {"q": query, "num": k_val}
    
    web_context_text = ""
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        
        snippets = []
        links = []
        
        for item in data.get("organic", []):
            # --- UPDATED: Explicit Source Labeling ---
            snippets.append(f"Source: {item.get('link')}\nTitle: {item.get('title')}\nSnippet: {item.get('snippet')}")
            links.append(item.get('link'))
            
        web_context_text = "\n\n".join(snippets)
        
        # Deep search disabled for cloud deployment (requires Chrome browser)
        if "deep search" in query.lower():
            print("   [Web] Deep Search requested but disabled in cloud mode")
                
    except Exception as e:
        print(f"   [Web] Error during search: {e}")
        
    return web_context_text

# ==========================================
# 4. MODULE: SYNTHESIS ENGINE (Web Search + LLM)
# ==========================================
def generate_final_response(query, rag_context, web_context):
    """Generate response using web search results + LLM (no RAG for regular chat)"""
    
    # For regular chat, we only use web search results (not RAG)
    # RAG is used in the Library feature (/api/library/chat) with uploaded files
    
    system_prompt = """You are **Relyce AI**, a helpful and knowledgeable AI assistant.

**Your Role:**
You are a friendly, intelligent assistant that helps users with a wide range of questions. You can discuss topics like technology, business, science, general knowledge, and more.

**Response Guidelines:**
1. **Be Helpful:** Provide clear, accurate, and useful answers.
2. **Be Conversational:** For greetings and casual chat, respond naturally and warmly.
3. **Use Web Results:** When web search results are provided, use them to give accurate, up-to-date information.
4. **Cite Sources:** If you use information from web search, mention the source.
5. **Be Honest:** If you don't know something or web search didn't find relevant results, say so politely.
6. **No Hallucination:** Don't make up facts. Base your answers on the provided web results or your general knowledge.

**Formatting:**
- Use markdown for better readability (headers, bullet points, bold text)
- Keep responses concise but informative
- Include source URLs when referencing web content"""

    # Build the user message
    if web_context and web_context.strip():
        user_message = f"""User Question: {query}

Web Search Results:
{web_context}

Please answer the user's question using the web search results above. Cite relevant sources."""
    else:
        # No web results - just answer based on general knowledge
        user_message = f"""User Question: {query}

(No web search results available - please answer based on your knowledge)"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating response: {e}"

# Functions exported for server use:
# - perform_web_search(query, k_val)
# - retrieve_rag_context(query, bm25, chroma)
# - generate_final_response(query, rag_context, web_context)
# - setup_rag_system()
# - setup_driver()