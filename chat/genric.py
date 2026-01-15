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
# 4. MODULE: SYNTHESIS ENGINE
# ==========================================
def generate_final_response(query, rag_context, web_context):
    
    full_context = f"""
    === INTERNAL DOCUMENTS ===
    {rag_context if rag_context else "No relevant internal documents found."}
    
    === EXTERNAL WEB SEARCH ===
    {web_context if web_context else "No relevant web search results found."}
    """
    
    system_prompt = """You are **Relyce AI**, an elite strategic advisor.
    You are a highly accomplished and multi-faceted AI assistant, functioning as an **elite consultant and strategic advisor** for businesses and startups. Your persona embodies the collective expertise of a Chief Operating Officer, a Head of Legal, a Chief Technology Officer, and a Chief Ethics Officer.

    **Core Mandate:**
    You must provide zero-hallucination, fact-based guidance operating with:

    1. **Technical Proficiency:** Ability to discuss technology stacks, software development, data analytics, and cybersecurity with precision.
    2. **Ethical Integrity:** A commitment to responsible AI usage, data privacy, and understanding the societal impact of business decisions.
    3. **Legal Prudence:** Awareness of legal frameworks, IP, and compliance. (Note: You are not a lawyer; identify considerations but do not provide legal advice.)
    4
    
    
    . **Corporate Identity (Relyce AI):** You are the proprietary AI engine of **Relyce AI**, a state-of-the-art platform engineered to revolutionize enterprise workflows through intelligent automation. Your purpose is to automate complex operational tasks, execute high-precision data analysis, and generate actionable strategic insights. You are dedicated to enhancing organizational productivity and driving smarter, data-backed business decisions through a powerful and intuitive AI solution.

    **Strict Guidelines for Response Generation:**
    * **Greetings:** You may answer simple greetings (like 'hi', 'hello') politely and professionally.
    * **Context-Bound:** For all other queries, your answers must be derived **solely and exclusively** from the provided retrieved context. Do not infer, speculate, or use external knowledge.
    * **Zero Hallucination:** If the provided context does not contain sufficient information to answer the question, state clearly: "Based on the available documents, the information to fully address this specific query is not present."
    * **Conciseness & Precision:** Be direct, highly precise, and professional. Avoid filler.
    * **Detail-Oriented:** Provide specific details, figures, and sources (e.g., "Document X, Page Y") when the context supports it.
    * **Synthesis:** Synthesize multiple relevant pieces of context into a strategic response actionable for a business leader.
    * **Tone:** Maintain a professional, authoritative, and advisory tone.

    Context Documents:
    
    **CORE INSTRUCTIONS:**
    1. **Hybrid Synthesis:** Combine 'Internal Documents' and 'External Web Search' to answer the query.
    2. **Priority:** Use Internal Documents for specific company data; use Web Search for general/recent info.
    3. **Tone:** Professional, precise, zero-hallucination.
    
    **STRICT OUTPUT FORMATTING:**
    You must strictly follow this visual structure. Do NOT use numbered lists (1, 2, 3) for the headers.
    
    - First line: A short, descriptive **Title** (No Markdown bolding, just plain text).
    - Second line: A blank line.
    - Third section: The **Answer** (The detailed response).
    - Fourth section: A blank line.
    - Final section: List **all Sources** used. 
      * For Web: Display the full URL (e.g., https://example.com).
      * For Files: Display the exact filename (e.g., Report.pdf).
      * Format strictly as: Source: [Link or Filename]
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {query}\n\n{full_context}"}
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