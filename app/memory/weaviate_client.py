"""
Weaviate Cloud Client — Singleton connector for vector memory.
Uses the existing Weaviate Cloud instance for semantic memory storage.
Embeddings generated via OpenRouter text-embedding-3-small.
"""
import asyncio
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.config import WEAVIATE_URL, WEAVIATE_API_KEY

# ============================================
# SINGLETON CLIENT
# ============================================

_client: Optional[weaviate.WeaviateClient] = None
_lock = asyncio.Lock()

COLLECTION_NAME = "UserMemory"
RAG_COLLECTION_NAME = "DocumentRAG"



async def get_weaviate_client() -> Optional[weaviate.WeaviateClient]:
    """Get or create a singleton Weaviate Cloud client."""
    global _client
    
    if _client is not None:
        return _client
    
    async with _lock:
        if _client is not None:
            return _client
            
        if not WEAVIATE_URL or not WEAVIATE_API_KEY:
            print("[Weaviate] No WEAVIATE_URL or WEAVIATE_API_KEY configured, vector memory disabled")
            return None
        
        try:
            _client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
                skip_init_checks=True,
            )
            print(f"[Weaviate] Connected to cloud: {WEAVIATE_URL}")
            
            # Ensure collection exists
            await _ensure_collection()
            
            return _client
        except Exception as e:
            print(f"[Weaviate] Connection failed: {e}")
            return None


async def _ensure_collection():
    """Create the UserMemory collection if it doesn't exist."""
    global _client
    if not _client:
        return
    
    try:
        if not _client.collections.exists(COLLECTION_NAME):
            _client.collections.create(
                name=COLLECTION_NAME,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="memory_type", data_type=DataType.TEXT),
                    Property(name="importance", data_type=DataType.NUMBER),
                    Property(name="created_at", data_type=DataType.TEXT),
                    Property(name="last_seen", data_type=DataType.TEXT),
                    Property(name="expires_at", data_type=DataType.TEXT),
                    Property(name="retrieval_count", data_type=DataType.INT),
                    Property(name="content_hash", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),  # We provide our own embeddings
            )
            print(f"[Weaviate] Created collection: {COLLECTION_NAME}")
        else:
            print(f"[Weaviate] Collection '{COLLECTION_NAME}' already exists")

        # --- DocumentRAG Collection ---
        if not _client.collections.exists(RAG_COLLECTION_NAME):
            _client.collections.create(
                name=RAG_COLLECTION_NAME,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="session_id", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
            )
            print(f"[Weaviate] Created collection: {RAG_COLLECTION_NAME}")
        else:
            print(f"[Weaviate] Collection '{RAG_COLLECTION_NAME}' already exists")
    except Exception as e:
        print(f"[Weaviate] Error ensuring collection: {e}")


async def close_weaviate_client():
    """Close the Weaviate client connection."""
    global _client
    if _client:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
        print("[Weaviate] Client closed")


# ============================================
# EMBEDDING HELPER
# ============================================

async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate an embedding vector for the given text using OpenRouter.
    Uses text-embedding-3-small (1536 dimensions).
    """
    try:
        from app.llm.router import get_openrouter_client
        from app.config import EMBEDDING_MODEL
        
        client = get_openrouter_client()
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[Weaviate] Embedding generation failed: {e}")
        return None
