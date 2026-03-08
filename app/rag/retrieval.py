"""
RAG Retrieval Module
Handles semantic search across indexed documents in Weaviate.
"""
import asyncio
from typing import List, Dict, Any
from app.memory.weaviate_client import get_weaviate_client, generate_embedding, COLLECTION_NAME as MEMORY_COLLECTION
from app.rag.indexing import RAG_COLLECTION_NAME
from app.safety.content_policy import sanitize_rag_text, has_prompt_injection_markers


async def retrieve_rag_context(user_id: str, query: str, session_id: str, filename: str = None, top_k: int = 5) -> str:
    """
    Retrieve relevant chunks from Weaviate DocumentRAG collection.
    If filename is provided, filters results to that specific document.
    """
    try:
        client = await get_weaviate_client()
        if not client:
            return ""

        query_embedding = await generate_embedding(query)
        if not query_embedding:
            return ""

        collection = client.collections.get(RAG_COLLECTION_NAME)

        # Build filter
        from weaviate.classes.query import Filter
        filters = Filter.by_property("session_id").equal(session_id)
        if filename:
            filters = filters & Filter.by_property("filename").equal(filename)

        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            filters=filters,
            return_metadata=["distance"]
        )

        if not results.objects:
            return ""

        # Format context (sanitized to reduce RAG prompt-injection risk)
        context_parts = []
        for obj in results.objects:
            text = obj.properties.get("text", "")
            source = obj.properties.get("filename", "unknown")
            if has_prompt_injection_markers(text):
                print(f"[RAG] Prompt-injection markers sanitized in '{source}'")
            safe_text = sanitize_rag_text(text)
            context_parts.append(f"--- Document: {source} ---\n{safe_text}")

        return "\n\n".join(context_parts)

    except Exception as e:
        print(f"[RAG] Retrieval failed: {e}")
        return ""


async def search_all_documents(user_id: str, query: str, session_id: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Returns raw search results spanning multiple documents in the same session.
    """
    try:
        client = await get_weaviate_client()
        if not client:
            return []

        query_embedding = await generate_embedding(query)
        if not query_embedding:
            return []

        collection = client.collections.get(RAG_COLLECTION_NAME)

        from weaviate.classes.query import Filter
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            filters=Filter.by_property("session_id").equal(session_id),
            return_metadata=["distance"]
        )

        return [
            {
                "text": sanitize_rag_text(obj.properties.get("text") or ""),
                "filename": obj.properties.get("filename"),
                "score": 1.0 - (obj.metadata.distance or 1.0)
            }
            for obj in results.objects
        ]
    except Exception as e:
        print(f"[RAG] All-doc search failed: {e}")
        return []
