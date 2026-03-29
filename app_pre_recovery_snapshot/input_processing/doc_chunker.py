"""
Document Chunk Reasoning — Multi-pass document analysis for long content.
Splits large documents into chunks, analyzes each in parallel,
then aggregates insights for final synthesis.

Pipeline:
  1. Chunk text (~1200 words per chunk, max 20 chunks)
  2. Parallel per-chunk analysis (fast model)
  3. Aggregate insights
  4. Return structured summary for final LLM synthesis

Used by: web_fetch pipeline and future PDF/document handling.
"""
import asyncio
import os
from typing import List, Dict, Optional

from app.config import MEMORY_EXTRACTION_MODEL


# ============================================
# CONFIGURATION
# ============================================

CHUNK_SIZE_WORDS = 1200     # words per chunk
MAX_CHUNKS = 20             # don't process unlimited chunks
ANALYSIS_TIMEOUT = 5.0      # per-chunk timeout
MAX_PARALLEL = min(5, os.cpu_count() or 2)  # dynamic based on CPU
CHUNK_TRIGGER_TOKENS = 2000  # trigger chunking above this


# ============================================
# CHUNKING
# ============================================

def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS) -> List[str]:
    """Split text into word-based chunks."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks[:MAX_CHUNKS]


# ============================================
# PER-CHUNK ANALYSIS
# ============================================

CHUNK_ANALYSIS_PROMPT = """Analyze this section of a document.

Extract:
- Key ideas and findings
- Technical details
- Notable claims or conclusions

Section:
{chunk}

Respond with a concise bullet-point summary (max 100 words)."""


async def _analyze_chunk(chunk: str, index: int) -> Dict:
    """Analyze a single chunk. Returns structured summary."""
    try:
        from app.llm.router import get_openrouter_client
        client = get_openrouter_client()

        prompt = CHUNK_ANALYSIS_PROMPT.format(chunk=chunk[:5000])

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=MEMORY_EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            ),
            timeout=ANALYSIS_TIMEOUT,
        )

        summary = (response.choices[0].message.content or "").strip()
        return {"index": index, "summary": summary, "success": True}

    except asyncio.TimeoutError:
        return {"index": index, "summary": "", "success": False, "error": "timeout"}
    except Exception as e:
        return {"index": index, "summary": "", "success": False, "error": str(e)[:100]}


# ============================================
# PARALLEL ANALYSIS
# ============================================

async def analyze_document(text: str) -> str:
    """
    Full document chunk reasoning pipeline.
    Returns aggregated insights as a structured string.
    """
    chunks = chunk_text(text)

    if not chunks:
        return ""

    # Single chunk — no need for multi-pass
    if len(chunks) == 1:
        result = await _analyze_chunk(chunks[0], 0)
        return result.get("summary", "")

    # Parallel analysis with concurrency limit
    semaphore = asyncio.Semaphore(MAX_PARALLEL)

    async def _bounded_analyze(chunk, idx):
        async with semaphore:
            return await _analyze_chunk(chunk, idx)

    tasks = [_bounded_analyze(chunk, i) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    # Aggregate successful analyses
    analyses = []
    for r in sorted(results, key=lambda x: x["index"]):
        if r.get("success") and r.get("summary"):
            analyses.append(f"[Section {r['index'] + 1}] {r['summary']}")

    if not analyses:
        return ""

    aggregated = "\n\n".join(analyses)

    print(f"[DocChunker] Analyzed {len(chunks)} chunks → {len(analyses)} successful")
    return aggregated


# ============================================
# SMART CHUNK RETRIEVAL (for Q&A)
# ============================================

def retrieve_relevant_chunks(
    chunks: List[str],
    query: str,
    top_k: int = 5,
) -> List[str]:
    """
    For question-answering: retrieve only relevant chunks instead of all.
    Uses keyword overlap scoring (no embeddings needed).
    """
    query_words = set(query.lower().split())

    scored = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            scored.append((overlap, i, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]]


# ============================================
# CONVENIENCE
# ============================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word."""
    return int(len(text.split()) * 0.75)


def needs_chunking(text: str) -> bool:
    """Check if text is long enough to need chunking (token-based)."""
    return _estimate_tokens(text) > CHUNK_TRIGGER_TOKENS
