"""
Citation Engine — Source attribution for AI responses.
Tags response content with source references.

Sources tracked:
  - vector_memory: [Memory]
  - knowledge_graph: [Graph]
  - web_content: [Web: URL]
  - session_summary: [Session]
  - general_knowledge: (no citation)

Usage:
    tracker = CitationTracker()
    tracker.add("vector", "FastAPI supports async", "memory_id_123")
    tracker.add("web", "FastAPI docs...", "https://fastapi.tiangolo.com")
    footnotes = tracker.get_footnotes()
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Citation:
    """A single source reference."""
    source_type: str       # "vector", "graph", "web", "session"
    content_preview: str   # first 80 chars of source content
    reference: str         # memory ID, URL, or label
    used: bool = False     # whether cited in final response


class CitationTracker:
    """
    Track all sources used during context assembly.
    Generates footnotes for the response.
    """
    def __init__(self):
        self.citations: List[Citation] = []

    def add(
        self,
        source_type: str,
        content: str,
        reference: str = "",
    ):
        """Register a source used in context."""
        self.citations.append(Citation(
            source_type=source_type,
            content_preview=content[:80].strip(),
            reference=reference,
        ))

    def add_vector_memories(self, memories: list):
        """Register vector memory sources."""
        for i, mem in enumerate(memories or []):
            text = getattr(mem, 'text', '') if hasattr(mem, 'text') else str(mem)
            self.add("vector", text, f"memory_{i+1}")

    def add_graph_triples(self, triples: list):
        """Register knowledge graph sources."""
        for i, triple in enumerate(triples or []):
            if hasattr(triple, 'subject'):
                text = f"{triple.subject} {triple.relation} {triple.object}"
            else:
                text = str(triple)
            self.add("graph", text, f"graph_{i+1}")

    def add_web_content(self, url: str, title: str = ""):
        """Register web content source."""
        label = title or url
        self.add("web", label, url)

    def add_serper_results(self, results: any):
        """Register Serper search results with batch resilience."""
        organic = []
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and "organic" in first:
                organic = first["organic"]
            else:
                organic = results # Fallback to flat list
        elif isinstance(results, dict):
            organic = results.get("organic", [])

        for i, res in enumerate(organic or []):
            if isinstance(res, dict):
                title = res.get("title", f"Search Result {i+1}")
                url = res.get("link", res.get("url", ""))
                self.add("web", title, url)

    def add_rag_results(self, context: str):
        """Register RAG document context."""
        if not context:
            return
        # RAG context is often combined, we'll add it as a single block for now
        # but could be split by "File: ..." if retrieve_rag_context formats it that way
        self.add("doc", context, "uploaded_docs")

    def get_footnotes(self) -> str:
        """
        Generate footnote block for the response.
        Only includes sources that were actually used.
        """
        if not self.citations:
            return ""

        lines = ["\n---\n**Sources:**"]
        seen = set()

        for i, c in enumerate(self.citations, 1):
            # Deduplicate
            key = f"{c.source_type}:{c.reference}"
            if key in seen:
                continue
            seen.add(key)

            label = _format_label(c)
            lines.append(f"  [{i}] {label}")

        if len(lines) <= 1:
            return ""

        return "\n".join(lines)

    def get_inline_tags(self) -> Dict[str, str]:
        """
        Get source tags for inline citation.
        Maps content previews to citation markers.
        """
        tags = {}
        for i, c in enumerate(self.citations, 1):
            tag = _format_tag(c, i)
            tags[c.content_preview] = tag
        return tags

    def get_metadata(self) -> Dict:
        """Return citation metadata for trace logging."""
        return {
            "total_sources": len(self.citations),
            "source_types": list(set(c.source_type for c in self.citations)),
            "web_urls": [c.reference for c in self.citations if c.source_type == "web"],
        }


# ============================================
# FORMATTING
# ============================================

_SOURCE_LABELS = {
    "vector": "Memory",
    "graph": "Knowledge Graph",
    "web": "Web",
    "session": "Session Context",
    "doc": "Document",
}


def _format_label(c: Citation) -> str:
    """Format a citation for the footnotes block."""
    source_name = _SOURCE_LABELS.get(c.source_type, c.source_type)

    if c.source_type == "web" and c.reference:
        return f"{source_name}: {c.reference}"
    elif c.reference:
        return f"{source_name}: {c.content_preview}"
    else:
        return f"{source_name}: {c.content_preview}"


def _format_tag(c: Citation, index: int) -> str:
    """Format inline citation tag."""
    source_name = _SOURCE_LABELS.get(c.source_type, c.source_type)
    return f"[{source_name}]"


# ============================================
# SYSTEM PROMPT INSTRUCTION
# ============================================

CITATION_INSTRUCTION = (
    "\n\nWhen your answer uses information from the provided context sources, "
    "add a brief source tag like [Memory] or [Web] or [Graph] after the relevant claim. "
    "Only cite when information directly comes from a provided source. "
    "Do not cite general knowledge.\n"
)

# ============================================
# RETRIEVAL QUALITY SCORING
# ============================================

async def score_retrievals(user_id: str, retrieved_meta: dict, full_response: str):
    """
    Retrieval Quality Scoring. Track whether retrieved context actually influenced answers.
    Bumps importance if cited, penalizes if ignored.
    This prevents memory pollution over time.
    """
    has_vector = "[Memory]" in full_response 
    has_graph = "[Graph]" in full_response or "[Knowledge Graph]" in full_response
    has_doc = "[Document]" in full_response

    # Vector Memory Scoring
    if retrieved_meta.get("vector"):
        try:
            from app.memory.vector_memory import bump_memory_scores
            await bump_memory_scores(user_id, retrieved_meta["vector"], positive=has_vector)
        except Exception as e:
            print(f"[CitationEngine] Error scoring vector memory: {e}")

    # Knowledge Graph Scoring
    if retrieved_meta.get("graph"):
        try:
            from app.memory.knowledge_graph import bump_graph_scores
            await bump_graph_scores(user_id, retrieved_meta["graph"], positive=has_graph)
        except Exception as e:
            print(f"[CitationEngine] Error scoring graph memory: {e}")
