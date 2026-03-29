"""
Retrieval Intelligence Layer — Ranks, deduplicates, and synthesizes
all context sources before prompt injection.

Pipeline:
  1. Collect sources (vector memories, graph triples, web content)
  2. Score each source (relevance × reliability × recency)
  3. Deduplicate (skip if cosine/text overlap > 0.90)
  4. Synthesize into labeled context blocks
  5. Return ordered, clean context

Runs AFTER memory retrieval, BEFORE context packing.
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


# ============================================
# CONFIG
# ============================================

DIVERSITY_THRESHOLD = 0.90   # skip if overlap > this
MAX_VECTOR_INJECT = 5        # max vector memories
MAX_GRAPH_INJECT = 5         # max graph triples


# ============================================
# SOURCE SCORING
# ============================================

@dataclass
class ScoredSource:
    """A unified source with its relevance score."""
    text: str
    source_type: str        # "vector_memory", "knowledge_graph", "web_content"
    relevance: float = 0.0
    reliability: float = 1.0
    recency: float = 1.0
    final_score: float = 0.0

    def compute_score(self):
        self.final_score = (
            self.relevance * 0.5 +
            self.reliability * 0.3 +
            self.recency * 0.2
        )


def _text_overlap(a: str, b: str) -> float:
    """Fast word-overlap similarity between two texts."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


# ============================================
# DEDUPLICATION
# ============================================

def deduplicate_sources(sources: List[ScoredSource]) -> List[ScoredSource]:
    """Remove near-duplicate sources using text overlap."""
    if len(sources) <= 1:
        return sources

    # Sort by score descending (keep highest-scored version)
    sources.sort(key=lambda s: s.final_score, reverse=True)

    kept = []
    for src in sources:
        is_dup = False
        for existing in kept:
            if _text_overlap(src.text, existing.text) > DIVERSITY_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(src)

    removed = len(sources) - len(kept)
    if removed:
        print(f"[RetrievalIntel] Dedup: removed {removed} near-duplicate sources")

    return kept


# ============================================
# MAIN PIPELINE
# ============================================

def process_retrieval(
    vector_memories: Optional[List] = None,
    graph_triples: Optional[List] = None,
    web_content: Optional[str] = None,
    query: str = "",
) -> Dict[str, Any]:
    """
    Full retrieval intelligence pipeline.
    Returns cleaned, ranked, deduplicated sources ready for context packing.

    Returns:
        {
            "vector_memories": [filtered list],
            "graph_triples": [filtered list],
            "web_content": str or None,
            "stats": {"total": N, "kept": M, "removed": R}
        }
    """
    all_sources: List[ScoredSource] = []

    # --- Score vector memories ---
    clean_memories = []
    if vector_memories:
        for mem in vector_memories[:MAX_VECTOR_INJECT]:
            text = getattr(mem, 'text', '') if hasattr(mem, 'text') else str(mem)
            sim = getattr(mem, 'similarity', 0.5) if hasattr(mem, 'similarity') else 0.5
            imp = getattr(mem, 'importance', 0.5) if hasattr(mem, 'importance') else 0.5

            src = ScoredSource(
                text=text,
                source_type="vector_memory",
                relevance=sim,
                reliability=0.8,   # memories are user-derived
                recency=imp,       # importance correlates with recency
            )
            src.compute_score()
            all_sources.append(src)

    # --- Score graph triples ---
    clean_triples = []
    if graph_triples:
        for triple in graph_triples[:MAX_GRAPH_INJECT]:
            subj = getattr(triple, 'subject', '') if hasattr(triple, 'subject') else ''
            obj = getattr(triple, 'object', '') if hasattr(triple, 'object') else ''
            text = f"{subj} → {obj}"

            src = ScoredSource(
                text=text,
                source_type="knowledge_graph",
                relevance=getattr(triple, 'score', 0.5) if hasattr(triple, 'score') else 0.5,
                reliability=0.9,   # graph triples are structured
                recency=0.7,
            )
            src.compute_score()
            all_sources.append(src)

    total = len(all_sources)

    # --- Deduplicate all text-based sources ---
    deduped = deduplicate_sources(all_sources)

    # --- Rebuild filtered lists ---
    for src in deduped:
        if src.source_type == "vector_memory" and vector_memories:
            # Find matching memory
            for mem in vector_memories:
                mem_text = getattr(mem, 'text', '') if hasattr(mem, 'text') else str(mem)
                if mem_text == src.text:
                    clean_memories.append(mem)
                    break
        elif src.source_type == "knowledge_graph" and graph_triples:
            for triple in graph_triples:
                subj = getattr(triple, 'subject', '') if hasattr(triple, 'subject') else ''
                obj = getattr(triple, 'object', '') if hasattr(triple, 'object') else ''
                if f"{subj} → {obj}" == src.text:
                    clean_triples.append(triple)
                    break

    kept = len(deduped)
    removed = total - kept

    if total > 0:
        print(f"[RetrievalIntel] Sources: {total} total → {kept} kept ({removed} removed)")

    return {
        "vector_memories": clean_memories,
        "graph_triples": clean_triples,
        "web_content": web_content,
        "stats": {"total": total, "kept": kept, "removed": removed},
    }
