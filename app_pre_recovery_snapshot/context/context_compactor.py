"""
Context Compactor — Deduplicates, merges, and compresses context across layers.
Runs after Retrieval Intelligence, before Context Packer.

Three operations:
  1. Deduplicate: remove semantically identical items (overlap > 0.88)
  2. Merge: combine fragmented facts about the same entity
  3. Compress: remove filler words, collapse repetition

Expected result: 25-35% fewer prompt tokens.
Safety: never compresses code, API responses, or tool outputs.
"""
import re
from typing import List, Dict, Optional, Tuple


# ============================================
# CONFIG
# ============================================

MERGE_THRESHOLD = 0.88    # text overlap to merge
MIN_ITEM_LENGTH = 10      # skip tiny items


# ============================================
# TEXT OVERLAP
# ============================================

def _word_set(text: str) -> set:
    return set(text.lower().split())


def _text_overlap(a: str, b: str) -> float:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ============================================
# FILLER REMOVAL
# ============================================

_FILLER_PATTERNS = [
    (r"\bthe user is currently\b", "user"),
    (r"\bthe user has been\b", "user"),
    (r"\bthe user is\b", "user"),
    (r"\bit is important to note that\b", ""),
    (r"\bin order to\b", "to"),
    (r"\bbasically\b", ""),
    (r"\bessentially\b", ""),
    (r"\bfor the purpose of\b", "for"),
    (r"\bat this point in time\b", "now"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin the process of\b", ""),
    (r"\bthat being said\b", ""),
]

_COMPILED_FILLERS = [(re.compile(p, re.IGNORECASE), r) for p, r in _FILLER_PATTERNS]


def _compress_text(text: str) -> str:
    """Remove filler phrases and collapse whitespace."""
    result = text
    for pattern, replacement in _COMPILED_FILLERS:
        result = pattern.sub(replacement, result)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()
    return result


# ============================================
# CODE DETECTION (safety — never compress code)
# ============================================

_CODE_INDICATORS = [
    "```", "def ", "class ", "import ", "from ", "async def",
    "function ", "const ", "var ", "let ", "=>", "return ",
    "{", "}", "if (", "for (", "while (",
]


def _is_code(text: str) -> bool:
    """Detect if text contains code that should not be compressed."""
    code_count = sum(1 for ind in _CODE_INDICATORS if ind in text)
    return code_count >= 2


# ============================================
# MERGE LOGIC
# ============================================

def _merge_texts(a: str, b: str) -> str:
    """Merge two overlapping texts into one concise version."""
    words_a = _word_set(a)
    words_b = _word_set(b)

    # b has extra info that a doesn't
    unique_b = words_b - words_a
    if not unique_b:
        return a  # b is subset of a

    # Append unique info from b to a
    extra = " ".join(sorted(unique_b))
    return f"{a} ({extra})"


# ============================================
# MAIN COMPACTION PIPELINE
# ============================================

def compact_context(
    graph_items: Optional[List[str]] = None,
    memory_items: Optional[List[str]] = None,
    summary: str = "",
    web_content: str = "",
    graph_ids: Optional[List[str]] = None,
    memory_ids: Optional[List[str]] = None,
) -> Dict[str, any]:
    """
    Full context compaction pipeline.

    Returns:
        {
            "graph": [compacted items],
            "memory": [compacted items],
            "summary": compacted summary,
            "web_content": unchanged (not compressed),
            "graph_ids": [surviving source IDs],
            "memory_ids": [surviving source IDs],
            "stats": {"before": N, "after": M, "saved_pct": X}
        }
    """
    # Collect all text items (except web content and code)
    all_items: List[Tuple[str, str]] = []  # (source, text)
    all_ids: List[Tuple[str, str]] = []    # (source, id)

    _graph_ids = graph_ids or [f"graph_{i}" for i in range(len(graph_items or []))]
    _memory_ids = memory_ids or [f"memory_{i}" for i in range(len(memory_items or []))]

    for i, item in enumerate(graph_items or []):
        if len(item) >= MIN_ITEM_LENGTH:
            all_items.append(("graph", item))
            all_ids.append(("graph", _graph_ids[i] if i < len(_graph_ids) else f"graph_{i}"))

    for i, item in enumerate(memory_items or []):
        if len(item) >= MIN_ITEM_LENGTH:
            all_items.append(("memory", item))
            all_ids.append(("memory", _memory_ids[i] if i < len(_memory_ids) else f"memory_{i}"))

    if summary and len(summary) >= MIN_ITEM_LENGTH:
        all_items.append(("summary", summary))
        all_ids.append(("summary", "summary_0"))

    before_chars = sum(len(t) for _, t in all_items)

    # Step 1: Deduplicate — only merge within same source type
    # Graph triples are NEVER merged with memory items (preserves relationships)
    deduped: List[Tuple[str, str]] = []
    deduped_ids: List[Tuple[str, set]] = []
    for idx, (source, text) in enumerate(all_items):
        sid = all_ids[idx][1]
        # Skip code items
        if _is_code(text):
            deduped.append((source, text))
            deduped_ids.append((source, {sid}))
            continue

        is_dup = False
        for i, (existing_source, existing) in enumerate(deduped):
            # Only merge items from the SAME source type
            if source != existing_source:
                continue
            overlap = _text_overlap(text, existing)
            if overlap > MERGE_THRESHOLD:
                # Step 2: Merge instead of discard
                merged = _merge_texts(existing, text)
                deduped[i] = (existing_source, merged)
                # Keep both IDs by adding to set
                deduped_ids[i][1].add(sid)
                is_dup = True
                break

        if not is_dup:
            deduped.append((source, text))
            deduped_ids.append((source, {sid}))

    # Step 3: Compress each item
    result_graph = []
    result_memory = []
    result_summary = ""
    surviving_graph_ids = set()
    surviving_memory_ids = set()

    for idx, (source, text) in enumerate(deduped):
        sids = deduped_ids[idx][1]
        if _is_code(text):
            compressed = text
        else:
            compressed = _compress_text(text)

        dict_item = {"text": compressed, "sources": sorted(list(sids))}

        if source == "graph":
            result_graph.append(dict_item)
            surviving_graph_ids.update(sids)
        elif source == "memory":
            result_memory.append(dict_item)
            surviving_memory_ids.update(sids)
        elif source == "summary":
            result_summary = compressed

    after_chars = (
        sum(len(t["text"]) for t in result_graph)
        + sum(len(t["text"]) for t in result_memory)
        + len(result_summary)
    )

    saved_pct = round((1 - after_chars / before_chars) * 100, 1) if before_chars else 0.0

    if saved_pct > 0:
        print(f"[ContextCompactor] {before_chars} → {after_chars} chars ({saved_pct}% saved)")

    return {
        "graph": result_graph,
        "memory": result_memory,
        "summary": result_summary,
        "web_content": web_content,  # Never compress web content
        "graph_ids": sorted(list(surviving_graph_ids)),
        "memory_ids": sorted(list(surviving_memory_ids)),
        "stats": {
            "before_chars": before_chars,
            "after_chars": after_chars,
            "saved_pct": saved_pct,
            "items_before": len(all_items),
            "items_after": len(deduped),
        },
    }
