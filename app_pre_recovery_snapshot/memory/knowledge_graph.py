"""
Knowledge Graph Memory — Firestore-based entity-relationship triples.
Adds structured relational reasoning alongside vector semantic memory.

Storage: Firestore → users/{uid}/knowledge_graph/triples
Schema: {subject, relation, object, user_id, importance, created_at, last_seen, source}

Pipeline:
  Extract ≤5 triples per exchange (background, post-stream)
  Retrieve by entity match (parallel with vector memory)
  Inject as <knowledge_graph> block into system prompt
"""
import asyncio
import json
import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from app.config import MEMORY_EXTRACTION_MODEL


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    importance: float = 0.7
    created_at: str = ""
    last_seen: str = ""
    doc_id: str = ""


MAX_TRIPLES_PER_USER = 300
MAX_INJECT = 5


# ============================================
# EXTRACTION
# ============================================

GRAPH_EXTRACTION_PROMPT = """Extract knowledge graph triples from this conversation.

A triple is: (subject, relationship, object)

Rules:
- Only extract durable, stable relationships
- Maximum 5 triples
- Each element must be 1-4 words
- DO NOT extract greetings, debugging steps, or temporary issues
- Focus on: architecture, tools, goals, preferences, identities

User message: {user_message}

Assistant response (summary): {assistant_summary}

Respond ONLY with a JSON array (no markdown):
[{{"subject": "User", "relation": "building", "object": "AI backend"}}, ...]

If nothing worth extracting, respond with: []"""


async def extract_triples(
    user_message: str,
    assistant_response: str,
    user_id: str = "",
) -> List[Triple]:
    """Extract knowledge graph triples from a conversation exchange."""
    if not user_message or len(user_message.strip()) < 10:
        return []

    assistant_summary = assistant_response[:400] if assistant_response else ""

    try:
        from app.llm.router import get_openrouter_client
        client = get_openrouter_client()

        prompt = GRAPH_EXTRACTION_PROMPT.format(
            user_message=user_message[:300],
            assistant_summary=assistant_summary,
        )

        response = await client.chat.completions.create(
            model=MEMORY_EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1,
        )

        raw = response.choices[0].message.content or "[]"
        triples = _parse_triples(raw)

        if triples:
            print(f"[KnowledgeGraph] Extracted {len(triples)} triples for {user_id}")

        return triples

    except Exception as e:
        print(f"[KnowledgeGraph] Extraction failed: {e}")
        return []


def _parse_triples(raw_text: str) -> List[Triple]:
    """Parse LLM response into Triple objects."""
    clean = raw_text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```\w*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()

    try:
        items = json.loads(clean)
        if not isinstance(items, list):
            return []
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    triples = []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue

        subj = str(item.get("subject", "")).strip()[:50]
        rel = str(item.get("relation", "")).strip()[:30]
        obj = str(item.get("object", "")).strip()[:50]

        if not subj or not rel or not obj:
            continue
        if len(subj) < 2 or len(obj) < 2:
            continue

        triples.append(Triple(
            subject=_normalize_entity(subj),
            relation=_normalize_entity(rel),
            object=_normalize_entity(obj),
            importance=min(1.0, float(item.get("importance", 0.7))),
        ))

    return triples


# ============================================
# STORAGE (Firestore)
# ============================================

def _normalize_entity(text: str) -> str:
    """Normalize entity: lowercase, strip punctuation, collapse spaces."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)     # collapse whitespace
    return text


def _triple_hash(t: Triple) -> str:
    """Deterministic hash for deduplication (uses normalized entities)."""
    key = f"{_normalize_entity(t.subject)}|{_normalize_entity(t.relation)}|{_normalize_entity(t.object)}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _get_triples_ref(user_id: str):
    from app.auth import get_firestore_db
    db = get_firestore_db()
    if not db:
        return None
    return db.collection("users").document(user_id).collection("knowledge_graph")


async def store_triples(user_id: str, triples: List[Triple]) -> int:
    """Store triples with dedup by hash. Returns count stored."""
    if not triples or user_id == "anonymous":
        return 0

    ref = _get_triples_ref(user_id)
    if not ref:
        return 0

    stored = 0
    now = datetime.utcnow().isoformat()

    try:
        for t in triples:
            h = _triple_hash(t)

            # Dedup check
            existing = ref.where("hash", "==", h).limit(1).stream()
            found = False
            for doc in existing:
                # Update: bump importance + last_seen
                doc.reference.update({
                    "importance": min(1.0, doc.to_dict().get("importance", 0.7) + 0.05),
                    "last_seen": now,
                })
                found = True
                break

            if found:
                continue

            # Capacity check
            count_query = ref.count()
            count_result = count_query.get()
            current_count = count_result[0][0].value if count_result else 0

            if current_count >= MAX_TRIPLES_PER_USER:
                # Evict lowest importance
                lowest = ref.order_by("importance").limit(1).stream()
                for doc in lowest:
                    if doc.to_dict().get("importance", 0) < t.importance:
                        doc.reference.delete()
                    else:
                        continue

            # Insert
            ref.add({
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "importance": t.importance,
                "created_at": now,
                "last_seen": now,
                "hash": h,
                "user_id": user_id,
            })
            stored += 1
            print(f"[KnowledgeGraph] Stored: ({t.subject}, {t.relation}, {t.object})")

    except Exception as e:
        print(f"[KnowledgeGraph] Indexing error: {e}")

    return stored


# ============================================
# RETRIEVAL QUALITY SCORING
# ============================================

async def bump_graph_scores(user_id: str, doc_ids: List[str], positive: bool = True):
    """
    Adjust the importance score and update last_seen of knowledge graph triples.
    Positive (+0.05) if cited by LLM, negative (-0.02) if retrieved but ignored.
    This prevents memory pollution over time.
    """
    if not doc_ids or user_id == "anonymous":
        return
        
    try:
        from app.auth import get_firestore_db
        db = get_firestore_db()
        if not db:
            return
            
        ref = db.collection("users").document(user_id).collection("knowledge_graph")
        now = datetime.utcnow().isoformat()
        
        for doc_id in doc_ids:
            if not doc_id or doc_id.startswith("graph_"):
                continue  # Skip un-ID'd graphs
                
            doc_ref = ref.document(doc_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                continue
                
            current_imp = doc.to_dict().get("importance", 0.7)
            
            if positive:
                new_imp = min(1.0, current_imp + 0.05)
            else:
                new_imp = max(0.1, current_imp - 0.02)
                
            doc_ref.update({
                "importance": new_imp,
                "last_seen": now
            })
            
    except Exception as e:
        print(f"[KnowledgeGraph] Failed to bump graph scores: {e}")


# ============================================
# RETRIEVAL
# ============================================

def _extract_entities(query: str) -> List[str]:
    """Fast entity extraction from query using keyword heuristics."""
    # Remove stopwords and short words, keep meaningful tokens
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "must",
        "how", "what", "when", "where", "why", "who", "which",
        "this", "that", "these", "those", "it", "its", "my", "your",
        "for", "with", "from", "into", "about", "between", "through",
        "to", "of", "in", "on", "at", "by", "and", "or", "not", "but",
        "if", "then", "than", "so", "very", "just", "also", "too",
        "i", "me", "we", "you", "he", "she", "they", "them",
    }

    words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
    entities = [w for w in words if w not in stopwords]

    return list(dict.fromkeys(entities))[:10]  # Deduplicated, max 10


async def retrieve_graph(
    user_id: str,
    query: str,
    limit: int = MAX_INJECT,
) -> List[Triple]:
    """Retrieve knowledge graph triples related to query entities."""
    if not user_id or user_id == "anonymous":
        return []

    entities = _extract_entities(query)
    if not entities:
        return []

    ref = _get_triples_ref(user_id)
    if not ref:
        return []

    try:
        # Fetch all user triples (Firestore can't do OR across fields)
        all_docs = ref.order_by("importance", direction="DESCENDING").limit(100).stream()

        now = datetime.utcnow()
        candidates: List[Tuple[float, Triple]] = []

        for doc in all_docs:
            data = doc.to_dict()
            subj = data.get("subject", "").lower()
            obj = data.get("object", "").lower()

            # Entity match: check if any query entity appears in subject or object
            match_score = 0.0
            for entity in entities:
                if entity in subj or entity in obj:
                    match_score += 1.0
                # Partial match
                elif any(entity in word or word in entity for word in subj.split()):
                    match_score += 0.5
                elif any(entity in word or word in entity for word in obj.split()):
                    match_score += 0.5

            if match_score == 0:
                continue

            importance = float(data.get("importance", 0.5))

            # Recency factor
            recency = 1.0
            last_seen = data.get("last_seen", "")
            if last_seen:
                try:
                    days_ago = (now - datetime.fromisoformat(last_seen)).days
                    if days_ago > 30:
                        recency = 0.7
                    elif days_ago > 90:
                        recency = 0.4
                except (ValueError, TypeError):
                    pass

            final_score = match_score * 0.5 + importance * 0.3 + recency * 0.2

            candidates.append((final_score, Triple(
                subject=data.get("subject", ""),
                relation=data.get("relation", ""),
                object=data.get("object", ""),
                importance=importance,
                created_at=data.get("created_at", ""),
                last_seen=last_seen,
                doc_id=doc.id,
            )))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [t for _, t in candidates[:limit]]

        # Debug logging
        if top:
            print(f"[KnowledgeGraph] Retrieval for: \"{query[:60]}\"")
            for t in top:
                print(f"  ({t.subject}, {t.relation}, {t.object}) imp={t.importance:.2f}")

        # Reinforce retrieved triples (fire and forget)
        for t in top:
            if t.doc_id:
                asyncio.create_task(_reinforce_triple(ref, t.doc_id, t.importance))

        return top

    except Exception as e:
        print(f"[KnowledgeGraph] Retrieval error: {e}")
        return []


# ============================================
# PROMPT FORMATTING
# ============================================

def format_graph_for_prompt(triples: List[Triple]) -> str:
    """Format triples as a compact block for system prompt injection."""
    if not triples:
        return ""

    lines = [f"- {t.subject} {t.relation} {t.object}" for t in triples]

    return (
        "\n<knowledge_graph>\n"
        + "\n".join(lines)
        + "\n</knowledge_graph>\n"
    )


# ============================================
# ORCHESTRATOR (called from websocket)
# ============================================

async def extract_and_store_graph(
    user_id: str,
    user_message: str,
    assistant_response: str,
) -> int:
    """Full pipeline: extract triples → store with dedup. Returns count stored."""
    triples = await extract_triples(user_message, assistant_response, user_id)
    if not triples:
        return 0

    return await store_triples(user_id, triples)


# ============================================
# HELPERS
# ============================================

async def _reinforce_triple(ref, doc_id: str, current_importance: float):
    """Bump importance on retrieval."""
    try:
        ref.document(doc_id).update({
            "last_seen": datetime.utcnow().isoformat(),
            "importance": min(1.0, current_importance + 0.05),
        })
    except Exception:
        pass
