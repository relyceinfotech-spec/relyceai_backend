"""
Relyce AI - Embedding-Based Intent Classification
Uses OpenRouter's text-embedding-3-small for semantic intent matching.
Includes gray-zone tie-breaker with gpt-4o-mini.
"""
import asyncio
import math
import time
from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI
import httpx

from app.config import OPENROUTER_API_KEY
from app.auth import get_firestore_db

# ============================================
# CONFIGURATION
# ============================================
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TIEBREAKER_MODEL = "openai/gpt-4o-mini"
THRESHOLD_HIGH = 0.75
THRESHOLD_MEDIUM = 0.60
TIMEOUT_SECONDS = 0.3

# ============================================
# INTENT DEFINITIONS (7 intents × 5-8 examples)
# ============================================
INTENT_EXAMPLES: Dict[str, List[str]] = {
    "casual": [
        "hi",
        "hello",
        "how are you",
        "tell me a joke",
        "good morning",
        "what's up",
        "hey macha",
        "yo"
    ],
    "coding_simple": [
        "explain this code",
        "what does this function do",
        "sql select example",
        "basic javascript example",
        "simple api request example",
        "how to write a function"
    ],
    "coding_complex": [
        "debug this issue",
        "why is my system failing",
        "design a scalable backend",
        "optimize database performance",
        "fix production bug",
        "refactor this architecture"
    ],
    "analysis_internal": [
        "why does this happen",
        "compare two systems",
        "analyze the impact",
        "pros and cons of this approach",
        "which is better and why",
        "evaluate these options"
    ],
    "business": [
        "business strategy advice",
        "startup growth plan",
        "pricing strategy analysis",
        "go to market strategy",
        "business decision help",
        "revenue model suggestions"
    ],
    "web_factual": [
        "latest news",
        "current price",
        "today updates",
        "recent announcement",
        "what happened today",
        "weather today"
    ],
    "web_analysis": [
        "why did this happen today",
        "analyze recent market changes",
        "compare latest technologies",
        "impact of new policy",
        "predict future trend",
        "explain current situation"
    ]
}

# ============================================
# KEYWORD RULES (for gray-zone validation)
# ============================================
KEYWORD_RULES: Dict[str, List[str]] = {
    "casual": ["hi", "hello", "hey", "yo", "macha", "sup", "morning", "evening", "joke"],
    "coding_simple": ["explain", "what does", "example", "how to", "basic", "simple"],
    "coding_complex": ["debug", "fix", "error", "failing", "optimize", "refactor", "design", "architecture"],
    "analysis_internal": ["why", "compare", "analyze", "pros", "cons", "better", "evaluate", "which"],
    "business": ["business", "startup", "strategy", "revenue", "pricing", "market", "growth"],
    "web_factual": ["latest", "today", "current", "now", "recent", "news", "price", "weather"],
    "web_analysis": ["trend", "predict", "impact", "changes", "situation", "happening"]
}

# ============================================
# ROUTING CONFIGURATION
# ============================================
INTENT_ROUTES: Dict[str, Dict] = {
    "casual": {"needs_web": False, "needs_reasoning": False, "model": "gemini"},
    "coding_simple": {"needs_web": False, "needs_reasoning": False, "model": "qwen"},
    "coding_complex": {"needs_web": False, "needs_reasoning": True, "model": "qwen"},
    "analysis_internal": {"needs_web": False, "needs_reasoning": True, "model": "gemini"},
    "business": {"needs_web": False, "needs_reasoning": False, "model": "openai"},
    "web_factual": {"needs_web": True, "needs_reasoning": False, "model": "gemini"},
    "web_analysis": {"needs_web": True, "needs_reasoning": True, "model": "gemini"}
}

# ============================================
# FAST PATH RULES (skip embeddings entirely)
# ============================================
OBVIOUS_CASUAL = {
    "hi", "hello", "hey", "yo", "sup", "hola", "morning", "evening",
    "bye", "goodbye", "thanks", "thank you", "thx", "ty",
    "good morning", "good evening", "good night", "how are you",
    "hey macha", "hi macha", "macha", "bro", "vanakkam", "namaste"
}

OBVIOUS_WEB_KEYWORDS = [
    "latest", "today", "current", "now", "recent", "news",
    "price of", "weather", "stock", "2024", "2025", "2026"
]

def _fast_path_check(query: str) -> Optional[Dict]:
    """
    Check if query can be classified instantly without embeddings.
    Returns None if embedding classification is needed.
    ~5-10ms vs ~400-500ms for embedding API.
    """
    q = query.lower().strip()
    
    # 1. Obvious casual queries (greetings, thanks, etc)
    if q in OBVIOUS_CASUAL or len(q) <= 10 and any(c in q for c in ["hi", "hey", "yo"]):
        route = INTENT_ROUTES["casual"]
        print(f"[Embedding] ⚡ FAST PATH: casual (instant)")
        return {
            "intent": "casual",
            "confidence": 1.0,
            "needs_web": route["needs_web"],
            "needs_reasoning": route["needs_reasoning"],
            "model": route["model"],
            "path": "fast_path_casual"
        }
    
    # 2. Obvious web queries (latest, today, price, etc)
    if any(kw in q for kw in OBVIOUS_WEB_KEYWORDS):
        route = INTENT_ROUTES["web_factual"]
        print(f"[Embedding] ⚡ FAST PATH: web_factual (instant)")
        return {
            "intent": "web_factual",
            "confidence": 1.0,
            "needs_web": route["needs_web"],
            "needs_reasoning": route["needs_reasoning"],
            "model": route["model"],
            "path": "fast_path_web"
        }
    
    return None

# ============================================
# LRU CACHE FOR QUERY RESULTS
# ============================================
from functools import lru_cache

# Cache for query classification results (max 1000 queries)
_query_result_cache: Dict[str, Dict] = {}
_CACHE_MAX_SIZE = 1000

def _get_cached_result(query: str) -> Optional[Dict]:
    """Check if query result is cached."""
    return _query_result_cache.get(query.lower().strip())

def _cache_result(query: str, result: Dict) -> None:
    """Cache query result (with size limit)."""
    key = query.lower().strip()
    if len(_query_result_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (FIFO)
        oldest_key = next(iter(_query_result_cache))
        del _query_result_cache[oldest_key]
    _query_result_cache[key] = result

# ============================================
# IN-MEMORY EMBEDDING CACHE
# ============================================
_intent_cache: Dict[str, List[Dict]] = {}
_cache_loaded: bool = False

# ============================================
# OPENROUTER CLIENT
# ============================================
_openrouter_client: Optional[AsyncOpenAI] = None

def get_openrouter_client() -> AsyncOpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        _openrouter_client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    return _openrouter_client


# ============================================
# CORE FUNCTIONS
# ============================================

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for text using OpenRouter."""
    try:
        client = get_openrouter_client()
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"[Embedding] Error generating embedding: {e}")
        return []


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def rules_agree(intent: str, query: str) -> bool:
    """Check if keyword rules agree with embedding classification."""
    query_lower = query.lower()
    keywords = KEYWORD_RULES.get(intent, [])
    
    # Check if any keyword matches
    matches = sum(1 for kw in keywords if kw in query_lower)
    return matches > 0


async def gpt4o_mini_tiebreaker(query: str) -> str:
    """Use gpt-4o-mini to classify intent when embeddings are uncertain."""
    try:
        client = get_openrouter_client()
        
        intent_list = ", ".join(INTENT_EXAMPLES.keys())
        prompt = f"""Classify this query into ONE of these intents: {intent_list}

Query: "{query}"

Rules:
- casual: greetings, small talk, jokes
- coding_simple: basic code questions, explanations
- coding_complex: debugging, optimization, system design  
- analysis_internal: comparisons, evaluations, pros/cons
- business: strategy, startup, revenue questions
- web_factual: current events, prices, weather, news
- web_analysis: trend analysis, predictions, market analysis

Reply with ONLY the intent name, nothing else."""

        response = await client.chat.completions.create(
            model=TIEBREAKER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Validate result
        if result in INTENT_EXAMPLES:
            print(f"[Embedding] gpt-4o-mini tiebreaker: {result}")
            return result
        
        # Fallback
        return "casual"
        
    except Exception as e:
        print(f"[Embedding] Tiebreaker error: {e}")
        return "casual"


# ============================================
# FIRESTORE FUNCTIONS
# ============================================

async def load_intent_embeddings() -> bool:
    """Load embeddings from Firestore into memory cache (Non-blocking)."""
    global _intent_cache, _cache_loaded
    
    if _cache_loaded:
        return True
    
    try:
        db = get_firestore_db()
        if not db:
            print("[Embedding] Firestore not available")
            return False
        
        def _fetch_sync():
            """Synchronous Firestore fetching to run in thread"""
            cache_data = {}
            for intent in INTENT_EXAMPLES.keys():
                try:
                    doc = db.collection('intent_embeddings').document(intent).get()
                    if doc.exists:
                        data = doc.to_dict()
                        cache_data[intent] = data.get('examples', [])
                except Exception as ex:
                    print(f"[Embedding] Failed to fetch {intent}: {ex}")
            return cache_data

        # Run blocking I/O in a separate thread
        import asyncio
        fetched_cache = await asyncio.to_thread(_fetch_sync)
        
        if fetched_cache:
            _intent_cache.update(fetched_cache)
            _cache_loaded = True
            print(f"[Embedding] Loaded {len(_intent_cache)} intents into cache")
            return True
        
        print("[Embedding] No embeddings found in Firestore")
        return False
        
    except Exception as e:
        print(f"[Embedding] Error loading embeddings: {e}")
        return False


async def save_intent_embeddings() -> bool:
    """Save intent embeddings to Firestore (one-time seed)."""
    try:
        db = get_firestore_db()
        if not db:
            print("[Embedding] Firestore not available")
            return False
        
        for intent, examples in INTENT_EXAMPLES.items():
            print(f"[Embedding] Generating embeddings for: {intent}")
            
            embedded_examples = []
            for text in examples:
                embedding = await generate_embedding(text)
                if embedding:
                    embedded_examples.append({
                        "text": text,
                        "embedding": embedding
                    })
                await asyncio.sleep(0.1)  # Rate limit
            
            # Save to Firestore
            db.collection('intent_embeddings').document(intent).set({
                'examples': embedded_examples,
                'updated_at': time.time()
            })
            print(f"[Embedding] Saved {len(embedded_examples)} examples for {intent}")
        
        return True
        
    except Exception as e:
        print(f"[Embedding] Error saving embeddings: {e}")
        return False


# ============================================
# MAIN CLASSIFICATION FUNCTION
# ============================================

async def classify_intent(query: str) -> Dict:
    """
    Classify query intent using embeddings with gray-zone tie-breaker.
    
    Returns: {
        "intent": str,
        "confidence": float,
        "needs_web": bool,
        "needs_reasoning": bool,
        "model": str,
        "path": str
    }
    """
    t_start = time.time()
    
    # 1. FAST PATH: Check for obvious patterns (instant ~5ms)
    fast_result = _fast_path_check(query)
    if fast_result:
        return fast_result
    
    # 2. LRU CACHE: Check if we've seen this query before (~1ms)
    cached = _get_cached_result(query)
    if cached:
        print(f"[Embedding] 📦 CACHE HIT: {cached['intent']} (instant)")
        cached["path"] = "cache_hit"
        return cached
    
    # 3. Ensure embedding cache is loaded from Firestore
    if not _cache_loaded:
        await load_intent_embeddings()
    
    # If cache empty, use fallback
    if not _intent_cache:
        print("[Embedding] Cache empty, using fallback")
        return {
            "intent": "casual",
            "confidence": 0.0,
            "needs_web": False,
            "needs_reasoning": False,
            "model": "gemini",
            "path": "fallback_no_cache"
        }
    
    # Generate query embedding
    query_embedding = await generate_embedding(query)
    if not query_embedding:
        return {
            "intent": "casual",
            "confidence": 0.0,
            "needs_web": False,
            "needs_reasoning": False,
            "model": "gemini",
            "path": "fallback_embedding_error"
        }
    
    # Compare with all cached embeddings
    best_intent = "casual"
    best_score = 0.0
    
    for intent, examples in _intent_cache.items():
        for example in examples:
            score = cosine_similarity(query_embedding, example.get('embedding', []))
            if score > best_score:
                best_score = score
                best_intent = intent
    
    elapsed_ms = (time.time() - t_start) * 1000
    
    # Decision logic with thresholds
    path = ""
    final_intent = best_intent
    
    if best_score >= THRESHOLD_HIGH:
        # High confidence - trust embedding
        path = "embedding_high"
        print(f"[Embedding] intent={best_intent} score={best_score:.2f} path={path} ({elapsed_ms:.0f}ms)")
    
    elif best_score >= THRESHOLD_MEDIUM:
        # Gray zone - check keywords first
        if rules_agree(best_intent, query):
            path = "embedding_medium_rules_agree"
            print(f"[Embedding] intent={best_intent} score={best_score:.2f} path={path} ({elapsed_ms:.0f}ms)")
        else:
            # Tie-breaker
            path = "embedding_medium_tiebreaker"
            final_intent = await gpt4o_mini_tiebreaker(query)
            print(f"[Embedding] tiebreaker: {best_intent}->{final_intent} score={best_score:.2f} path={path}")
    
    else:
        # Low confidence - safe fallback
        path = "fallback_low_confidence"
        final_intent = "casual"
        print(f"[Embedding] score={best_score:.2f} too low, fallback to casual")
    
    # Get route config
    route = INTENT_ROUTES.get(final_intent, INTENT_ROUTES["casual"])
    
    result = {
        "intent": final_intent,
        "confidence": best_score,
        "needs_web": route["needs_web"],
        "needs_reasoning": route["needs_reasoning"],
        "model": route["model"],
        "path": path
    }
    
    # 4. CACHE RESULT for future queries
    _cache_result(query, result)
    
    return result


async def classify_intent_with_timeout(query: str) -> Dict:
    """Classify intent with timeout fallback."""
    try:
        return await asyncio.wait_for(
            classify_intent(query),
            timeout=TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        print(f"[Embedding] Timeout after {TIMEOUT_SECONDS*1000:.0f}ms, using fallback")
        return {
            "intent": "casual",
            "confidence": 0.0,
            "needs_web": False,
            "needs_reasoning": False,
            "model": "gemini",
            "path": "timeout_fallback"
        }
