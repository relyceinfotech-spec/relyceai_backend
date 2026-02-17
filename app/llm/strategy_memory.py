"""
Relyce AI - Strategy Memory
Persistent per-user preferences learned from interaction patterns.

Stores:
- preferred_language: Auto-detected from code queries
- preferred_depth: "concise" | "detailed" | "step-by-step"
- preferred_style: "code-first" | "explanation-first" | "mixed"

Persisted in Firestore: user_strategies/{user_id}
Cached in-memory with same LRU pattern as EmotionEngine.
"""
import time
import asyncio
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from collections import OrderedDict
from app.auth import get_firestore_db


@dataclass
class UserStrategy:
    """User's learned preferences."""
    preferred_language: str = "auto"                # python, javascript, etc.
    preferred_depth: str = "balanced"               # concise, balanced, detailed, step-by-step
    preferred_style: str = "mixed"                  # code-first, explanation-first, mixed
    language_counts: Dict[str, int] = field(default_factory=dict)    # {lang: count}
    depth_signals: Dict[str, int] = field(default_factory=lambda: {"concise": 0, "detailed": 0, "step-by-step": 0})
    style_signals: Dict[str, int] = field(default_factory=lambda: {"code-first": 0, "explanation-first": 0, "mixed": 0})
    total_interactions: int = 0
    last_updated: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserStrategy':
        return cls(
            preferred_language=data.get("preferred_language", "auto"),
            preferred_depth=data.get("preferred_depth", "balanced"),
            preferred_style=data.get("preferred_style", "mixed"),
            language_counts=data.get("language_counts", {}),
            depth_signals=data.get("depth_signals", {"concise": 0, "detailed": 0, "step-by-step": 0}),
            style_signals=data.get("style_signals", {"code-first": 0, "explanation-first": 0, "mixed": 0}),
            total_interactions=data.get("total_interactions", 0),
            last_updated=data.get("last_updated", 0.0)
        )

    def to_dict(self) -> Dict:
        return asdict(self)


# Language detection keywords
LANGUAGE_HINTS = {
    "python": ["python", "pip", "django", "flask", "fastapi", "pandas", "numpy", ".py", "def ", "import "],
    "javascript": ["javascript", "js", "node", "npm", "react", "vue", "angular", "next", "vite", ".js", "const ", "let "],
    "typescript": ["typescript", "ts", ".tsx", ".ts", "interface ", "type "],
    "java": ["java", "spring", "maven", "gradle", ".java", "public class"],
    "csharp": ["c#", "csharp", ".net", "dotnet", "asp.net"],
    "rust": ["rust", "cargo", ".rs", "fn main"],
    "go": ["golang", "go ", ".go", "func main"],
    "sql": ["sql", "select ", "insert ", "update ", "delete from", "join "],
    "html": ["html", "css", "div", "class=", "<html"],
}


class StrategyMemory:
    def __init__(self):
        # LRU cache: {user_id: UserStrategy}
        self._cache: OrderedDict[str, UserStrategy] = OrderedDict()
        self._cache_max = 200
        self._dirty: set = set()

    # ==========================================
    # Cache Ops
    # ==========================================
    def _cache_get(self, user_id: str) -> Optional[UserStrategy]:
        if user_id in self._cache:
            self._cache.move_to_end(user_id)
            return self._cache[user_id]
        return None

    def _cache_set(self, user_id: str, strategy: UserStrategy):
        if user_id in self._cache:
            self._cache.move_to_end(user_id)
        elif len(self._cache) >= self._cache_max:
            evicted_id, _ = self._cache.popitem(last=False)
            if evicted_id in self._dirty:
                self._dirty.discard(evicted_id)
        self._cache[user_id] = strategy

    # ==========================================
    # Load / Save
    # ==========================================
    async def load_strategy(self, user_id: str) -> UserStrategy:
        """Load user strategy from cache first, then Firestore."""
        cached = self._cache_get(user_id)
        if cached is not None:
            return cached

        try:
            db = get_firestore_db()
            if not db:
                return UserStrategy()
            
            doc_ref = db.collection("user_strategies").document(user_id)
            doc = await asyncio.to_thread(doc_ref.get)
            
            if doc.exists:
                strategy = UserStrategy.from_dict(doc.to_dict())
                self._cache_set(user_id, strategy)
                return strategy
            return UserStrategy()
        except Exception as e:
            print(f"[StrategyMemory] Error loading: {e}")
            return UserStrategy()

    async def save_strategy(self, user_id: str, strategy: UserStrategy):
        """Save to cache + background Firestore write."""
        strategy.last_updated = time.time()
        self._cache_set(user_id, strategy)
        self._dirty.add(user_id)

        try:
            db = get_firestore_db()
            if not db:
                return
            data = strategy.to_dict()
            asyncio.create_task(
                asyncio.to_thread(
                    lambda: db.collection("user_strategies").document(user_id).set(data)
                )
            )
        except Exception as e:
            print(f"[StrategyMemory] Error saving: {e}")

    # ==========================================
    # Learning Logic
    # ==========================================
    def detect_language(self, query: str) -> Optional[str]:
        """Detect programming language from query text."""
        query_lower = query.lower()
        scores = {}
        for lang, keywords in LANGUAGE_HINTS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[lang] = score
        
        if scores:
            return max(scores, key=scores.get)
        return None

    def detect_depth_signal(self, query: str) -> Optional[str]:
        """Detect depth preference from query patterns."""
        query_lower = query.lower()
        
        # Step-by-step signals
        if any(kw in query_lower for kw in ["step by step", "explain each", "walk me through", "break it down"]):
            return "step-by-step"
        # Detailed signals
        if any(kw in query_lower for kw in ["explain why", "deep dive", "in detail", "thoroughly", "how does it work"]):
            return "detailed"
        # Concise signals
        if any(kw in query_lower for kw in ["quick", "just give me", "tldr", "short", "one liner", "briefly"]):
            return "concise"
        return None

    def detect_style_signal(self, query: str) -> Optional[str]:
        """Detect response style preference."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["show me code", "show me the code", "code example", "just the code", "write code", "give me code", "just code"]):
            return "code-first"
        if any(kw in query_lower for kw in ["explain", "why does", "what is", "concept", "theory"]):
            return "explanation-first"
        return None

    def update_from_query(self, strategy: UserStrategy, query: str) -> UserStrategy:
        """Learn from a user query. Updates counts and resolves preferences."""
        strategy.total_interactions += 1
        
        # Language detection
        lang = self.detect_language(query)
        if lang:
            strategy.language_counts[lang] = strategy.language_counts.get(lang, 0) + 1
            # Resolve preferred language (most used)
            if strategy.language_counts:
                strategy.preferred_language = max(strategy.language_counts, key=strategy.language_counts.get)
        
        # Depth signal
        depth = self.detect_depth_signal(query)
        if depth:
            strategy.depth_signals[depth] = strategy.depth_signals.get(depth, 0) + 1
            strategy.preferred_depth = max(strategy.depth_signals, key=strategy.depth_signals.get)
        
        # Style signal
        style = self.detect_style_signal(query)
        if style:
            strategy.style_signals[style] = strategy.style_signals.get(style, 0) + 1
            strategy.preferred_style = max(strategy.style_signals, key=strategy.style_signals.get)
        
        return strategy

    def get_instruction(self, strategy: UserStrategy) -> Optional[str]:
        """Generate system prompt snippet from learned preferences."""
        if strategy.total_interactions < 3:
            return None  # Not enough data yet
        
        parts = []
        
        if strategy.preferred_language != "auto":
            parts.append(f"User's primary language: **{strategy.preferred_language}**. Default examples to this language.")
        
        depth_map = {
            "concise": "Keep responses concise and to-the-point.",
            "detailed": "Provide detailed explanations with reasoning.",
            "step-by-step": "Break down explanations step-by-step.",
            "balanced": "Balance brevity with clear explanation."
        }
        parts.append(depth_map.get(strategy.preferred_depth, ""))
        
        style_map = {
            "code-first": "Lead with code, then explain.",
            "explanation-first": "Explain the concept first, then show code.",
            "mixed": "Mix code and explanation naturally."
        }
        parts.append(style_map.get(strategy.preferred_style, ""))
        
        instruction = " ".join(p for p in parts if p)
        return instruction if instruction else None


# Singleton
strategy_memory = StrategyMemory()
