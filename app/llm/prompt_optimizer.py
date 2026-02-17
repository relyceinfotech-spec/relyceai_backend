"""
Relyce AI - Prompt Optimizer v2
Multi-Armed Bandit (Epsilon-Greedy) for automatic prompt A/B testing.

Architecture:
- Defines prompt variants per intent category
- Selects variant using epsilon-greedy (80% exploit, 20% explore)
- Tracks outcomes via FeedbackEngine signals
- Persists stats to Firestore (survives restarts)
- Converges on best-performing variant per intent
"""
import time
import random
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from app.auth import get_firestore_db


# ==========================================
# Prompt Variants (configurable per intent)
# ==========================================
PROMPT_VARIANTS = {
    "debugging": {
        "step-by-step": "Debug methodically: 1) Reproduce the issue 2) Identify root cause 3) Apply fix 4) Verify. Show each step.",
        "root-cause-first": "Jump straight to the root cause. Explain WHY the bug exists, then show the fix.",
        "fix-first": "Show the fix immediately. Then explain what was wrong and why this fixes it.",
    },
    "coding_complex": {
        "architecture-first": "Start with the high-level architecture/approach, then implement step by step.",
        "code-first": "Write the complete implementation first, then explain the key design decisions.",
        "incremental": "Build the solution incrementally: start simple, then add complexity layer by layer.",
    },
    "coding_simple": {
        "concise": "Give the most concise solution possible. Minimal explanation.",
        "educational": "Explain the approach briefly, show the code, then highlight key patterns used.",
    },
    "system_design": {
        "top-down": "Start with the system overview, then drill into each component.",
        "bottom-up": "Start with the core data model, then build up to the full system.",
        "tradeoffs": "Present 2-3 approaches with trade-offs, then recommend the best one for this case.",
    },
    "code_explanation": {
        "line-by-line": "Walk through the code line by line, explaining what each part does.",
        "conceptual": "Explain the high-level concept and pattern first, then show how the code implements it.",
        "visual": "Use analogies and mental models to explain the code's behavior.",
    },
    "general": {
        "default": "",  # No extra instruction
        "structured": "Structure your response with clear headers and bullet points.",
        "conversational": "Respond naturally and conversationally while being helpful.",
    },
}


@dataclass
class VariantStats:
    """Track performance stats for a prompt variant."""
    wins: int = 0
    total: int = 0
    last_updated: float = 0.0

    @property
    def score(self) -> float:
        if self.total == 0:
            return 0.5  # Neutral prior
        if self.total <= 3:
            return (self.wins + 1) / (self.total + 2)  # Bayesian prior for small samples
        return self.wins / self.total

    def to_dict(self) -> Dict:
        return {"wins": self.wins, "total": self.total, "last_updated": self.last_updated}

    @classmethod
    def from_dict(cls, data: Dict) -> "VariantStats":
        return cls(
            wins=data.get("wins", 0),
            total=data.get("total", 0),
            last_updated=data.get("last_updated", 0.0)
        )


class PromptOptimizer:
    def __init__(self, epsilon: float = 0.2):
        """
        epsilon: Exploration rate (0.2 = 20% random, 80% best-known)
        """
        self.epsilon = epsilon
        # {intent: {variant_name: VariantStats}}
        self._stats: Dict[str, Dict[str, VariantStats]] = defaultdict(
            lambda: defaultdict(VariantStats)
        )
        self._last_selected: Dict[str, str] = {}  # {session_id: variant_name}
        self._dirty = False
        self._loaded = False
        self._last_flush = time.time()
        self._flush_interval = 120  # seconds

    # ==========================================
    # Persistence
    # ==========================================
    async def load_stats(self):
        """Load variant stats from Firestore on startup."""
        if self._loaded:
            return
        try:
            db = get_firestore_db()
            if not db:
                self._loaded = True
                return

            doc = await asyncio.to_thread(
                lambda: db.collection("ai_learning").document("prompt_variants").get()
            )
            if doc.exists:
                data = doc.to_dict() or {}
                for intent, variants in data.items():
                    if isinstance(variants, dict):
                        for name, stats_data in variants.items():
                            if isinstance(stats_data, dict):
                                self._stats[intent][name] = VariantStats.from_dict(stats_data)
                print(f"[PromptOptimizer] Loaded variant stats from Firestore")

            self._loaded = True
        except Exception as e:
            print(f"[PromptOptimizer] Stats load failed: {e}")
            self._loaded = True

    async def save_stats(self):
        """Save variant stats to Firestore."""
        if not self._dirty:
            return

        try:
            db = get_firestore_db()
            if not db:
                return

            data = {}
            for intent, variants in self._stats.items():
                data[intent] = {name: stats.to_dict() for name, stats in variants.items()}

            await asyncio.to_thread(
                lambda: db.collection("ai_learning").document("prompt_variants").set(data)
            )
            self._dirty = False
            self._last_flush = time.time()
            print(f"[PromptOptimizer] Saved variant stats to Firestore")
        except Exception as e:
            print(f"[PromptOptimizer] Stats save failed: {e}")

    # ==========================================
    # Core API
    # ==========================================
    def select_variant(self, sub_intent: str, session_id: str = "") -> Tuple[str, str]:
        """
        Select a prompt variant using epsilon-greedy strategy.

        Returns: (variant_name, variant_instruction)
        """
        # Get variants for this intent
        variants = PROMPT_VARIANTS.get(sub_intent, PROMPT_VARIANTS["general"])
        variant_names = list(variants.keys())

        if not variant_names:
            return "default", ""

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: random variant
            selected = random.choice(variant_names)
        else:
            # Exploit: best-performing variant
            best_name = None
            best_score = -1
            for name in variant_names:
                stats = self._stats[sub_intent][name]
                score = stats.score
                if score > best_score:
                    best_score = score
                    best_name = name
            selected = best_name or variant_names[0]

        # Track selection for outcome attribution
        if session_id:
            self._last_selected[session_id] = selected

        return selected, variants[selected]

    def record_outcome(self, sub_intent: str, variant_name: str, success: bool):
        """Record the outcome of using a specific variant."""
        stats = self._stats[sub_intent][variant_name]
        stats.total += 1
        if success:
            stats.wins += 1
        stats.last_updated = time.time()
        self._dirty = True

        # Schedule periodic flush
        if time.time() - self._last_flush > self._flush_interval:
            try:
                asyncio.create_task(self.save_stats())
            except RuntimeError:
                pass

    def get_last_selected(self, session_id: str) -> Optional[str]:
        """Get the last variant selected for a session (for feedback attribution)."""
        return self._last_selected.get(session_id)

    def get_stats_summary(self) -> Dict:
        """Get a summary of all variant performance stats."""
        summary = {}
        for intent, variants in self._stats.items():
            summary[intent] = {}
            for name, stats in variants.items():
                summary[intent][name] = {
                    "score": round(stats.score, 3),
                    "wins": stats.wins,
                    "total": stats.total
                }
        return summary


# Singleton
prompt_optimizer = PromptOptimizer(epsilon=0.2)
