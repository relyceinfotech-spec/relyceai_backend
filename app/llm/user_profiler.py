"""
Relyce AI - Unified User Profiler
Persistent per-user profile that combines all learning signals:
- Skill level (from emotion engine)
- Learning preferences (from interaction patterns)
- Engagement patterns (from session data)
- Frustration triggers (from emotion history)

Persisted to Firestore at users/{uid}/ai_profile
LRU cached in memory for fast access.
"""
import time
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from collections import OrderedDict
from app.auth import get_firestore_db


@dataclass
class UserProfile:
    """Comprehensive user profile built over time."""
    user_id: str = ""

    # Skill modeling (0=beginner, 1=expert)
    skill_level: float = 0.5
    skill_confidence: float = 0.0       # How confident we are in this estimate
    skill_samples: int = 0              # Number of data points

    # Learning preferences (all 0-1 scales, 0.5=neutral)
    prefers_examples: float = 0.5       # 0=no examples, 1=always show examples
    prefers_code_first: float = 0.5     # 0=explain-first, 1=code-first
    prefers_concise: float = 0.5        # 0=detailed, 1=concise
    prefers_step_by_step: float = 0.5   # 0=holistic, 1=step-by-step

    # Engagement stats
    total_interactions: int = 0
    topics_of_interest: Dict[str, int] = field(default_factory=dict)
    avg_query_length: float = 0.0

    # Resolution patterns
    successful_strategies: Dict[str, int] = field(default_factory=dict)
    failed_strategies: Dict[str, int] = field(default_factory=dict)

    last_updated: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "UserProfile":
        return cls(
            user_id=data.get("user_id", ""),
            skill_level=data.get("skill_level", 0.5),
            skill_confidence=data.get("skill_confidence", 0.0),
            skill_samples=data.get("skill_samples", 0),
            prefers_examples=data.get("prefers_examples", 0.5),
            prefers_code_first=data.get("prefers_code_first", 0.5),
            prefers_concise=data.get("prefers_concise", 0.5),
            prefers_step_by_step=data.get("prefers_step_by_step", 0.5),
            total_interactions=data.get("total_interactions", 0),
            topics_of_interest=data.get("topics_of_interest", {}),
            avg_query_length=data.get("avg_query_length", 0.0),
            successful_strategies=data.get("successful_strategies", {}),
            failed_strategies=data.get("failed_strategies", {}),
            last_updated=data.get("last_updated", 0.0)
        )


# Learning rate: how fast preferences shift
LEARNING_RATE = 0.1
SKILL_LEARNING_RATE = 0.05


class UserProfiler:
    """
    Manages persistent user profiles with in-memory LRU cache.
    Learns preferences from interaction patterns and outcomes.
    """

    def __init__(self, cache_max: int = 200):
        self._cache: OrderedDict[str, UserProfile] = OrderedDict()
        self._cache_max = cache_max
        self._dirty: set = set()
        self._flush_interval = 30  # seconds
        self._last_flush = time.time()

    # ==========================================
    # Cache Operations
    # ==========================================
    def _cache_get(self, user_id: str) -> Optional[UserProfile]:
        if user_id in self._cache:
            self._cache.move_to_end(user_id)
            return self._cache[user_id]
        return None

    def _cache_set(self, user_id: str, profile: UserProfile):
        if user_id in self._cache:
            self._cache.move_to_end(user_id)
        elif len(self._cache) >= self._cache_max:
            evicted_id, _ = self._cache.popitem(last=False)
            if evicted_id in self._dirty:
                self._dirty.discard(evicted_id)
                # Fire-and-forget save for evicted dirty profiles
                try:
                    asyncio.create_task(self._save_profile(evicted_id))
                except RuntimeError:
                    pass
        self._cache[user_id] = profile

    # ==========================================
    # Load / Save
    # ==========================================
    async def load_profile(self, user_id: str) -> UserProfile:
        """Load user profile from cache or Firestore."""
        cached = self._cache_get(user_id)
        if cached is not None:
            return cached

        try:
            db = get_firestore_db()
            if not db:
                profile = UserProfile(user_id=user_id)
                self._cache_set(user_id, profile)
                return profile

            doc = await asyncio.to_thread(
                lambda: db.collection("users").document(user_id)
                         .collection("ai_profile").document("profile").get()
            )
            if doc.exists:
                profile = UserProfile.from_dict(doc.to_dict())
                profile.user_id = user_id
            else:
                profile = UserProfile(user_id=user_id)

            self._cache_set(user_id, profile)
            return profile
        except Exception as e:
            print(f"[UserProfiler] Error loading profile: {e}")
            profile = UserProfile(user_id=user_id)
            self._cache_set(user_id, profile)
            return profile

    async def _save_profile(self, user_id: str):
        """Save a single user profile to Firestore."""
        profile = self._cache.get(user_id)
        if not profile:
            return

        try:
            db = get_firestore_db()
            if not db:
                return
            data = profile.to_dict()
            data["last_updated"] = time.time()
            await asyncio.to_thread(
                lambda: db.collection("users").document(user_id)
                         .collection("ai_profile").document("profile").set(data)
            )
        except Exception as e:
            print(f"[UserProfiler] Error saving profile for {user_id}: {e}")

    async def flush_dirty(self):
        """Flush all dirty profiles to Firestore."""
        if not self._dirty:
            return

        to_flush = self._dirty.copy()
        self._dirty.clear()
        self._last_flush = time.time()

        for uid in to_flush:
            await self._save_profile(uid)

        if to_flush:
            print(f"[UserProfiler] Flushed {len(to_flush)} profiles")

    # ==========================================
    # Learning from Interactions
    # ==========================================
    def update_from_interaction(
        self,
        profile: UserProfile,
        sub_intent: str,
        query: str,
        response_length: int,
        emotions: List[str],
        outcome: str = "unknown",
        model_used: str = "",
        prompt_variant: str = "default"
    ) -> UserProfile:
        """
        Update user profile based on interaction signals.
        Called after each response is generated.
        """
        profile.total_interactions += 1

        # Track topics of interest
        if sub_intent:
            profile.topics_of_interest[sub_intent] = (
                profile.topics_of_interest.get(sub_intent, 0) + 1
            )

        # Update average query length (running average)
        n = profile.total_interactions
        profile.avg_query_length = (
            (profile.avg_query_length * (n - 1) + len(query)) / n
        )

        # ---- Infer learning preferences from behavior ----

        # Preference: concise vs detailed
        # If user asks short questions → they probably prefer concise answers
        if len(query) < 50:
            self._nudge_preference(profile, "prefers_concise", towards=1.0)
        elif len(query) > 300:
            self._nudge_preference(profile, "prefers_concise", towards=0.0)

        # Preference: code-first vs explanation-first
        # Coding intents suggest code-first preference
        coding_intents = {"coding_simple", "coding_complex", "debugging", "sql", "ui_implementation"}
        if sub_intent in coding_intents:
            self._nudge_preference(profile, "prefers_code_first", towards=1.0)
        elif sub_intent in {"code_explanation", "general", "casual_chat"}:
            self._nudge_preference(profile, "prefers_code_first", towards=0.0)

        # Preference: step-by-step
        # Complex intents with confusion → user needs step-by-step
        if "confused" in emotions and sub_intent in {"coding_complex", "system_design", "debugging"}:
            self._nudge_preference(profile, "prefers_step_by_step", towards=1.0)

        # Preference: examples
        # When user asks "how" or "example" → prefers examples
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["example", "show me", "how to", "how do"]):
            self._nudge_preference(profile, "prefers_examples", towards=1.0)

        # ---- Skill level update ----
        profile.skill_samples += 1
        if sub_intent in {"system_design", "architecture_planning", "coding_complex"}:
            profile.skill_level = min(1.0, profile.skill_level + SKILL_LEARNING_RATE)
            profile.skill_confidence = min(1.0, profile.skill_confidence + 0.02)
        if "confused" in emotions:
            profile.skill_level = max(0.0, profile.skill_level - SKILL_LEARNING_RATE * 0.5)

        # ---- Track strategy outcomes ----
        if outcome in ("success", "partial"):
            strategy_key = f"{model_used}:{prompt_variant}"
            profile.successful_strategies[strategy_key] = (
                profile.successful_strategies.get(strategy_key, 0) + 1
            )
        elif outcome == "failure":
            strategy_key = f"{model_used}:{prompt_variant}"
            profile.failed_strategies[strategy_key] = (
                profile.failed_strategies.get(strategy_key, 0) + 1
            )

        profile.last_updated = time.time()
        self._dirty.add(profile.user_id)

        # Check if we should flush
        if time.time() - self._last_flush > self._flush_interval:
            try:
                asyncio.create_task(self.flush_dirty())
            except RuntimeError:
                pass

        return profile

    @staticmethod
    def _nudge_preference(profile: UserProfile, attr: str, towards: float):
        """Gradually shift a preference towards a target value."""
        current = getattr(profile, attr)
        delta = (towards - current) * LEARNING_RATE
        setattr(profile, attr, max(0.0, min(1.0, current + delta)))

    # ==========================================
    # Personalization Output
    # ==========================================
    def get_personalization_instruction(self, profile: UserProfile) -> str:
        """
        Generate system prompt personalization snippet based on learned profile.
        Only activates after enough interactions (>5) to avoid premature personalization.
        """
        if profile.total_interactions < 5:
            return ""  # Not enough data yet

        instructions = []

        # Skill-based instruction
        if profile.skill_level < 0.3 and profile.skill_confidence > 0.3:
            instructions.append("This user is a beginner. Use simple language, avoid jargon, include analogies.")
        elif profile.skill_level > 0.7 and profile.skill_confidence > 0.3:
            instructions.append("This user is an expert. Be concise, skip basics, focus on advanced details.")

        # Style preferences
        if profile.prefers_concise > 0.7:
            instructions.append("This user prefers concise, to-the-point answers.")
        elif profile.prefers_concise < 0.3:
            instructions.append("This user prefers detailed, thorough explanations.")

        if profile.prefers_code_first > 0.7:
            instructions.append("Show code first, then explain if needed.")
        elif profile.prefers_code_first < 0.3:
            instructions.append("Explain the concept first, then show code.")

        if profile.prefers_step_by_step > 0.7:
            instructions.append("Break solutions into clear step-by-step instructions.")

        if profile.prefers_examples > 0.7:
            instructions.append("Include concrete examples when explaining concepts.")

        # Topic expertise
        if profile.topics_of_interest:
            top_topics = sorted(profile.topics_of_interest.items(), key=lambda x: x[1], reverse=True)[:3]
            topic_names = [t[0].replace("_", " ") for t in top_topics]
            instructions.append(f"User often works on: {', '.join(topic_names)}.")

        if not instructions:
            return ""

        return "**Personalization (learned from past interactions):**\n" + "\n".join(f"- {i}" for i in instructions)

    def get_recommended_style(self, profile: UserProfile) -> Dict:
        """Return optimal response style parameters based on profile."""
        return {
            "concise": profile.prefers_concise > 0.6,
            "code_first": profile.prefers_code_first > 0.6,
            "step_by_step": profile.prefers_step_by_step > 0.6,
            "include_examples": profile.prefers_examples > 0.6,
            "skill_level": profile.skill_level,
        }

    async def shutdown(self):
        """Flush all dirty profiles on shutdown."""
        await self.flush_dirty()
        print("[UserProfiler] Shutdown complete")


# Singleton
user_profiler = UserProfiler()
