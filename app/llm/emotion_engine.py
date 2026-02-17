"""
Relyce AI - Emotional State Engine
Handles persistent emotional state, decay, and skill-level adaptation.

Cache Architecture:
- In-memory LRU cache (primary) -> Firestore (durable backup)
- Write batching: flush every N updates OR every T seconds
- Dirty flag: only persist if state actually changed
- Pre-warm: load state proactively on session connect
- Fail-safe: retry queue for failed Firestore writes
"""
import time
import copy
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set
from collections import OrderedDict
import asyncio
from app.auth import get_firestore_db

@dataclass
class EmotionalState:
    frustration: float = 0.0
    confusion: float = 0.0
    excitement: float = 0.0
    urgency: float = 0.0
    curiosity: float = 0.0
    skill_level: float = 0.5  # 0.0=Beginner, 1.0=Expert
    confidence: float = 0.5   # 0.0=Unsure, 1.0=Confident
    last_updated: float = 0.0
    last_emotions: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionalState':
        return cls(
            frustration=data.get("frustration", 0.0),
            confusion=data.get("confusion", 0.0),
            excitement=data.get("excitement", 0.0),
            urgency=data.get("urgency", 0.0),
            curiosity=data.get("curiosity", 0.0),
            skill_level=data.get("skill_level", 0.5),
            confidence=data.get("confidence", 0.5),
            last_updated=data.get("last_updated", 0.0),
            last_emotions=data.get("last_emotions", [])
        )

    def to_dict(self) -> Dict:
        return asdict(self)


class EmotionEngine:
    def __init__(self):
        # Variable Decay Rates
        self.decay_rates = {
            "frustration": 0.9,  # Slow decay
            "confusion": 0.8,
            "excitement": 0.7,   # Fast decay
            "urgency": 0.6,      # Very fast decay
            "curiosity": 0.8
        }
        self.default_decay = 0.8
        
        self.base_emotion_increment = 0.3
        self.adaptive_boost = 0.2  # +0.2 if repeated
        self.skill_increment = 0.05

        # ========== Cache Layer ==========
        # LRU cache using OrderedDict (move_to_end on access)
        self._cache: OrderedDict[str, EmotionalState] = OrderedDict()
        self._cache_max = 500

        # Dirty tracking: sessions that have unsaved changes
        self._dirty: Set[str] = set()

        # Write batching config
        self._write_batch_threshold = 5    # Flush after N dirty writes
        self._write_timer_seconds = 10.0   # Or flush every T seconds
        self._flush_task: Optional[asyncio.Task] = None
        self._update_count_since_flush = 0

        # Fail-safe retry queue: [(session_id, data_dict), ...]
        self._retry_queue: List[tuple] = []
        self._max_retries = 3

        # Long-term skill cache: {user_id: skill_level}
        self._skill_cache: Dict[str, float] = {}
        self._skill_dirty: Set[str] = set()

    # ==========================================
    # Cache Operations
    # ==========================================
    def _cache_get(self, session_id: str) -> Optional[EmotionalState]:
        """Get from cache (O(1)), returns None on miss."""
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            return self._cache[session_id]
        return None

    def _cache_set(self, session_id: str, state: EmotionalState):
        """Set in cache with LRU eviction."""
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
        elif len(self._cache) >= self._cache_max:
            # Evict LRU. If it's dirty, force-flush before evicting.
            evicted_id, evicted_state = self._cache.popitem(last=False)
            if evicted_id in self._dirty:
                self._dirty.discard(evicted_id)
                self._enqueue_write(evicted_id, evicted_state)
        self._cache[session_id] = state

    # ==========================================
    # Pre-warm (call on session connect)
    # ==========================================
    async def prewarm(self, session_id: str):
        """Pre-warm cache for a session. Call on WebSocket connect."""
        if session_id in self._cache:
            return  # Already cached
        await self.load_state(session_id)

    # ==========================================
    # Load / Save
    # ==========================================
    async def load_state(self, session_id: str, user_id: Optional[str] = None) -> EmotionalState:
        """Load: Cache first -> Firestore fallback. Merges persistent skill if user_id provided."""
        cached = self._cache_get(session_id)
        if cached is not None:
            # Merge persistent skill if available
            if user_id and user_id in self._skill_cache:
                cached.skill_level = self._skill_cache[user_id]
            return cached

        try:
            db = get_firestore_db()
            if not db:
                state = EmotionalState()
                if user_id:
                    state.skill_level = await self._load_user_skill(user_id, db)
                return state
            
            doc_ref = db.collection("chat_states").document(session_id)
            doc = await asyncio.to_thread(doc_ref.get)
            
            if doc.exists:
                state = EmotionalState.from_dict(doc.to_dict())
            else:
                state = EmotionalState()
            
            # Overlay persistent skill level from user store
            if user_id:
                state.skill_level = await self._load_user_skill(user_id, db)
            
            self._cache_set(session_id, state)
            return state
        except Exception as e:
            print(f"[EmotionEngine] Error loading state: {e}")
            return EmotionalState()

    async def save_state(self, session_id: str, state: EmotionalState, user_id: Optional[str] = None):
        """
        Save: Update cache immediately + mark dirty.
        Firestore write is batched (not every call).
        Also persists skill_level to permanent user store.
        """
        state.last_updated = time.time()
        self._cache_set(session_id, state)
        self._dirty.add(session_id)
        self._update_count_since_flush += 1

        # Persist skill to user-level store
        if user_id:
            self._skill_cache[user_id] = state.skill_level
            self._skill_dirty.add(user_id)

        # Check if we should flush now
        if self._update_count_since_flush >= self._write_batch_threshold:
            asyncio.create_task(self._flush_dirty())
        else:
            # Ensure timer-based flush is scheduled
            self._ensure_flush_timer()

    def _ensure_flush_timer(self):
        """Schedule a timer-based flush if not already running."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_delay())

    async def _flush_after_delay(self):
        """Flush dirty states after a delay."""
        await asyncio.sleep(self._write_timer_seconds)
        await self._flush_dirty()

    async def _flush_dirty(self):
        """Flush all dirty sessions to Firestore."""
        if not self._dirty:
            return
        
        # Snapshot dirty set and reset counter
        to_flush = self._dirty.copy()
        self._dirty.clear()
        self._update_count_since_flush = 0

        db = get_firestore_db()
        if not db:
            return

        for sid in to_flush:
            state = self._cache.get(sid)
            if state is None:
                continue
            try:
                data = state.to_dict()
                data["expire_at"] = time.time() + (24 * 3600)
                await asyncio.to_thread(
                    lambda s=sid, d=data: db.collection("chat_states").document(s).set(d)
                )
            except Exception as e:
                print(f"[EmotionEngine] Flush failed for {sid}: {e}")
                self._retry_queue.append((sid, state.to_dict()))

        # Process retry queue
        await self._process_retries(db)

    def _enqueue_write(self, session_id: str, state: EmotionalState):
        """Enqueue a write for fail-safe persistence (fire-and-forget)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._background_write(session_id, state))
        except Exception:
            self._retry_queue.append((session_id, state.to_dict()))

    async def _background_write(self, session_id: str, state: EmotionalState):
        """Single background write for evicted dirty entries."""
        try:
            db = get_firestore_db()
            if not db:
                return
            data = state.to_dict()
            data["expire_at"] = time.time() + (24 * 3600)
            await asyncio.to_thread(
                lambda: db.collection("chat_states").document(session_id).set(data)
            )
        except Exception as e:
            print(f"[EmotionEngine] Background write failed for {session_id}: {e}")
            self._retry_queue.append((session_id, state.to_dict()))

    async def _process_retries(self, db):
        """Process the retry queue (max 3 retries per item)."""
        if not self._retry_queue or not db:
            return
        
        remaining = []
        for sid, data in self._retry_queue:
            data["_retry_count"] = data.get("_retry_count", 0) + 1
            if data["_retry_count"] > self._max_retries:
                print(f"[EmotionEngine] DEAD LETTER: {sid} failed after {self._max_retries} retries")
                continue
            try:
                data["expire_at"] = time.time() + (24 * 3600)
                await asyncio.to_thread(
                    lambda s=sid, d=data: db.collection("chat_states").document(s).set(d)
                )
            except Exception:
                remaining.append((sid, data))
        
        self._retry_queue = remaining

    async def shutdown(self):
        """Flush all dirty states on shutdown. Call from app lifespan."""
        print("[EmotionEngine] Flushing on shutdown...")
        await self._flush_dirty()
        await self._flush_skills()

    # ==========================================
    # Long-Term Skill Persistence
    # ==========================================
    async def _load_user_skill(self, user_id: str, db=None) -> float:
        """Load persistent skill level for a user."""
        if user_id in self._skill_cache:
            return self._skill_cache[user_id]
        try:
            if not db:
                db = get_firestore_db()
            if not db:
                return 0.5
            doc = await asyncio.to_thread(
                lambda: db.collection("user_skills").document(user_id).get()
            )
            if doc.exists:
                skill = doc.to_dict().get("skill_level", 0.5)
                self._skill_cache[user_id] = skill
                return skill
        except Exception as e:
            print(f"[EmotionEngine] Error loading user skill: {e}")
        return 0.5

    async def _flush_skills(self):
        """Flush dirty skill levels to Firestore."""
        if not self._skill_dirty:
            return
        to_flush = self._skill_dirty.copy()
        self._skill_dirty.clear()
        try:
            db = get_firestore_db()
            if not db:
                return
            for uid in to_flush:
                skill = self._skill_cache.get(uid, 0.5)
                await asyncio.to_thread(
                    lambda u=uid, s=skill: db.collection("user_skills").document(u).set(
                        {"skill_level": s, "last_updated": time.time()}
                    )
                )
            print(f"[EmotionEngine] Flushed {len(to_flush)} user skills")
        except Exception as e:
            print(f"[EmotionEngine] Skill flush failed: {e}")

    # ==========================================
    # State Update Logic
    # ==========================================
    def update_state(self, state: EmotionalState, detected_emotions: List[str], intent: str) -> EmotionalState:
        """
        Update state based on new emotions and intent.
        Applies variable decay and adaptive increments.
        """
        # 1. Decay all emotions first
        state.frustration *= self.decay_rates.get("frustration", self.default_decay)
        state.confusion *= self.decay_rates.get("confusion", self.default_decay)
        state.excitement *= self.decay_rates.get("excitement", self.default_decay)
        state.urgency *= self.decay_rates.get("urgency", self.default_decay)
        state.curiosity *= self.decay_rates.get("curiosity", self.default_decay)
        
        # 2. Increment detected emotions (Adaptive)
        for emotion in detected_emotions:
            increment = self.base_emotion_increment
            if state.last_emotions and emotion in state.last_emotions:
                increment += self.adaptive_boost
            
            if emotion == "frustrated":
                state.frustration += increment
            elif emotion == "confused":
                state.confusion += increment
            elif emotion == "excited":
                state.excitement += increment
            elif emotion == "urgent":
                state.urgency += increment
            elif emotion in ["curious", "curiosity"]:
                state.curiosity += increment
        
        state.last_emotions = detected_emotions

        # 3. Update Skill Level & Confidence
        if intent in ["coding_complex", "system_design", "debugging", "architecture_planning"]:
            state.skill_level += self.skill_increment
            state.confidence += 0.05
        
        if "confused" in detected_emotions:
            state.skill_level -= self.skill_increment
            state.confidence -= 0.1
        
        if "frustrated" in detected_emotions:
            state.confidence -= 0.1
        
        # 4. Clamp values
        for attr in ["frustration", "confusion", "excitement", "urgency", "curiosity", "skill_level", "confidence"]:
            setattr(state, attr, max(0.0, min(1.0, getattr(state, attr))))
        
        return state

    # ==========================================
    # Instruction Generation
    # ==========================================
    def get_instruction(self, state: EmotionalState) -> str:
        """Generate system prompt instruction based on state matrix."""
        instructions = []
        
        if state.skill_level < 0.3:
            instructions.append("**User Level: BEGINNER**. Explain concepts simply. Avoid jargon. Use analogies.")
        elif state.skill_level > 0.7:
            instructions.append("**User Level: EXPERT**. Be concise. Skip basics. Focus on advanced details/trade-offs.")
        else:
            instructions.append("**User Level: INTERMEDIATE**. Balance explanation with code. Assume basic knowledge.")
        
        if state.frustration > 0.6:
            instructions.append("**HIGH FRUSTRATION DETECTED**. Stop apologizing. Fix the problem immediately. Be extremely direct.")
        elif state.confusion > 0.6:
            instructions.append("**HIGH CONFUSION DETECTED**. Slow down. Break it into step-by-step pieces. Check for understanding.")
        elif state.urgency > 0.6:
            instructions.append("**URGENCY DETECTED**. No pleasantries. Solution first. Code immediately.")
        elif state.excitement > 0.6:
            instructions.append("**HIGH EXCITEMENT**. Match energy! Use encouraging language.")
        elif state.curiosity > 0.6:
            instructions.append("**HIGH CURIOSITY**. Go deep. Explain the 'Why'. Teach the underlying concept.")
            
        return "\n".join(instructions)

    # ==========================================
    # Frustration Prediction (Priority #6)
    # ==========================================
    def predict_frustration(self, state: EmotionalState, query: str = "", sub_intent: str = "") -> float:
        """
        Predict probability of user becoming frustrated.
        
        Leading indicators:
        - Rising confusion (confusion > 0.4 and not yet frustrated)
        - High urgency + confusion combo (time pressure + don't understand)
        - Low-but-present frustration trending up (0.2-0.5 range)
        - Query length increasing (longer queries = struggling to articulate)
        - Debugging/complex intents with confusion = pre-frustration
        """
        score = 0.0

        # Rising confusion is the #1 leading indicator of frustration
        if state.confusion > 0.4:
            score += 0.3
        elif state.confusion > 0.2:
            score += 0.1

        # Urgency + confusion = time pressure + stuck = frustrated soon
        if state.urgency > 0.5 and state.confusion > 0.3:
            score += 0.2

        # Frustration already present but low = could escalate
        if 0.2 < state.frustration < 0.5:
            score += 0.2

        # Debugging/complex intent + confusion = pre-frustration pattern
        if sub_intent in ("debugging", "coding_complex", "system_design") and state.confusion > 0.2:
            score += 0.15

        # Long queries often signal user struggling to explain their problem
        if len(query) > 200:
            score += 0.1
        elif len(query) > 400:
            score += 0.2

        # Low confidence + any negative emotion = approaching frustration
        if state.confidence < 0.3 and (state.confusion > 0.2 or state.frustration > 0.1):
            score += 0.15

        return min(score, 1.0)

    def get_proactive_instruction(self, frustration_prob: float) -> Optional[str]:
        """
        Generate proactive help instruction when frustration is predicted.
        Returns None if probability is too low.
        """
        if frustration_prob < 0.5:
            return None

        if frustration_prob > 0.7:
            return (
                "**PROACTIVE HELP MODE (HIGH):** The user is likely approaching frustration. "
                "Simplify your explanation significantly. Break the problem into smaller steps. "
                "Offer alternative approaches. Be warm but action-oriented — fix the problem, "
                "don't just sympathize."
            )
        return (
            "**PROACTIVE HELP MODE:** The user may be struggling. "
            "Keep your response clear and structured. Offer to break the problem down "
            "if they'd like. Ask if a different approach would help."
        )

# Singleton instance
emotion_engine = EmotionEngine()
