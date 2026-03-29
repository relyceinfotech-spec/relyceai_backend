"""
Relyce AI - Self-Learning Feedback Engine v2
Tracks interaction outcomes and learns optimal strategies per user profile.

Architecture:
- Logs every interaction with metadata (intent, model, emotion, latency)
- Infers outcome from user's NEXT message (implicit feedback)
- Scores strategies by (skill_bucket, emotion_bucket, intent) key
- Persists strategy scores to Firestore (survives restarts)
- Aggregates scores globally across users (cold-start fallback)
- Wires outcomes to prompt_optimizer for variant learning
- Exposes get_best_strategy() and get_best_model() for routing
"""
import time
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, OrderedDict
from app.auth import get_firestore_db


# ==========================================
# Interaction Log Entry
# ==========================================
@dataclass
class InteractionLog:
    session_id: str = ""
    user_id: str = ""
    timestamp: float = 0.0
    query: str = ""
    intent: str = ""
    sub_intent: str = ""
    model_used: str = ""
    emotional_state: str = ""       # dominant emotion at time of query
    skill_bucket: str = "mid"       # "low", "mid", "high"
    response_length: int = 0
    latency_ms: float = 0.0
    prompt_variant: str = "default"
    outcome: str = "pending"        # "success", "partial", "failure", "pending"

    def to_dict(self) -> Dict:
        return asdict(self)


# ==========================================
# Strategy Score Entry
# ==========================================
@dataclass
class StrategyScore:
    wins: int = 0
    total: int = 0
    score: float = 0.5
    last_updated: float = 0.0

    def record(self, success: bool):
        self.total += 1
        if success:
            self.wins += 1
        # Weighted score with time decay: recent outcomes matter more
        # Simple approach: rolling average with minimum sample prior
        if self.total <= 3:
            self.score = (self.wins + 1) / (self.total + 2)  # Bayesian prior
        else:
            self.score = self.wins / self.total
        self.last_updated = time.time()

    def to_dict(self) -> Dict:
        return {"wins": self.wins, "total": self.total, "score": self.score, "last_updated": self.last_updated}

    @classmethod
    def from_dict(cls, data: Dict) -> "StrategyScore":
        return cls(
            wins=data.get("wins", 0),
            total=data.get("total", 0),
            score=data.get("score", 0.5),
            last_updated=data.get("last_updated", 0.0)
        )


# ==========================================
# Strategy Score Tracker (Persistent)
# ==========================================
class StrategyScorer:
    """
    Tracks which strategies work best for each user profile bucket.
    Key = (skill_bucket, emotion_bucket, intent)
    Value = {strategy_name: StrategyScore}
    Now persistent via Firestore.
    """
    def __init__(self):
        # {profile_key: {strategy_name: StrategyScore}}
        self._scores: Dict[str, Dict[str, StrategyScore]] = defaultdict(dict)
        self._dirty = False
        self._loaded = False

    def record_outcome(self, profile_key: str, strategy: str, success: bool):
        """Update strategy score based on outcome."""
        if strategy not in self._scores[profile_key]:
            self._scores[profile_key][strategy] = StrategyScore()
        self._scores[profile_key][strategy].record(success)
        self._dirty = True

    def get_best(self, profile_key: str, prefix: str = "") -> Optional[Dict]:
        """Get the best-performing strategy for a profile key, optionally filtered by prefix."""
        strategies = self._scores.get(profile_key)
        if not strategies:
            return None

        filtered = {k: v for k, v in strategies.items() if k.startswith(prefix)} if prefix else strategies
        if not filtered:
            return None

        best_name = max(filtered, key=lambda k: filtered[k].score)
        entry = filtered[best_name]
        return {
            "strategy": best_name,
            "score": entry.score,
            "total_samples": entry.total,
            "confidence": entry.total  # alias for routing decisions
        }

    def to_dict(self) -> Dict:
        """Serialize all scores for Firestore."""
        result = {}
        for profile_key, strategies in self._scores.items():
            result[profile_key] = {name: s.to_dict() for name, s in strategies.items()}
        return result

    def load_from_dict(self, data: Dict):
        """Load scores from Firestore data."""
        for profile_key, strategies in data.items():
            for name, score_data in strategies.items():
                self._scores[profile_key][name] = StrategyScore.from_dict(score_data)
        self._loaded = True


# ==========================================
# Global Strategy Aggregator
# ==========================================
class GlobalStrategyAggregator:
    """
    Aggregates strategy scores across all users.
    Used as cold-start fallback when a user has insufficient data.
    """
    def __init__(self):
        self._global_scores: Dict[str, Dict[str, StrategyScore]] = defaultdict(dict)
        self._dirty = False

    def merge_outcome(self, profile_key: str, strategy: str, success: bool):
        """Add a data point to global aggregation."""
        if strategy not in self._global_scores[profile_key]:
            self._global_scores[profile_key][strategy] = StrategyScore()
        self._global_scores[profile_key][strategy].record(success)
        self._dirty = True

    def get_global_best(self, profile_key: str, prefix: str = "", min_samples: int = 5) -> Optional[Dict]:
        """Get globally best strategy when user has insufficient data."""
        strategies = self._global_scores.get(profile_key)
        if not strategies:
            return None

        filtered = {k: v for k, v in strategies.items() if k.startswith(prefix)} if prefix else strategies
        # Only return if we have enough global data
        qualified = {k: v for k, v in filtered.items() if v.total >= min_samples}
        if not qualified:
            return None

        best_name = max(qualified, key=lambda k: qualified[k].score)
        entry = qualified[best_name]
        return {
            "strategy": best_name,
            "score": entry.score,
            "total_samples": entry.total,
            "source": "global"
        }

    def to_dict(self) -> Dict:
        result = {}
        for profile_key, strategies in self._global_scores.items():
            result[profile_key] = {name: s.to_dict() for name, s in strategies.items()}
        return result

    def load_from_dict(self, data: Dict):
        for profile_key, strategies in data.items():
            for name, score_data in strategies.items():
                self._global_scores[profile_key][name] = StrategyScore.from_dict(score_data)


# ==========================================
# Feedback Engine
# ==========================================
class FeedbackEngine:
    def __init__(self):
        # Recent interaction buffer per session (for outcome inference)
        self._last_interaction: Dict[str, InteractionLog] = {}
        self._scorer = StrategyScorer()
        self._global = GlobalStrategyAggregator()

        # Batch log buffer for Firestore (flush periodically)
        self._log_buffer: List[Dict] = []
        self._log_flush_threshold = 10
        self._score_flush_interval = 60  # Flush scores every 60 seconds
        self._last_score_flush = time.time()
        self._flush_task: Optional[asyncio.Task] = None
        self._scores_loaded = False

    # ==========================================
    # Startup: Load persisted scores
    # ==========================================
    async def load_scores(self):
        """Load strategy scores from Firestore on startup."""
        if self._scores_loaded:
            return
        try:
            db = get_firestore_db()
            if not db:
                self._scores_loaded = True
                return

            # Load per-user strategy scores
            doc = await asyncio.to_thread(
                lambda: db.collection("ai_learning").document("strategy_scores").get()
            )
            if doc.exists:
                self._scorer.load_from_dict(doc.to_dict() or {})
                print(f"[FeedbackEngine] Loaded strategy scores from Firestore")

            # Load global aggregation
            global_doc = await asyncio.to_thread(
                lambda: db.collection("ai_learning").document("global_scores").get()
            )
            if global_doc.exists:
                self._global.load_from_dict(global_doc.to_dict() or {})
                print(f"[FeedbackEngine] Loaded global scores from Firestore")

            self._scores_loaded = True
        except Exception as e:
            print(f"[FeedbackEngine] Score load failed: {e}")
            self._scores_loaded = True  # Don't retry forever

    # ==========================================
    # Skill Bucket Helpers
    # ==========================================
    @staticmethod
    def _skill_bucket(skill_level: float) -> str:
        if skill_level < 0.3:
            return "low"
        elif skill_level > 0.7:
            return "high"
        return "mid"

    @staticmethod
    def _emotion_bucket(emotions: List[str]) -> str:
        """Get dominant emotion category."""
        priority = ["frustrated", "confused", "urgent", "excited", "curious"]
        for emotion in priority:
            if emotion in emotions:
                return emotion
        return "neutral"

    @staticmethod
    def _profile_key(skill_bucket: str, emotion_bucket: str, intent: str) -> str:
        return f"{skill_bucket}:{emotion_bucket}:{intent}"

    # ==========================================
    # Outcome Inference
    # ==========================================
    def _infer_outcome(self, prev_log: InteractionLog, new_emotions: List[str], new_intent: str) -> str:
        """
        Infer outcome of previous interaction based on user's next message.

        Signals:
        - Topic change + no frustration = RESOLVED (success)
        - Follow-up on same topic + no negative emotion = PARTIAL (needs more)
        - Repeated frustration/confusion = FAILURE
        - Excitement/curiosity after response = SUCCESS
        """
        prev_intent = prev_log.sub_intent

        # Failure signals
        if "frustrated" in new_emotions and prev_log.emotional_state == "frustrated":
            return "failure"  # Repeated frustration = we failed
        if "confused" in new_emotions and prev_log.emotional_state == "confused":
            return "failure"  # Still confused = didn't help

        # Success signals
        if "excited" in new_emotions or "curious" in new_emotions:
            return "success"  # Positive reaction
        if new_intent != prev_intent and "frustrated" not in new_emotions:
            return "success"  # Moved on without frustration = resolved

        # Partial: same topic, neutral emotion
        return "partial"

    # ==========================================
    # Core API
    # ==========================================
    def log_interaction(
        self,
        session_id: str,
        user_id: str,
        query: str,
        intent: str,
        sub_intent: str,
        model_used: str,
        emotions: List[str],
        skill_level: float,
        response_length: int,
        latency_ms: float,
        prompt_variant: str = "default"
    ):
        """
        Log an interaction. Also infers outcome of PREVIOUS interaction.
        Wires outcomes to prompt_optimizer and global aggregator.
        """
        skill_bucket = self._skill_bucket(skill_level)
        emotion_bucket = self._emotion_bucket(emotions)

        # Infer outcome of previous interaction (if exists for this session)
        prev = self._last_interaction.get(session_id)
        if prev:
            outcome = self._infer_outcome(prev, emotions, sub_intent)
            prev.outcome = outcome

            # Update strategy scores
            prev_profile_key = self._profile_key(prev.skill_bucket, prev.emotional_state, prev.sub_intent)
            is_success = outcome == "success"
            is_not_failure = outcome != "failure"

            # Score model
            self._scorer.record_outcome(
                prev_profile_key,
                f"model:{prev.model_used}",
                is_success
            )
            # Score prompt variant
            self._scorer.record_outcome(
                prev_profile_key,
                f"prompt:{prev.prompt_variant}",
                is_success
            )

            # Update global aggregation
            self._global.merge_outcome(prev_profile_key, f"model:{prev.model_used}", is_success)
            self._global.merge_outcome(prev_profile_key, f"prompt:{prev.prompt_variant}", is_success)

            # Wire to prompt optimizer
            try:
                from app.llm.prompt_optimizer import prompt_optimizer
                prompt_optimizer.record_outcome(prev.sub_intent, prev.prompt_variant, is_success)
            except Exception:
                pass  # Non-critical

            # Buffer for Firestore
            self._log_buffer.append(prev.to_dict())

            print(f"[FeedbackEngine] Outcome inferred: {outcome} for {prev.sub_intent} "
                  f"(model={prev.model_used}, variant={prev.prompt_variant})")

        # Store current interaction as "last"
        current = InteractionLog(
            session_id=session_id,
            user_id=user_id,
            timestamp=time.time(),
            query=query[:200],  # Truncate for storage
            intent=intent,
            sub_intent=sub_intent,
            model_used=model_used,
            emotional_state=emotion_bucket,
            skill_bucket=skill_bucket,
            response_length=response_length,
            latency_ms=latency_ms,
            prompt_variant=prompt_variant
        )
        self._last_interaction[session_id] = current

        # Check if we should flush logs to Firestore
        if len(self._log_buffer) >= self._log_flush_threshold:
            try:
                asyncio.create_task(self._flush_logs())
            except RuntimeError:
                pass  # No event loop

        # Periodically flush strategy scores
        if time.time() - self._last_score_flush > self._score_flush_interval:
            try:
                asyncio.create_task(self._flush_scores())
            except RuntimeError:
                pass

    def get_best_strategy(
        self,
        skill_level: float,
        emotions: List[str],
        sub_intent: str
    ) -> Optional[Dict]:
        """
        Get the best-performing strategy for the given user profile.
        Falls back to global aggregation if user has insufficient data.
        Returns: {"model": best_model, "prompt_variant": best_variant, "confidence": int} or None
        """
        skill_bucket = self._skill_bucket(skill_level)
        emotion_bucket = self._emotion_bucket(emotions)
        profile_key = self._profile_key(skill_bucket, emotion_bucket, sub_intent)

        result = {}

        # Find best model (user-level first, then global fallback)
        model_best = self._scorer.get_best(profile_key, prefix="model:")
        if model_best and model_best["total_samples"] >= 3:
            result["model"] = model_best["strategy"].replace("model:", "")
            result["model_score"] = model_best["score"]
            result["confidence"] = model_best["total_samples"]
        else:
            # Fallback to global
            global_model = self._global.get_global_best(profile_key, prefix="model:")
            if global_model:
                result["model"] = global_model["strategy"].replace("model:", "")
                result["model_score"] = global_model["score"]
                result["confidence"] = global_model["total_samples"]
                result["source"] = "global"

        # Find best prompt variant
        prompt_best = self._scorer.get_best(profile_key, prefix="prompt:")
        if prompt_best and prompt_best["total_samples"] >= 3:
            result["prompt_variant"] = prompt_best["strategy"].replace("prompt:", "")
            result["prompt_score"] = prompt_best["score"]
        else:
            global_prompt = self._global.get_global_best(profile_key, prefix="prompt:")
            if global_prompt:
                result["prompt_variant"] = global_prompt["strategy"].replace("prompt:", "")
                result["prompt_score"] = global_prompt["score"]

        return result if result else None

    def get_best_model(
        self,
        skill_level: float,
        emotions: List[str],
        sub_intent: str,
        min_confidence: int = 10
    ) -> Optional[Dict]:
        """
        Get the best-performing model for routing decisions.
        Only returns if confidence is high enough to override defaults.
        Returns: {"model": model_id, "score": float, "confidence": int} or None
        """
        strategy = self.get_best_strategy(skill_level, emotions, sub_intent)
        if not strategy:
            return None
        if strategy.get("confidence", 0) < min_confidence:
            return None
        if "model" not in strategy:
            return None
        return {
            "model": strategy["model"],
            "score": strategy.get("model_score", 0.5),
            "confidence": strategy.get("confidence", 0)
        }

    # ==========================================
    # Persistence
    # ==========================================
    async def _flush_logs(self):
        """Flush interaction logs to Firestore."""
        if not self._log_buffer:
            return

        to_flush = self._log_buffer.copy()
        self._log_buffer.clear()

        try:
            db = get_firestore_db()
            if not db:
                return

            batch = db.batch()
            collection = db.collection("interaction_logs")

            for log_entry in to_flush:
                doc_ref = collection.document()
                batch.set(doc_ref, log_entry)

            await asyncio.to_thread(batch.commit)
            print(f"[FeedbackEngine] Flushed {len(to_flush)} interaction logs")
        except Exception as e:
            print(f"[FeedbackEngine] Flush failed: {e}")
            # Re-add to buffer for retry
            self._log_buffer.extend(to_flush)

    async def _flush_scores(self):
        """Flush strategy scores to Firestore."""
        if not self._scorer._dirty and not self._global._dirty:
            return

        self._last_score_flush = time.time()

        try:
            db = get_firestore_db()
            if not db:
                return

            # Flush user-level scores
            if self._scorer._dirty:
                scores_data = self._scorer.to_dict()
                # Firestore doesn't allow nested maps with dots in keys, so sanitize
                safe_data = {}
                for k, v in scores_data.items():
                    safe_key = k.replace(".", "_").replace("/", "_")
                    safe_strategies = {}
                    for sk, sv in v.items():
                        safe_sk = sk.replace(".", "_").replace("/", "_")
                        safe_strategies[safe_sk] = sv
                    safe_data[safe_key] = safe_strategies

                await asyncio.to_thread(
                    lambda: db.collection("ai_learning").document("strategy_scores").set(safe_data)
                )
                self._scorer._dirty = False
                print(f"[FeedbackEngine] Flushed strategy scores ({len(scores_data)} profiles)")

            # Flush global scores
            if self._global._dirty:
                global_data = self._global.to_dict()
                safe_global = {}
                for k, v in global_data.items():
                    safe_key = k.replace(".", "_").replace("/", "_")
                    safe_strategies = {}
                    for sk, sv in v.items():
                        safe_sk = sk.replace(".", "_").replace("/", "_")
                        safe_strategies[safe_sk] = sv
                    safe_global[safe_key] = safe_strategies

                await asyncio.to_thread(
                    lambda: db.collection("ai_learning").document("global_scores").set(safe_global)
                )
                self._global._dirty = False
                print(f"[FeedbackEngine] Flushed global scores")

        except Exception as e:
            print(f"[FeedbackEngine] Score flush failed: {e}")

    async def shutdown(self):
        """Flush remaining logs and scores on shutdown."""
        # Finalize any pending last interactions
        for session_id, log in self._last_interaction.items():
            if log.outcome == "pending":
                log.outcome = "unknown"
                self._log_buffer.append(log.to_dict())

        await self._flush_logs()
        await self._flush_scores()
        print("[FeedbackEngine] Shutdown complete")


# Singleton
feedback_engine = FeedbackEngine()
