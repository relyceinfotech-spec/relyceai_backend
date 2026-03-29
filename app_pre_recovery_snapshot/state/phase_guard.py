"""
Phase Guard — Deterministic state machine for agent execution phases.

Enforces strict phase progression:
    analyzing → researching → planning → executing → repairing → finalizing → completed

Rules:
  - No regression (e.g., finalizing → planning is BLOCKED)
  - Lateral moves within the same tier are allowed (e.g., repairing → executing)
  - Thread-safe
"""
from __future__ import annotations

import threading
from typing import Optional, Dict

# ============================================
# PHASE DEFINITIONS (ordered by progression)
# ============================================

PHASE_ORDER = [
    "analyzing",
    "researching",
    "planning",
    "plan_preview",
    "executing",
    "using_tool",
    "repairing",
    "rollback_initiated",
    "finalizing",
    "completed",
]

# Map phase name → progression index
_PHASE_INDEX: Dict[str, int] = {phase: idx for idx, phase in enumerate(PHASE_ORDER)}

# Lateral equivalences (phases at the same tier)
_LATERAL_TIERS = {
    "executing": 4,
    "using_tool": 4,
    "repairing": 6,
    "rollback_initiated": 6,
}


class PhaseViolationError(RuntimeError):
    """Raised when an invalid phase transition is attempted."""
    pass


class PhaseGuard:
    """
    Enforces deterministic phase progression per execution.

    Usage:
        guard = PhaseGuard()
        guard.transition("session-1", "analyzing")    # OK
        guard.transition("session-1", "planning")      # OK
        guard.transition("session-1", "analyzing")     # RAISES PhaseViolationError
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, raises PhaseViolationError on regression.
                    If False, logs warning and skips the invalid transition.
        """
        self._lock = threading.Lock()
        self._sessions: Dict[str, str] = {}  # session_id → current phase
        self.strict = strict
        self.violations: int = 0

    def transition(self, session_id: str, new_phase: str) -> bool:
        """
        Attempt to transition to a new phase.

        Returns True if transition was accepted, False if rejected (non-strict mode).
        Raises PhaseViolationError in strict mode on regression.
        """
        with self._lock:
            current = self._sessions.get(session_id)

            # First phase — always allowed
            if current is None:
                self._sessions[session_id] = new_phase
                return True

            # Same phase — idempotent, always allowed
            if current == new_phase:
                return True

            current_idx = _PHASE_INDEX.get(current, -1)
            new_idx = _PHASE_INDEX.get(new_phase, -1)

            # Unknown phases — allow (future-proofing)
            if current_idx == -1 or new_idx == -1:
                self._sessions[session_id] = new_phase
                return True

            # Check lateral equivalence
            current_tier = _LATERAL_TIERS.get(current, current_idx)
            new_tier = _LATERAL_TIERS.get(new_phase, new_idx)

            # Forward or lateral — allowed
            if new_tier >= current_tier:
                self._sessions[session_id] = new_phase
                return True

            # REGRESSION DETECTED
            self.violations += 1
            if self.strict:
                raise PhaseViolationError(
                    f"Phase regression blocked: {current} (tier {current_tier}) → "
                    f"{new_phase} (tier {new_tier}) for session {session_id}"
                )
            else:
                print(
                    f"[PhaseGuard] WARNING: Blocked regression {current} → {new_phase} "
                    f"for session {session_id} (violations: {self.violations})"
                )
                return False

    def get_phase(self, session_id: str) -> Optional[str]:
        """Get current phase for a session."""
        with self._lock:
            return self._sessions.get(session_id)

    def reset(self, session_id: str) -> None:
        """Reset phase tracking for a session (on new execution)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def get_stats(self) -> dict:
        """Return guard statistics."""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "total_violations": self.violations,
                "strict_mode": self.strict,
            }


# ============================================
# MODULE SINGLETON
# ============================================

_default_guard: Optional[PhaseGuard] = None


def get_phase_guard() -> PhaseGuard:
    """Returns the module-level singleton PhaseGuard."""
    global _default_guard
    if _default_guard is None:
        _default_guard = PhaseGuard(strict=False)
    return _default_guard
