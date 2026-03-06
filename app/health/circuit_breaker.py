"""
Phase 8: Circuit Breaker
State machine that controls whether the system accepts new requests.

States:
  CLOSED  → System healthy, accepting requests
  OPEN    → System unhealthy, rejecting new requests
  (No half-open state in MVP — cooldown timer resets to CLOSED)

Design rules:
  - Never opens mid-task
  - Only evaluated at request boundary
  - Cooldown prevents flapping
  - Thread-safe
"""
from __future__ import annotations

import time
import threading
from typing import Optional


# Cooldown period before breaker can close again (seconds)
DEFAULT_COOLDOWN_SECONDS = 30


class CircuitBreaker:
    """
    Production-grade circuit breaker with cooldown.

    Usage:
        breaker = CircuitBreaker()
        breaker.evaluate(health_signal)
        if not breaker.allow_execution():
            return 503
    """

    def __init__(self, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS):
        self._lock = threading.Lock()
        self.state: str = "CLOSED"
        self.cooldown_seconds = cooldown_seconds
        self.last_open_time: Optional[float] = None
        self.last_signal: str = "HEALTHY"
        self.open_count: int = 0

    def evaluate(self, health_signal: str) -> None:
        """
        Update breaker state based on health signal.

        If unhealthy → OPEN (with timestamp)
        If healthy AND cooldown elapsed → CLOSED
        """
        with self._lock:
            self.last_signal = health_signal

            if health_signal != "HEALTHY":
                if self.state == "CLOSED":
                    self.state = "OPEN"
                    self.last_open_time = time.time()
                    self.open_count += 1
                # If already OPEN, refresh the open timestamp
                elif self.state == "OPEN":
                    self.last_open_time = time.time()
            else:
                if self.state == "OPEN" and self.last_open_time is not None:
                    elapsed = time.time() - self.last_open_time
                    if elapsed >= self.cooldown_seconds:
                        self.state = "CLOSED"
                        self.last_open_time = None

    def allow_execution(self) -> bool:
        """Returns True if system is accepting new tasks."""
        with self._lock:
            return self.state == "CLOSED"

    def force_close(self) -> None:
        """Manual override to close the breaker (admin use)."""
        with self._lock:
            self.state = "CLOSED"
            self.last_open_time = None

    def force_open(self) -> None:
        """Manual override to open the breaker (emergency kill switch)."""
        with self._lock:
            self.state = "OPEN"
            self.last_open_time = time.time()
            self.open_count += 1

    def get_state(self) -> dict:
        """Return breaker state snapshot."""
        with self._lock:
            return {
                "state": self.state,
                "last_signal": self.last_signal,
                "open_count": self.open_count,
                "cooldown_seconds": self.cooldown_seconds,
                "last_open_time": self.last_open_time,
            }


# ============================================
# MODULE SINGLETON
# ============================================

_default_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Returns the module-level singleton breaker."""
    global _default_breaker
    if _default_breaker is None:
        _default_breaker = CircuitBreaker()
    return _default_breaker
