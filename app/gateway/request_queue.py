"""
Request Queue — Concurrency limiter for incoming requests.
Prevents server overload during traffic spikes.

Config:
  MAX_ACTIVE = 20   active request slots
  QUEUE_LIMIT = 200 max waiting in queue
  QUEUE_TIMEOUT = 30s max wait time
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


MAX_ACTIVE_REQUESTS = 20
QUEUE_LIMIT = 200
QUEUE_TIMEOUT = 30.0  # seconds


@dataclass
class QueueStats:
    """Live queue statistics."""
    active: int = 0
    queued: int = 0
    total_processed: int = 0
    total_rejected: int = 0
    total_timeouts: int = 0
    peak_active: int = 0
    peak_queued: int = 0


# Module-level state
_semaphore: Optional[asyncio.Semaphore] = None
_stats = QueueStats()


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_ACTIVE_REQUESTS)
    return _semaphore


class RequestSlot:
    """
    Context manager to acquire a request processing slot.

    Usage:
        async with RequestSlot() as slot:
            if slot.acquired:
                # process request
            else:
                # return 503
    """
    def __init__(self):
        self.acquired = False
        self.wait_time = 0.0

    async def __aenter__(self):
        global _stats
        sem = _get_semaphore()

        # Check if queue is full
        waiters = MAX_ACTIVE_REQUESTS - sem._value + _stats.queued
        if waiters >= QUEUE_LIMIT:
            _stats.total_rejected += 1
            print(f"[RequestQueue] REJECTED: queue full ({_stats.queued}/{QUEUE_LIMIT})")
            return self

        _stats.queued += 1
        _stats.peak_queued = max(_stats.peak_queued, _stats.queued)

        start = time.time()
        try:
            await asyncio.wait_for(sem.acquire(), timeout=QUEUE_TIMEOUT)
            self.acquired = True
            self.wait_time = time.time() - start

            _stats.queued -= 1
            _stats.active += 1
            _stats.peak_active = max(_stats.peak_active, _stats.active)
            _stats.total_processed += 1

            if self.wait_time > 1.0:
                print(f"[RequestQueue] Slot acquired after {self.wait_time:.1f}s wait")

        except asyncio.TimeoutError:
            _stats.queued -= 1
            _stats.total_timeouts += 1
            print(f"[RequestQueue] TIMEOUT: waited {QUEUE_TIMEOUT}s")

        return self

    async def __aexit__(self, *args):
        global _stats
        if self.acquired:
            _stats.active -= 1
            _get_semaphore().release()


def get_queue_stats() -> dict:
    """Return current queue metrics."""
    return {
        "active_requests": _stats.active,
        "queued_requests": _stats.queued,
        "max_active": MAX_ACTIVE_REQUESTS,
        "max_queue": QUEUE_LIMIT,
        "total_processed": _stats.total_processed,
        "total_rejected": _stats.total_rejected,
        "total_timeouts": _stats.total_timeouts,
        "peak_active": _stats.peak_active,
        "peak_queued": _stats.peak_queued,
    }
