"""
Streaming Metrics — Track TTFT, tokens/sec, and stream quality.
Attaches to the streaming pipeline to measure real user experience.
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamSession:
    """Metrics for a single streaming response."""
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    token_count: int = 0
    char_count: int = 0
    interrupted: bool = False

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time and self.start_time:
            return (self.first_token_time - self.start_time) * 1000
        return 0.0

    @property
    def tokens_per_sec(self) -> float:
        if self.end_time and self.first_token_time:
            duration = self.end_time - self.first_token_time
            if duration > 0:
                return self.token_count / duration
        return 0.0

    @property
    def total_duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class StreamMetrics:
    """
    Aggregate streaming performance metrics.
    Thread-safe singleton.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.total_streams: int = 0
        self.total_tokens: int = 0
        self.total_ttft_ms: float = 0.0
        self.total_duration_ms: float = 0.0
        self.interruptions: int = 0
        self._recent_ttft: list = []  # last 50
        self._recent_tps: list = []   # last 50

    def record_stream(self, session: StreamSession):
        """Record a completed stream session."""
        with self._lock:
            self.total_streams += 1
            self.total_tokens += session.token_count

            if session.ttft_ms > 0:
                self.total_ttft_ms += session.ttft_ms
                self._recent_ttft.append(session.ttft_ms)
                if len(self._recent_ttft) > 50:
                    self._recent_ttft.pop(0)

            if session.tokens_per_sec > 0:
                self._recent_tps.append(session.tokens_per_sec)
                if len(self._recent_tps) > 50:
                    self._recent_tps.pop(0)

            self.total_duration_ms += session.total_duration_ms

            if session.interrupted:
                self.interruptions += 1

    def get_metrics(self) -> dict:
        """Return streaming performance snapshot."""
        with self._lock:
            avg_ttft = (
                sum(self._recent_ttft) / len(self._recent_ttft)
                if self._recent_ttft else 0.0
            )
            avg_tps = (
                sum(self._recent_tps) / len(self._recent_tps)
                if self._recent_tps else 0.0
            )
            avg_duration = (
                self.total_duration_ms / self.total_streams
                if self.total_streams else 0.0
            )
            interrupt_rate = (
                self.interruptions / self.total_streams
                if self.total_streams else 0.0
            )

            return {
                "total_streams": self.total_streams,
                "total_tokens_streamed": self.total_tokens,
                "avg_ttft_ms": round(avg_ttft, 1),
                "avg_tokens_per_sec": round(avg_tps, 1),
                "avg_duration_ms": round(avg_duration, 1),
                "interruptions": self.interruptions,
                "interrupt_rate": round(interrupt_rate, 4),
                "p50_ttft_ms": round(sorted(self._recent_ttft)[len(self._recent_ttft)//2], 1) if self._recent_ttft else 0.0,
            }

    def reset(self):
        with self._lock:
            self.total_streams = 0
            self.total_tokens = 0
            self.total_ttft_ms = 0.0
            self.total_duration_ms = 0.0
            self.interruptions = 0
            self._recent_ttft.clear()
            self._recent_tps.clear()


# Module singleton
_instance: Optional[StreamMetrics] = None


def get_stream_metrics() -> StreamMetrics:
    global _instance
    if _instance is None:
        _instance = StreamMetrics()
    return _instance
