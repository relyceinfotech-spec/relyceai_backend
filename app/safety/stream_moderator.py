"""Streaming output moderation buffer.

Ensures content is moderated before any token is emitted to clients.
"""
from __future__ import annotations

from app.safety.content_policy import classify_nsfw


class StreamingOutputModerator:
    def __init__(self, holdback_chars: int = 120):
        self.holdback_chars = max(20, holdback_chars)
        self._pending = ""
        self._blocked = False
        self._blocked_reason = ""

    def ingest(self, token: str) -> str:
        """Add new token and return safe text chunk to emit now."""
        if self._blocked:
            return ""

        self._pending += token or ""
        blocked, reason = classify_nsfw(self._pending)
        if blocked:
            self._blocked = True
            self._blocked_reason = reason
            self._pending = ""
            return "\n\n[Content removed by safety filter.]"

        if len(self._pending) <= self.holdback_chars:
            return ""

        flush_upto = len(self._pending) - self.holdback_chars
        out = self._pending[:flush_upto]
        self._pending = self._pending[flush_upto:]
        return out

    def finalize(self) -> str:
        """Flush remaining safe text at stream end."""
        if self._blocked:
            return ""

        blocked, _ = classify_nsfw(self._pending)
        if blocked:
            self._blocked = True
            self._pending = ""
            return "\n\n[Content removed by safety filter.]"

        out = self._pending
        self._pending = ""
        return out

    @property
    def blocked(self) -> bool:
        return self._blocked

    @property
    def blocked_reason(self) -> str:
        return self._blocked_reason
