"""Streaming output moderation buffer.

Ensures content is moderated before any token is emitted to clients.
Also suppresses obvious internal prelude noise at stream start.
"""
from __future__ import annotations

import re
from app.safety.content_policy import classify_nsfw


class StreamingOutputModerator:
    def __init__(self, holdback_chars: int = 120, query: str = ""):
        self.holdback_chars = max(20, holdback_chars)
        self._pending = ""
        self._blocked = False
        self._blocked_reason = ""
        self._query_terms = self._extract_terms(query)
        self._start_noise_filtered = False

    def _extract_terms(self, text: str):
        return {t for t in re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())}

    def _looks_like_internal_noise(self, text: str) -> bool:
        low = (text or "").lower()
        markers = [
            "cross-device", "websocket", "transaction", "graph", "node", "tool_call",
            "assistant: first", "you have completed all required execution steps",
            "execution sequence", "rollback", "prompt injection", "system prompt",
        ]
        if not any(m in low for m in markers):
            return False
        if not self._query_terms:
            return True
        text_terms = self._extract_terms(text)
        overlap = len(self._query_terms.intersection(text_terms))
        return overlap == 0

    def ingest(self, token: str) -> str:
        """Add new token and return safe text chunk to emit now."""
        if self._blocked:
            return ""

        self._pending += token or ""

        # Early prelude-noise suppression (only once, near stream start).
        if (not self._start_noise_filtered) and 20 <= len(self._pending) <= 260:
            if self._looks_like_internal_noise(self._pending):
                self._pending = ""
                self._start_noise_filtered = True
                return ""

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
