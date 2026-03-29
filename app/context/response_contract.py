"""Backward-compatible response contract wrapper.

Canonical implementation now lives in app.response_system.adapter.
"""

from app.response_system.adapter import normalize_chat_response, build_final_answer_payload
from app.response_system.schema import SCHEMA_VERSION, ALLOWED_BLOCK_TYPES

__all__ = ["normalize_chat_response", "build_final_answer_payload", "SCHEMA_VERSION", "ALLOWED_BLOCK_TYPES"]
