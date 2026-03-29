"""Universal response schema package."""

from .schema import SCHEMA_VERSION, ALLOWED_BLOCK_TYPES
from .adapter import normalize_chat_response

__all__ = ["SCHEMA_VERSION", "ALLOWED_BLOCK_TYPES", "normalize_chat_response"]
