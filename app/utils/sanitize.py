"""
HTML sanitization helpers for chat content.
"""
from __future__ import annotations

from typing import Optional
import html

from app.config import SANITIZE_CHAT_HTML

try:
    import bleach
except Exception:  # pragma: no cover - optional dependency
    bleach = None

_ALLOWED_TAGS = [
    "b", "strong", "i", "em", "u", "code", "pre", "br",
    "p", "ul", "ol", "li", "blockquote"
]


def sanitize_message_content(content: Optional[str]) -> str:
    if content is None:
        return ""
    # For LLM chat history, we want to store RAW markdown to avoid double-encoding code blocks.
    # The frontend is responsible for rendering this safely (ReactMarkdown).
    # If we escape it here (e.g. < to &lt;), and then the frontend renders it, 
    # code blocks will show &lt; instead of <.
    return content

    # Previous logic was damaging code blocks:
    # if not SANITIZE_CHAT_HTML:
    #     return content
    # text = content.replace("\x00", "")
    # if bleach is None:
    #     return html.escape(text)
    # return bleach.clean(text, tags=_ALLOWED_TAGS, attributes={}, strip=True)

