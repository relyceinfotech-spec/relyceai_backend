"""
Compatibility registry for agent tools.

This package intentionally mirrors the real implementations in
`app.agent.tool_executor` so there is only one behavioral source of truth.
Importing this package registers the compatibility wrappers below.
"""

from app.agent.tools.search import _tool_search_web, _tool_search_news
from app.agent.tools.finance import _tool_search_currency
from app.agent.tools.documents import _tool_search_documents, _tool_summarize_url
from app.agent.tools.local_ops import (
    _tool_get_current_time,
    _tool_pdf_maker,
    _tool_extract_entities,
    _tool_validate_code,
    _tool_generate_tests,
    _tool_execute_code,
    _tool_read_file,
    _tool_calculate,
    _tool_retrieve_knowledge,
)

__all__ = [
    "_tool_search_web",
    "_tool_search_news",
    "_tool_search_currency",
    "_tool_search_documents",
    "_tool_summarize_url",
    "_tool_get_current_time",
    "_tool_pdf_maker",
    "_tool_extract_entities",
    "_tool_validate_code",
    "_tool_generate_tests",
    "_tool_execute_code",
    "_tool_read_file",
    "_tool_calculate",
    "_tool_retrieve_knowledge",
]
