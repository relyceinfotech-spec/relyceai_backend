from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from app.chat.mode_mapper import normalize_chat_mode


@dataclass
class CapabilityRoute:
    capability: str
    reason: str


class CapabilityRouter:
    """Deterministic capability classifier for platform routing."""

    def route(self, user_query: str, chat_mode: str = "smart", capability_hint: Optional[str] = None) -> CapabilityRoute:
        if capability_hint:
            return CapabilityRoute(capability=capability_hint.strip().lower(), reason="explicit_hint")

        mode = normalize_chat_mode(chat_mode)
        q = (user_query or "").strip().lower()

        if mode in {"smart", "agent", "research_pro"}:
            return CapabilityRoute(capability="research", reason=f"chat_mode:{mode}")
        if mode in {"coding", "code"}:
            return CapabilityRoute(capability="coding", reason=f"chat_mode:{mode}")
        if mode in {"documents", "document", "doc"}:
            return CapabilityRoute(capability="documents", reason=f"chat_mode:{mode}")
        if mode in {"automation", "workflow"}:
            return CapabilityRoute(capability="automation", reason=f"chat_mode:{mode}")

        if re.search(r"\b(pdf|document|docx|file|upload|summarize this)\b", q):
            return CapabilityRoute(capability="documents", reason="query_document_signal")
        if re.search(r"\b(code|debug|python|javascript|typescript|sql|api|function|bug|compile)\b", q):
            return CapabilityRoute(capability="coding", reason="query_coding_signal")
        if re.search(r"\b(automate|workflow|schedule|cron|daily job|pipeline)\b", q):
            return CapabilityRoute(capability="automation", reason="query_automation_signal")
        if re.search(r"\b(roi|revenue|sales|profit|business|market|funnel|kpi|pricing strategy)\b", q):
            return CapabilityRoute(capability="research", reason="query_business_signal")
        if re.search(r"\b(research|analyze|compare|verify|sources|citations?)\b", q):
            return CapabilityRoute(capability="research", reason="query_research_signal")

        return CapabilityRoute(capability="chat", reason="default_chat")
