from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.config import FAST_MODEL, CODING_MODEL, UI_MODEL, LLM_MODEL, REASONING_EFFORT
from app.llm.router import should_use_thinking_model
from app.chat.mode_mapper import normalize_chat_mode


TEMPERATURE_MAP = {
    "general": 0.65,
    "coding": 0.2,
    "business": 0.4,
    "ecommerce": 0.55,
    "creative": 0.9,
    "music": 0.95,
    "legal": 0.15,
    "health": 0.3,
    "education": 0.5,
}

LOGIC_CODING_INTENTS = {
    "debugging",
    "system_design",
    "coding_complex",
    "coding_simple",
    "code_explanation",
    "sql",
    "ui_implementation",
    "ui_demo_html",
    "ui_react",
}

UI_SUB_INTENTS = {
    "ui_implementation",
    "ui_demo_html",
    "ui_react",
}

MAX_VERIFY_TIME = 1.5
VERIFY_COMPLEXITY_THRESHOLD = 0.65
CREATIVE_INTENTS = {
    "content_creation",
    "ui_design",
    "ui_strategy",
}

REASONING_SUB_INTENTS = {
    "coding_simple",
    "code_explanation",
    "coding_complex",
    "debugging",
    "system_design",
    "sql",
    "ui_implementation",
    "ui_demo_html",
    "ui_react",
    "ui_design",
    "ui_strategy",
    "reasoning",
}

REASONING_VISIBLE_SUB_INTENTS = REASONING_SUB_INTENTS


@dataclass
class RequestTrace:
    """Trace object for request lifecycle debugging & auditing."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    intent: str = ""
    sub_intent: str = ""
    model_chain: list = field(default_factory=list)
    is_thinking_pass: bool = False
    confidence_gating_skipped: bool = False
    fallback_triggered: bool = False

    def log(self, stage: str, detail: str = ""):
        print(f"[TRACE:{self.trace_id}] {stage}: {detail}")

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "intent": self.intent,
            "sub_intent": self.sub_intent,
            "model_chain": self.model_chain,
            "thinking_pass": self.is_thinking_pass,
            "gating_skipped": self.confidence_gating_skipped,
            "fallback": self.fallback_triggered,
        }


def _resolve_temperature(personality: Optional[Dict], fallback_specialty: str = "general") -> float:
    raw_temp = None
    if personality and isinstance(personality.get("temperature"), (int, float)):
        raw_temp = personality.get("temperature")

    if raw_temp is None:
        raw_temp = TEMPERATURE_MAP.get(fallback_specialty, TEMPERATURE_MAP["general"])

    return max(0, min(1, raw_temp))


def _get_max_tokens_for_sub_intent(sub_intent: str) -> Optional[int]:
    if sub_intent == "ui_demo_html":
        return 3200
    if sub_intent in ["ui_react", "ui_implementation"]:
        return 3600
    return None


def _get_reasoning_config(sub_intent: str, user_settings: Optional[Dict]) -> Optional[Dict[str, Any]]:
    if sub_intent not in REASONING_SUB_INTENTS:
        return None
    visibility = ((user_settings or {}).get("personalization") or {}).get("thinkingVisibility", "auto")
    exclude = visibility == "off"
    effort = (REASONING_EFFORT or "low").lower()
    allowed = {"xhigh", "high", "medium", "low", "minimal", "none"}
    if effort not in allowed:
        effort = "low"
    if effort == "none":
        return {"exclude": True}
    return {"effort": effort, "exclude": exclude, "enabled": True}


def _apply_reasoning(create_kwargs: Dict[str, Any], model_to_use: str, sub_intent: str, user_settings: Optional[Dict]) -> None:
    if model_to_use == LLM_MODEL:
        return

    extra_body = create_kwargs.get("extra_body") or {}
    extra_body["provider"] = {"sort": "throughput"}

    reasoning = _get_reasoning_config(sub_intent, user_settings)
    if reasoning:
        extra_body["reasoning"] = reasoning

    create_kwargs["extra_body"] = extra_body

    if create_kwargs.get("stream"):
        create_kwargs["stream_options"] = {"include_usage": True}


def _should_show_reasoning_panel(mode: str, sub_intent: str, user_settings: Optional[Dict] = None) -> bool:
    mode = normalize_chat_mode(mode)
    if mode != "smart":
        return False
    visibility = ((user_settings or {}).get("personalization") or {}).get("thinkingVisibility", "auto")
    if visibility == "on":
        return True
    if visibility == "off":
        return False
    if sub_intent == "casual_chat":
        return False
    return sub_intent in REASONING_VISIBLE_SUB_INTENTS


def _user_wants_sources(user_query: str) -> bool:
    q = (user_query or "").lower()
    keywords = ["source", "sources", "citation", "cite", "references", "reference", "links", "link"]
    return any(k in q for k in keywords)


def _is_casual_query(user_query: str) -> bool:
    q = (user_query or "").strip().lower()
    return bool(re.match(r"^(hi|hello|hey|yo|sup|how are you|how r u|how are u|whats up)[!.?\s]*$", q))


def _is_profile_query(user_query: str) -> bool:
    q = (user_query or "").strip().lower()
    return bool(re.search(r"\b(my name|about me|who am i|remember me|tell about me)\b", q))


def _is_recall_query(user_query: str) -> bool:
    q = (user_query or "").strip().lower()
    patterns = [
        r"\bwhat (?:did|do) i (?:ask|say)\b",
        r"\bmy first (?:question|message|ask)\b",
        r"\bfirst thing i asked\b",
        r"\bwhat was my first\b",
        r"\bearlier (?:question|message|ask)\b",
        r"\bprevious (?:question|message|ask)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _build_recall_response(context_messages: Optional[List[Dict]], user_query: str) -> Optional[str]:
    if not context_messages:
        return "I cannot find earlier messages in this chat yet."

    user_msgs = []
    for msg in context_messages:
        if str(msg.get("role", "")).lower() != "user":
            continue
        content = str(msg.get("content", "")).strip()
        if content:
            user_msgs.append(content)

    if not user_msgs:
        return "I cannot find earlier user messages in this chat yet."

    q = (user_query or "").strip().lower()
    is_first = bool(re.search(r"\b(first|starting|initial)\b", q))
    is_last = bool(re.search(r"\b(last|latest|previous)\b", q))

    if is_first:
        return f"Your first message in this chat was: \"{user_msgs[0]}\""

    if is_last:
        return f"Your last message before this was: \"{user_msgs[-1]}\""

    recent = user_msgs[-5:]
    lines = [f"{idx + 1}. {msg}" for idx, msg in enumerate(recent)]
    return "Here are your recent messages in this chat:\n" + "\n".join(lines)


def _wants_single_file_html(user_query: str) -> bool:
    q = (user_query or "").lower()
    triggers = [
        "single html",
        "single file",
        "one html",
        "one file",
        "in one file",
        "full code in one html",
        "send as single code",
        "all in one html",
    ]
    return any(t in q for t in triggers)


def _single_file_html_instruction() -> str:
    return (
        "\n\nSTRICT OUTPUT RULES FOR THIS REQUEST:\n"
        "- Return exactly ONE file: index.html\n"
        "- Include CSS inside a <style> tag and JS inside a <script> tag in the same file\n"
        "- Do NOT output style.css or script.js\n"
        "- Output one fenced code block only, language html\n"
        "- Ensure valid tags: <title> and <link rel=\"stylesheet\" ...> must be complete\n"
    )


def _is_factual_lookup_query(query: str) -> bool:
    q = (query or "").lower()
    patterns = [
        "who is", "founder", "ceo", "board", "affiliation", "cbse", "matric",
        "address", "phone", "website", "price", "current", "latest", "today",
        "how many", "when was", "incorporated", "din", "cin",
    ]
    return any(p in q for p in patterns)


def _resolve_runtime_intent(mode: str, routed_intent: str, query: str = "") -> str:
    mode = normalize_chat_mode(mode)
    if mode in {"agent", "research_pro"}:
        return "AGENT"
    if mode == "smart":
        if _is_factual_lookup_query(query):
            return "AGENT"
        if routed_intent in {"DEEP_SEARCH", "EXTERNAL"}:
            return routed_intent
        return "INTERNAL"
    return routed_intent or "INTERNAL"


def get_model_for_intent(mode: str, sub_intent: str, personality: Optional[Dict] = None, query: str = "") -> Tuple[str, str, Optional[float], bool, str]:
    mode = normalize_chat_mode(mode)
    needs_thinking = should_use_thinking_model(query, sub_intent)

    if personality and (personality.get("id") == "coding_buddy" or personality.get("name") == "Coding Buddy"):
        if sub_intent in UI_SUB_INTENTS:
            return ("openrouter", UI_MODEL, 0.2, False, "coding_buddy_ui_override")
        if sub_intent in CREATIVE_INTENTS:
            return ("openrouter", CODING_MODEL, None, False, "coding_buddy_override")
        if sub_intent in ["debugging", "system_design", "coding_complex"] and needs_thinking:
            return ("openrouter", CODING_MODEL, None, True, "coding_buddy_override_thinking")
        temp = _resolve_temperature(personality, "coding")
        return ("openrouter", CODING_MODEL, temp, False, "coding_buddy_override")

    reasoning_model = FAST_MODEL

    if sub_intent == "system_design":
        return ("openrouter", reasoning_model, None, False, "system_design_reasoning")
    if sub_intent in UI_SUB_INTENTS:
        return ("openrouter", UI_MODEL, 0.2, False, "ui_override")
    if sub_intent == "debugging":
        return ("openrouter", reasoning_model, None, needs_thinking, "debugging_reasoning")
    if sub_intent in LOGIC_CODING_INTENTS:
        return ("openrouter", FAST_MODEL, None, False, "coding_fast")
    if sub_intent in CREATIVE_INTENTS:
        return ("openrouter", FAST_MODEL, None, False, "creative_override")
    if sub_intent in ["analysis", "research", "reasoning", "web_analysis", "web_factual"]:
        return ("openrouter", reasoning_model, 0.2, needs_thinking, "research_reasoning")

    fallback_specialty = personality.get("specialty", "general") if personality else "general"
    if fallback_specialty == "coding":
        return ("openrouter", CODING_MODEL, None, False, "persona_specialty_override")
    temp = _resolve_temperature(personality, fallback_specialty)
    return ("openrouter", FAST_MODEL, temp, False, "default_persona_model")

