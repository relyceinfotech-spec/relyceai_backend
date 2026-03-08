"""
Relyce AI - LLM Processor
Handles message processing with streaming support
Includes legacy prompts and routing logic consolidated into app/llm
"""
import asyncio
import json
import re
import html
from typing import AsyncGenerator, List, Dict, Any, Optional
from openai import OpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL, FAST_MODEL, CODING_MODEL, UI_MODEL, REASONING_EFFORT, ERNIE_THINKING_MODEL, SERPER_TOOLS
from app.llm.router import (
    select_tools_for_mode,
    analyze_and_route_query,
    execute_serper_batch,
    execute_serper_batch_sync,
    get_system_prompt_for_mode,
    get_tools_for_mode,
    get_system_prompt_for_personality,
    get_internal_system_prompt_for_personality,
    INTERNAL_SYSTEM_PROMPT,
    get_openai_client,
    get_openrouter_client,
    TONE_MAP,
    EMOTIONAL_BLOCK,
)
from app.llm.routing_log import log_routing_decision
from app.llm.guards import normalize_user_query, build_guard_system_messages
from app.llm.emotion_engine import emotion_engine
from app.llm.feedback_engine import feedback_engine
from app.llm.strategy_memory import strategy_memory
from app.llm.prompt_optimizer import prompt_optimizer
from app.llm.user_profiler import user_profiler
from app.llm.debug_agent import debug_agent
from app.llm.context_optimizer import context_optimizer
from app.llm.safe_agent import analyze_constraints, check_semantic_loss, ImpossibilityTier, DEBUG_SAFE_AGENT
from app.llm import output_validator
from app.chat.session_rules import increment_turn
from app.agent.agent_orchestrator import run_agent_pipeline, build_agent_system_prompt
from app.agent.self_monitor import evaluate_response as monitor_evaluate_response
from app.agent.static_validator import static_validation
from app.agent.repair_engine import repair_cycle as run_repair_cycle, generate_repair_strategy, build_repair_prompt
from app.agent.strategy_engine import merge_reasoning_tracks
from app.llm.skill_router_runtime import record_skill_outcome, get_last_skill_trace
from app.llm.token_counter import estimate_tokens as rough_estimate_tokens
from app.safety.content_policy import classify_nsfw

active_executions = {}
_session_followup_cache: Dict[str, List[str]] = {}

async def _execution_registry_cleanup():
    """Background task to clear stalled executions older than 300 seconds."""
    import time
    while True:
        await asyncio.sleep(60)
        current_time = time.time()
        for k, v in list(active_executions.items()):
            if current_time - v.get("created_at", 0) > 300:
                active_executions.pop(k, None)

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
NORMAL_GENERIC_FORMAT_RULES = """
NORMAL MODE (GENERIC) OUTPUT TEMPLATE:
- Start with a short direct answer (1-2 sentences).
- Then use markdown headings and concise bullets.
- Keep sections scannable; avoid long text walls.
- Do NOT output self-verification/meta-audit sections.
- Do NOT mention knowledge cutoff/tool limitations unless the user explicitly asks.

Use this structure when relevant:
## Overview
## Key Points
## Example (if relevant)
## Conclusion
"""

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

RAG_INSTRUCTION = """
[CRITICAL: DOCUMENT USAGE]
You have access to locally uploaded documents for this session. 
- If the user asks a question related to these documents, provide a detailed answer based strictly on the document content.
- After your main answer, add a section: "--- ?? Related Insights from Document ---" and include 2-3 other relevant or interesting points from the uploaded file that the user might find useful.
- If the user's question is NOT related to the documents, answer normally using your general knowledge and DO NOT include the insights section.
- Always cite the document using [Document].
"""

# ============================================
# PRODUCTION HARDENING: Trace ID & Gating
# ============================================
import uuid
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class RequestTrace:
    """Trace object for request lifecycle debugging & auditing"""
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
            "fallback": self.fallback_triggered
        }

def should_use_thinking_model(query: str, sub_intent: str) -> bool:
    """
    Confidence gating: Decide if query needs Ernie thinking pass.
    Disabling 2-pass thinking to rely on OpenRouter's native reasoning feature.
    """
    return False

def _resolve_temperature(personality: Optional[Dict], fallback_specialty: str = "general") -> float:
    """
    Resolve a temperature using an explicit personality temperature when provided,
    otherwise fall back to a specialty-based default.
    """
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
    """
    Build OpenRouter reasoning config for coding/UI-heavy generation.
    Returns None when reasoning should not be requested.
    """
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
    """Attach OpenRouter reasoning & throughput config when applicable."""
    if model_to_use == LLM_MODEL:
        return
        
    extra_body = create_kwargs.get("extra_body") or {}
    
    # Force throughput sorting to optimize TTFT globally for OpenRouter
    extra_body["provider"] = {"sort": "throughput"}
    
    reasoning = _get_reasoning_config(sub_intent, user_settings)
    if reasoning:
        extra_body["reasoning"] = reasoning
        
    create_kwargs["extra_body"] = extra_body
    
    if create_kwargs.get("stream"):
        create_kwargs["stream_options"] = {"include_usage": True}

REASONING_VISIBLE_SUB_INTENTS = REASONING_SUB_INTENTS

def _should_show_reasoning_panel(mode: str, sub_intent: str, user_settings: Optional[Dict] = None) -> bool:
    if mode != "normal":
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
    return bool(
        re.match(
            r"^(hi|hello|hey|yo|sup|how are you|how r u|how are u|whats up)[!.?\s]*$",
            q,
        )
    )


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
    for m in context_messages:
        if str(m.get("role", "")).lower() != "user":
            continue
        content = str(m.get("content", "")).strip()
        if content:
            user_msgs.append(content)

    if not user_msgs:
        return "I cannot find earlier user messages in this chat yet."

    q = (user_query or "").strip().lower()
    is_first = bool(re.search(r"\b(first|starting|initial)\b", q))
    is_last = bool(re.search(r"\b(last|latest|previous)\b", q))

    if is_first:
        first_msg = user_msgs[0]
        return f"Your first message in this chat was: \"{first_msg}\""

    if is_last:
        last_msg = user_msgs[-1]
        return f"Your last message before this was: \"{last_msg}\""

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
    """
    Execution intent policy by mode:
    - agent/business: always structured agent path
    - normal: prefer fast direct LLM path, keep deep-search for external lookups
    """
    if mode in {"agent", "business"}:
        return "AGENT"
    if mode == "normal":
        if _is_factual_lookup_query(query):
            return "AGENT"
        if routed_intent in {"DEEP_SEARCH", "EXTERNAL"}:
            return routed_intent
        return "INTERNAL"
    return routed_intent or "INTERNAL"

def get_model_for_intent(mode: str, sub_intent: str, personality: Optional[Dict] = None, query: str = "") -> Tuple[str, str, Optional[float], bool, str]:
    """
    Determine which client/model to use based on mode, sub_intent, and personality.
    Now includes confidence gating for thinking model.
    
    Returns: (client_type, model_id, temperature, needs_thinking, reason)
    - client_type: "openai" or "openrouter"
    - model_id: The model to use
    - temperature: Suggested temperature
    - needs_thinking: Whether to use 2-pass pipeline
    - reason: routing reason string
    """
    # Routing priority order:
    # 1. Security override (enforced elsewhere, placeholder)
    # 2. Coding override
    # 3. System-design override
    # 4. Mode-based override
    # 5. Default persona model

    # Check if thinking is warranted (confidence gating)
    needs_thinking = should_use_thinking_model(query, sub_intent)

    # Coding Buddy: always use the coding model (optionally allow thinking for complex tasks)
    if personality and (personality.get("id") == "coding_buddy" or personality.get("name") == "Coding Buddy"):
        if sub_intent in UI_SUB_INTENTS:
            return ("openrouter", UI_MODEL, 0.2, False, "coding_buddy_ui_override")
        if sub_intent in CREATIVE_INTENTS:
            return ("openrouter", CODING_MODEL, None, False, "coding_buddy_override")
        if sub_intent in ["debugging", "system_design", "coding_complex"] and needs_thinking:
            return ("openrouter", CODING_MODEL, None, True, "coding_buddy_override_thinking")
        temp = _resolve_temperature(personality, "coding")
        return ("openrouter", CODING_MODEL, temp, False, "coding_buddy_override")

    # Item #9: Cost-safe model routing
    # Reasoning model for heavy tasks, flash for everything else
    REASONING_MODEL = "deepseek/deepseek-chat"
    
    # System-design override ? reasoning model
    if sub_intent == "system_design":
        return ("openrouter", REASONING_MODEL, None, False, "system_design_reasoning")

    # UI intents: prefer stronger UI model with lower temperature
    if sub_intent in UI_SUB_INTENTS:
        return ("openrouter", UI_MODEL, 0.2, False, "ui_override")

    # Debugging ? reasoning model
    if sub_intent == "debugging":
        return ("openrouter", REASONING_MODEL, None, needs_thinking, "debugging_reasoning")

    # Coding intents: use fast model (cost-safe)
    if sub_intent in LOGIC_CODING_INTENTS:
        return ("openrouter", FAST_MODEL, None, False, "coding_fast")

    # Business mode
    if mode == "business":
        return ("openrouter", LLM_MODEL, 0.4, False, "mode_override")

    # UI/creative requests: flash (cheap)
    if sub_intent in CREATIVE_INTENTS:
        return ("openrouter", FAST_MODEL, None, False, "creative_override")
    
    # Analysis/Research ? reasoning model (Grounded Temperature 0.2)
    if sub_intent in ["analysis", "research", "reasoning", "web_analysis", "web_factual"]:
        return ("openrouter", REASONING_MODEL, 0.2, needs_thinking, "research_reasoning")
    
    # Default: General/Personality uses Gemini Flash Lite
    fallback_specialty = personality.get("specialty", "general") if personality else "general"
    # Coding-specialty personalities use GLM4.7 Flash directly
    if fallback_specialty == "coding":
        return ("openrouter", CODING_MODEL, None, False, "persona_specialty_override")
    temp = _resolve_temperature(personality, fallback_specialty)
    return ("openrouter", FAST_MODEL, temp, False, "default_persona_model")

class LLMProcessor:
    """
    Main LLM processor class that handles all chat modes.
    Consolidated logic from legacy mode scripts
    """
    
    def __init__(self):
        self.model = LLM_MODEL

    @staticmethod
    def _inject_guard_messages(messages: List[Dict], user_query: Optional[str]) -> List[Dict]:
        guards = build_guard_system_messages(user_query or "")
        if not guards or not messages:
            return messages
        guarded = list(messages)
        insert_at = next((i for i, m in enumerate(guarded) if m.get("role") == "user"), len(guarded))
        for guard in reversed(guards):
            guarded.insert(insert_at, {"role": "system", "content": guard})
        return guarded

    def _guard_kwargs(self, kwargs: Dict[str, Any], user_query: Optional[str]) -> Dict[str, Any]:
        if "messages" in kwargs:
            kwargs["messages"] = self._inject_guard_messages(kwargs["messages"], user_query)
        return kwargs

    def _get_sampling(self, personality: Optional[Dict], sub_intent: Optional[str] = None) -> Dict[str, Any]:
        """
        Backend-side authority: clamp and cap temperature and add specialty-based sampling hints.
        """
        if not personality:
            return {}

        specialty = personality.get("specialty") or "general"
        raw_temp = personality.get("temperature")

        if not isinstance(raw_temp, (int, float)):
            raw_temp = TEMPERATURE_MAP.get(specialty, TEMPERATURE_MAP["general"])

        # Clamp defensively
        temp = max(0, min(1, raw_temp))

        is_creative_intent = sub_intent in CREATIVE_INTENTS if sub_intent else False
        is_logic_coding_intent = sub_intent in LOGIC_CODING_INTENTS if sub_intent else False
        is_ui_intent = sub_intent in UI_SUB_INTENTS if sub_intent else False

        # Hard caps for safety-critical domains
        if specialty == "legal":
            temp = min(temp, 0.3)
        elif is_ui_intent:
            temp = min(temp, 0.2)
        elif is_logic_coding_intent or (specialty == "coding" and not is_creative_intent):
            temp = min(temp, 0.3)

        sampling: Dict[str, Any] = {}
        if not is_creative_intent:
            sampling["temperature"] = temp

        # Specialty-specific knobs
        if is_logic_coding_intent or specialty == "coding":
            sampling.update({"top_p": 0.9, "frequency_penalty": 0, "presence_penalty": 0})
        elif is_creative_intent or specialty == "creative":
            sampling.update({"top_p": 1, "presence_penalty": 0.6})

        return sampling
    
    def _sanitize_output_text(self, text: str, allow_double_hyphen: bool = False, trim_outer: bool = True) -> str:
        """
        Normalize punctuation to avoid em-dashes/double-hyphen in outputs.
        Also unescapes HTML entities to ensure code renders correctly.
        """
        if not text:
            return text
            
        text = html.unescape(text)
        is_html_document = bool(re.search(r"<!doctype\s+html|<html[\s>]|<head>|<body>", text, flags=re.IGNORECASE))
        if is_html_document:
            # Preserve HTML docs exactly; tool-artifact stripping can corrupt valid tags like <title>/<link>.
            return text
        
        sanitized = text
        sanitized = sanitized.replace("\u2014", " - ").replace("\u2013", " - ")
        # Remove mojibake emoji artifacts (e.g., "ðŸ¤”", "ðŸš€") caused by encoding mismatch.
        sanitized = re.sub(r"ðŸ[\x80-\xBF]{2,4}", "", sanitized)
        sanitized = re.sub(r"Ã[\x80-\xBF]+", "", sanitized)
        sanitized = re.sub(r"^\s*TOOL\b[#:\-\s]*", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*TOOL#\s*", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*_?CALL\s*:\s*.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*TOOL_CALL\s*:\s*.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*Assistant:\s*First,.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*First,\s*the user.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*You have completed all required execution steps\.?\s*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*WEB SEARCH\s*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*Search Result\s*\d+\s*[:\-].*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        # Strip raw tool dump artifacts so users only see the final answer.
        sanitized = re.sub(r"^\s*(Title|Snippet Details|Link|Source Link|Copy text|Export)\s*[:\-]?.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*##\s*Verification.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*(Verification Report|Accuracy Assessment|Completeness Assessment|Uncertainties|Improvements Recommended).*", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*.*tools?\s+disabled.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"^\s*.*no external fetches possible.*$", "", sanitized, flags=re.IGNORECASE | re.MULTILINE)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

        return sanitized.strip() if trim_outer else sanitized

    def _fix_html_css_output(self, text: str) -> str:
        if not text:
            return text
        content = text
        
        content = re.sub(r'<!\s*-\s*-', '<!--', content)
        content = re.sub(r'-\s*-\s*>', '-->', content)
        content = re.sub(r'<!\s*-\s*-\s*\[', '<!--[', content)
        content = re.sub(r'\]\s*-\s*-\s*>', ']-->', content)
        
        def fix_css_property(match):
            indent = match.group(1)
            name = match.group(2)
            return f'{indent}--{name}:'
        content = re.sub(r'(\n\s*)-\s+([a-zA-Z][a-zA-Z0-9-]*)\s*:', fix_css_property, content)
        content = re.sub(r'(\s{2,})-\s+([a-zA-Z][a-zA-Z0-9-]*)\s*:', fix_css_property, content)
        
        content = re.sub(r'var\s*\(\s*-\s+', 'var(--', content)
        content = re.sub(r'var\s*\(\s*-\s*-\s*', 'var(--', content)
        
        content = re.sub(r'--\s+([a-zA-Z])', r'--\1', content)
        content = re.sub(r'--([a-zA-Z][a-zA-Z0-9-]*)\s+([a-zA-Z])', r'--\1\2', content)
        
        content = content.replace("group: card;", "")
        content = content.replace("group: ;", "")
        
        if "line-clamp:" not in content and "-webkit-line-clamp:" in content:
            content = re.sub(r'(\s*)-webkit-line-clamp:\s*([0-9]+);', r'\1line-clamp: \2;\n\1-webkit-line-clamp: \2;', content)
        
        return content


    def _format_user_answer(self, text: str, mode: str) -> str:
        """Normalize final user-facing output into clear concise markdown blocks."""
        cleaned = self._sanitize_output_text(text or "")
        if not cleaned:
            return cleaned

        # Preserve code-heavy answers.
        if "```" in cleaned:
            return cleaned

        # Keep already-structured markdown.
        if re.search(r"(?m)^##\s+", cleaned):
            return cleaned

        # Very short casual replies should stay natural.
        if len(cleaned.split()) <= 12 and "?" not in cleaned:
            return cleaned

        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
        if not parts:
            return cleaned

        direct = parts[0]
        details = []
        for p in parts[1:]:
            p2 = p.strip("-* \t\n\r")
            if p2:
                details.append(p2)
            if len(details) >= 4:
                break

        if not details:
            return f"## Direct Answer\n{direct}"

        bullets = "\n".join(f"- {d}" for d in details)
        return f"## Direct Answer\n{direct}\n\n## Key Details\n{bullets}"
    def _sanitize_followups(self, items: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: set = set()
        for q in (items or []):
            if not isinstance(q, str):
                continue
            q2 = re.sub(r"\s+", " ", q).strip(" -?\t\n\r")
            if not q2:
                continue
            words = q2.split()
            if len(words) < 6 or len(words) > 14:
                continue
            key = re.sub(r"[^a-z0-9 ]+", "", q2.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(q2)
            if len(cleaned) >= 3:
                break
        return cleaned

    def _extract_followups_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        section = re.search(r"(?:^|\n)##\s*Related Questions\s*\n(?P<body>[\s\S]*?)(?:\n##\s|$)", text, flags=re.IGNORECASE)
        body = section.group("body") if section else text
        candidates: List[str] = []
        for line in body.splitlines():
            m = re.match(r"^\s*(?:[-*]|\u2022)\s+(.+?)\s*$", line)
            if m:
                candidates.append(m.group(1).strip())
        return self._sanitize_followups(candidates)

    def _extract_action_chips_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        chips = re.findall(r"\[([^\]\n]{3,40})\]", text)
        out: List[str] = []
        seen: set = set()
        for chip in chips:
            c = re.sub(r"\s+", " ", chip).strip()
            if not c:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
            if len(out) >= 3:
                break
        return out

    def _build_followup_payload(
        self,
        response_text: str,
        mode: str,
        session_id: Optional[str],
        persist: bool = True,
    ) -> Dict[str, Any]:
        followups = self._extract_followups_from_text(response_text or "")
        if session_id:
            previous = _session_followup_cache.get(session_id, [])
            previous_set = set(previous)
            filtered = []
            for q in followups:
                key = re.sub(r"[^a-z0-9 ]+", "", q.lower())
                if key and key not in previous_set:
                    filtered.append(q)
            followups = filtered
            if persist and followups:
                updated = (previous + [re.sub(r"[^a-z0-9 ]+", "", q.lower()) for q in followups])[-24:]
                _session_followup_cache[session_id] = updated

        payload: Dict[str, Any] = {
            "followups": followups[:3],
            "followup_mode": mode,
        }
        chips = self._extract_action_chips_from_text(response_text or "")
        if chips:
            payload["action_chips"] = chips[:3]
        return payload

    def _estimate_query_complexity(self, query: str) -> float:
        q = (query or "").strip().lower()
        if not q:
            return 0.0
        words = len(q.split())
        markers = [
            "compare", "analyze", "architecture", "workflow", "strategy", "trade-off",
            "optimize", "multi", "step", "research", "evaluate", "design",
        ]
        hit = sum(1 for m in markers if m in q)
        score = min(1.0, (words / 40.0) + (hit * 0.12))
        return score

    def _should_run_verifier(self, mode: str, query: str, draft: str) -> bool:
        m = (mode or "normal").lower()
        if m not in {"business", "agent"}:
            return False
        if self._estimate_query_complexity(query) <= VERIFY_COMPLEXITY_THRESHOLD:
            return False
        # Budget guard for extra verify pass.
        verify_prompt_tokens = rough_estimate_tokens([
            {"role": "system", "content": "verify"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": (draft or "")[:4000]},
        ])
        return verify_prompt_tokens <= 4500

    async def _verify_response_quality(self, query: str, draft: str, mode: str) -> str:
        verifier_messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict verification assistant. Improve the draft only if needed. "
                    "Check factual consistency, contradictions, and missing key steps. "
                    "If draft is already good, return it unchanged. "
                    "Do not add meta commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query:\n{query}\n\n"
                    f"Draft Answer:\n{draft}\n\n"
                    "Return the final improved answer only."
                ),
            },
        ]

        verify_kwargs = {
            "model": FAST_MODEL,
            "messages": verifier_messages,
            "max_tokens": 700,
            "temperature": 0.1,
        }
        verify_kwargs = self._guard_kwargs(verify_kwargs, query)
        client = get_openrouter_client()
        resp = await client.chat.completions.create(**verify_kwargs)
        verified = (resp.choices[0].message.content or "").strip()
        return self._sanitize_output_text(verified) if verified else draft

    async def summarize_context(self, messages: List[Dict], existing_summary: str = "") -> str:
        """
        Summarize a list of messages into a concise context string.
        Used for Option-2 Context Strategy.
        Refines existing summary if provided.
        """
        if not messages and not existing_summary: return ""
        
        conversation_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        
        # improved prompt for cumulative memory
        prompt_content = ""
        if existing_summary:
            prompt_content += f"EXISTING SUMMARY:\n{existing_summary}\n\n"
        
        prompt_content += f"NEW MESSAGES TO INTEGRATE:\n{conversation_text}\n\n"
        
        prompt_content += (
            "INSTRUCTIONS:\n"
            "Update the structured memory using the new messages.\n"
            "Output must use EXACTLY these sections in this order:\n"
            "### User Constraints\n"
            "### Code Entities\n"
            "### Decisions Made\n"
            "### Open Tasks\n"
            "### Important Assumptions\n"
            "Under each heading, use bullet points. If no items, write 'None'.\n"
            "Keep items short. Preserve names of functions/classes/files/APIs exactly as seen.\n"
            "Do NOT add new info. Do NOT include any other text."
        )
        
        try:
            response = await get_openrouter_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception:
            return existing_summary # Return old summary if update fails

    
    async def process_agent_query(
        self,
        user_query: str,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        mode: str = "normal",
        sub_intent: str = "general",
        user_id: Optional[str] = None,
        emotions: List[str] = [],
        emotional_instruction: Optional[str] = None
    ) -> str:
        """
        Handle internal queries (greetings, math, code, logic).
        Now supports PERSONALITY customization.
        """
        user_query = normalize_user_query(user_query)
        if personality:
            system_prompt = get_internal_system_prompt_for_personality(personality, user_settings, mode=mode)
        else:
            from app.llm.router import _build_user_context_string
            system_prompt = INTERNAL_SYSTEM_PROMPT + _build_user_context_string(user_settings)
            if mode == "normal":
                from app.llm.router import NORMAL_MARKDOWN_POLISH
                system_prompt = f"{system_prompt}\n{NORMAL_MARKDOWN_POLISH}"

        # Apply specialized internal prompt overlays for coding/technical intents
        from app.llm.router import INTERNAL_MODE_PROMPTS
        if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
            system_prompt = f"{system_prompt}\n\n**MODE SWITCH: {sub_intent.upper()}**\n{INTERNAL_MODE_PROMPTS[sub_intent]}"
        elif mode == "normal" and sub_intent == "general":
            system_prompt = f"{system_prompt}\n\n{NORMAL_GENERIC_FORMAT_RULES}"
        if sub_intent == "ui_demo_html" and _wants_single_file_html(user_query):
            system_prompt = f"{system_prompt}{_single_file_html_instruction()}"
        # Apply Emotion/Tone Instruction
        if emotional_instruction:
            system_prompt = f"{system_prompt}\n\n{emotional_instruction}"
        else:
            # Fallback to multi-label static map
            for emotion in emotions:
                if emotion in TONE_MAP:
                    system_prompt = f"{system_prompt}\n\n{TONE_MAP[emotion]}"
             
        # Check for Coding Buddy override
        model_to_use = self.model
        if sub_intent in UI_SUB_INTENTS:
            model_to_use = UI_MODEL
        elif sub_intent in LOGIC_CODING_INTENTS:
            model_to_use = CODING_MODEL
        if personality and personality.get("id") == "coding_buddy":
            model_to_use = UI_MODEL if sub_intent in UI_SUB_INTENTS else CODING_MODEL
            print(f"[LLM] Switching to {model_to_use} for Coding Buddy")

        try:
            reason = "ui_override" if sub_intent in UI_SUB_INTENTS else "coding_override" if (sub_intent in LOGIC_CODING_INTENTS or (personality and personality.get("id") == "coding_buddy")) else "default_persona_model"
            log_routing_decision(user_id, {
                "intent": "INTERNAL",
                "sub_intent": sub_intent,
                "mode": mode,
                "forced_model": model_to_use,
                "reason": reason
            })
        except Exception as e:
            print(f"[RoutingLog] Failed: {e}")

        client = get_openrouter_client() if model_to_use == LLM_MODEL else get_openrouter_client()
        create_kwargs = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        }
        max_tokens = _get_max_tokens_for_sub_intent(sub_intent)
        if max_tokens:
            create_kwargs["max_tokens"] = max_tokens
        temperature = self._get_temperature(personality)
        if sub_intent in UI_SUB_INTENTS and temperature is not None:
            temperature = min(temperature, 0.2)
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        _apply_reasoning(create_kwargs, model_to_use, sub_intent, user_settings)
        create_kwargs = self._guard_kwargs(create_kwargs, user_query)
        response = await client.chat.completions.create(**create_kwargs)
        return self._sanitize_output_text(response.choices[0].message.content)
    
    async def process_deep_search_query(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        sub_intent: str = "general",
        user_id: Optional[str] = None,
        emotions: List[str] = [],
        emotional_instruction: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> tuple[str, List[str]]:
        """
        Handle external queries that need web search.
        
        Returns: (response, tools_activated)
        """
        user_query = normalize_user_query(user_query)
        # Select tools based on mode
        selected_tools = await select_tools_for_mode(user_query, mode)
        tools_dict = get_tools_for_mode(mode)
        
        # Aggregate data from selected tools
        aggregated_context = {}
        for tool in selected_tools:
            if tool in tools_dict:
                endpoint = tools_dict[tool]
                param_key = "url" if tool == "Webpage" else "q"
                result = await execute_serper_batch(endpoint, [user_query], param_key=param_key)
                aggregated_context[tool] = result
        
        # Synthesize response
        context_str = json.dumps(aggregated_context, indent=2)
        
        # Determine system prompt
        if personality and mode == "normal":
            # ?? Personalities are ONLY honored in Normal Mode
            from app.llm.router import get_system_prompt_for_personality
            system_prompt = get_system_prompt_for_personality(personality, user_settings, user_id, user_query, session_id=session_id)
        else:
            # Business / Deep Search / Normal (without persona) -> Use default System Prompt defined in router.py
            # This ensures Business Mode uses the EXACT "Elite Strategic Advisor" prompt from Business.py
            system_prompt = get_system_prompt_for_mode(mode, user_settings, user_id, user_query, session_id=session_id)

        # Apply specialized internal prompt overlays for coding/technical intents
        from app.llm.router import INTERNAL_MODE_PROMPTS
        if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
            system_prompt = f"{system_prompt}\n\n**MODE SWITCH: {sub_intent.upper()}**\n{INTERNAL_MODE_PROMPTS[sub_intent]}"
        elif mode == "normal" and sub_intent == "general":
            system_prompt = f"{system_prompt}\n\n{NORMAL_GENERIC_FORMAT_RULES}"
        # Apply Emotion/Tone Instruction
        if emotional_instruction:
            system_prompt = f"{system_prompt}\n\n{emotional_instruction}"
        else:
            for emotion in emotions:
                if emotion in TONE_MAP:
                    system_prompt = f"{system_prompt}\n\n{TONE_MAP[emotion]}"

        
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_messages:
            messages.extend(context_messages[-6:])
        
        messages.append({
            "role": "user", 
            "content": f"Search Data:\n{context_str}\n\nUser Query: {user_query}"
        })
        
        # Resolve final model using unified routing (shared with streaming)
        client_type, model_to_use, temperature, _, route_reason = get_model_for_intent(
            mode, sub_intent, personality, user_query
        )

        try:
            log_routing_decision(user_id, {
                "intent": "EXTERNAL",
                "sub_intent": sub_intent,
                "mode": mode,
                "forced_model": model_to_use,
                "reason": route_reason
            })
        except Exception as e:
            print(f"[RoutingLog] Failed: {e}")

        client = get_openrouter_client() if client_type == "openai" else get_openrouter_client()
        create_kwargs = {
            "model": model_to_use,
            "messages": messages
        }
        create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        _apply_reasoning(create_kwargs, model_to_use, sub_intent, user_settings)
        create_kwargs = self._guard_kwargs(create_kwargs, user_query)
        response = await client.chat.completions.create(**create_kwargs)
        
        return self._sanitize_output_text(response.choices[0].message.content), selected_tools
    
    async def process_message(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - routes to appropriate handler.
        Returns complete response dict.
        """
        user_query = normalize_user_query(user_query)
        # Analyze intent using the consolidated router
        analysis = await analyze_and_route_query(user_query, mode, personality=personality)
        routed_intent = analysis.get("intent", "DEEP_SEARCH")
        sub_intent = analysis.get("sub_intent", "general")
        emotions = analysis.get("emotions", [])
        intent = _resolve_runtime_intent(mode, routed_intent, user_query)
        
        import time as _time
        _start = _time.time()

        # === Intelligence Layer: Load state ===
        emotional_instruction = None
        strategy_instruction = None
        prompt_variant_name = "default"
        prompt_variant_instruction = ""
        skill_level = 0.5

        if session_id:
            state = await emotion_engine.load_state(session_id, user_id=user_id)
            state = emotion_engine.update_state(state, emotions, sub_intent)
            await emotion_engine.save_state(session_id, state, user_id=user_id)
            emotional_instruction = emotion_engine.get_instruction(state)
            skill_level = state.skill_level

        # Strategy Memory: Learn from query + get instruction
        if user_id:
            user_strategy = await strategy_memory.load_strategy(user_id)
            user_strategy = strategy_memory.update_from_query(user_strategy, user_query)
            await strategy_memory.save_strategy(user_id, user_strategy)
            strategy_instruction = strategy_memory.get_instruction(user_strategy)

        # Prompt Optimizer: Select variant
        prompt_variant_name, prompt_variant_instruction = prompt_optimizer.select_variant(sub_intent, session_id or "")

        tools = []
        if intent == "INTERNAL":
            response_text = await self.process_agent_query(
                user_query,
                personality,
                user_settings,
                mode=mode,
                sub_intent=sub_intent,
                user_id=user_id,
                emotions=emotions,
                emotional_instruction=emotional_instruction,
                session_id=session_id
            )
            result_response = self._format_user_answer(response_text, mode)
        elif intent == "AGENT":
            # Run structured agent pipeline even for non-stream requests
            import time as _time
            _agent_tokens: List[str] = []
            async for token in self._process_agent_mode(
                user_query,
                context_messages,
                personality,
                user_settings,
                user_id,
                session_id,
                start_time=_time.time(),
                mode=mode,
                intent=routed_intent,
                sub_intent=sub_intent,
            ):
                if token.startswith("[INFO]"):
                    continue
                _agent_tokens.append(token)
            response_text = "".join(_agent_tokens)
            result_response = self._format_user_answer(response_text, mode)
        else:
            response_text, tools = await self.process_deep_search_query(
                user_query, mode, context_messages, personality, user_settings, 
                sub_intent, user_id, 
                emotions=emotions,
                emotional_instruction=emotional_instruction,
                session_id=session_id
            )
            result_response = self._format_user_answer(response_text, mode)

        # Optional verification loop for complex business/agent outputs.
        if self._should_run_verifier(mode, user_query, result_response):
            try:
                verified = await asyncio.wait_for(
                    self._verify_response_quality(user_query, result_response, mode),
                    timeout=MAX_VERIFY_TIME,
                )
                if verified:
                    result_response = verified
            except asyncio.TimeoutError:
                print(f"[Verifier] Timeout after {MAX_VERIFY_TIME:.1f}s, using draft response")
            except Exception as e:
                print(f"[Verifier] Non-blocking failure: {e}")

        blocked_out, _block_reason = classify_nsfw(result_response)
        if blocked_out:
            result_response = "I can help with educational or safety-oriented discussion, but I can't provide explicit sexual content."

        # === Feedback Engine: Log interaction ===
        _latency = (_time.time() - _start) * 1000
        try:
            skill_trace = get_last_skill_trace(session_id)
            record_skill_outcome(
                session_id=session_id,
                success=True,
                latency_ms=_latency,
                fallback_level=int(skill_trace.get("fallback_level", 0)),
            )
        except Exception as e:
            print(f"[SkillRouter] Outcome logging failed: {e}")

        feedback_engine.log_interaction(
            session_id=session_id or "",
            user_id=user_id or "",
            query=user_query,
            intent=intent,
            sub_intent=sub_intent,
            model_used=mode,
            emotions=emotions,
            skill_level=skill_level,
            response_length=len(result_response),
            latency_ms=_latency,
            prompt_variant=prompt_variant_name
        )

        # === User Profiler: Update from interaction ===
        if user_id:
            try:
                _profile = await user_profiler.load_profile(user_id)
                user_profiler.update_from_interaction(
                    _profile, sub_intent, user_query,
                    len(result_response), emotions,
                    model_used=mode, prompt_variant=prompt_variant_name
                )
            except Exception as e:
                print(f"[UserProfiler] Update failed: {e}")

        return {
            "success": True,
            "response": result_response,
            "mode_used": "agent" if intent == "AGENT" else "deep_search" if intent in ("DEEP_SEARCH", "EXTERNAL") else "internal",
            "tools_activated": tools if intent in ("DEEP_SEARCH", "EXTERNAL") else []
        }
    

    async def process_message_stream(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        resume_graph: Optional[Any] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version of process_message.
        Yields tokens as they're generated.
        """
        user_query = normalize_user_query(user_query)
        import time
        start_time = time.time()
        print(f"[LATENCY] Processing Stream Request... (Time: 0.0000s)")
        streamed_output = ""

        # Phase 1: Planning
        yield f'[INFO]{{"agent_state": "planning", "topic": "Analyzing query & intent"}}'
        
        # Combined Analysis (Intent + Tools)
        t_analysis_start = time.time()
        
        # ?? PARALLEL DATA LOADING
        # 1. Start Analysis Task
        analysis_task = asyncio.create_task(
            analyze_and_route_query(user_query, mode, context_messages, personality=personality)
        )
        
        # 2. Start Intelligence Loading Tasks (Not awaiting yet)
        intel_tasks = {}
        if user_id:
            intel_tasks["strategy"] = asyncio.create_task(strategy_memory.load_strategy(user_id))
            intel_tasks["profile"] = asyncio.create_task(user_profiler.load_profile(user_id))
        if session_id:
            intel_tasks["emotion"] = asyncio.create_task(emotion_engine.load_state(session_id, user_id=user_id))
            if user_id:
                from app.llm.safe_agent import analyze_constraints, CONSTRAINT_TIMEOUT_S
                # Wrap constraint analysis in wait_for internally to handle timeout gracefully in gather
                async def safe_constraint_task():
                    try:
                        return await asyncio.wait_for(
                            analyze_constraints(user_query, user_id, session_id, context_messages),
                            timeout=CONSTRAINT_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        print(f"[SafeAgent] Constraint analysis timed out ({CONSTRAINT_TIMEOUT_S*1000:.0f}ms)")
                        return None
                    except Exception as e:
                        print(f"[SafeAgent] Constraint analysis failed: {e}")
                        return None
                intel_tasks["constraint"] = asyncio.create_task(safe_constraint_task())

        # Wait for Analysis (Needed for intent routing)
        analysis = await analysis_task
        routed_intent = analysis.get("intent", "DEEP_SEARCH")
        selected_tools = analysis.get("tools", [])
        sub_intent = analysis.get("sub_intent", "general")
        intent = _resolve_runtime_intent(mode, routed_intent, user_query)
        show_reasoning_panel = _should_show_reasoning_panel(mode, sub_intent, user_settings)
        
        print(f"[LATENCY] Analysis/Router Complete ({intent}): {time.time() - start_time:.4f}s (Analysis took: {time.time() - t_analysis_start:.4f}s)")
        
        if intent == "AGENT":
            for _task in intel_tasks.values():
                _task.cancel()
            await asyncio.gather(*intel_tasks.values(), return_exceptions=True)
            async for token in self._process_agent_mode(
                user_query, context_messages, personality, user_settings,
                user_id, session_id, start_time,
                task_id=task_id, resume_graph=resume_graph,
                mode=mode,
                intent=routed_intent,
                sub_intent=sub_intent
            ):
                yield token
            return

        # System prompt selection
        if intent == "INTERNAL":
            if personality:
                system_prompt = get_internal_system_prompt_for_personality(personality, user_settings, user_id, mode=mode)
            else:
                from app.llm.router import _build_user_context_string
                system_prompt = INTERNAL_SYSTEM_PROMPT + _build_user_context_string(user_settings)
                if mode == "normal":
                    from app.llm.router import NORMAL_MARKDOWN_POLISH
                    system_prompt = f"{system_prompt}\n{NORMAL_MARKDOWN_POLISH}"
        else:
            if personality and mode == "normal":
                from app.llm.router import get_system_prompt_for_personality
                system_prompt = get_system_prompt_for_personality(personality, user_settings, user_id, user_query, session_id=session_id)
            else:
                system_prompt = get_system_prompt_for_mode(mode, user_settings, user_id, user_query, session_id=session_id)

        from app.llm.router import INTERNAL_MODE_PROMPTS
        if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
            specialized_prompt = INTERNAL_MODE_PROMPTS[sub_intent]
            system_prompt = f"{system_prompt}\n\n**MODE SWITCH: {sub_intent.upper()}**\n{specialized_prompt}"
        elif mode == "normal" and sub_intent == "general":
            system_prompt = f"{system_prompt}\n\n{NORMAL_GENERIC_FORMAT_RULES}"
        # Wait for all memory/profile tasks securely
        if sub_intent == "ui_demo_html" and _wants_single_file_html(user_query):
            system_prompt = f"{system_prompt}{_single_file_html_instruction()}"
        await asyncio.gather(*intel_tasks.values(), return_exceptions=True)

        # Extract intelligence results
        def get_task_res(key):
            if key in intel_tasks:
                exc = intel_tasks[key].exception()
                if not exc: return intel_tasks[key].result()
            return None

        user_strategy = get_task_res("strategy")
        user_profile = get_task_res("profile")
        state = get_task_res("emotion")
        constraint_result = get_task_res("constraint")

        # Apply Emotion/Tone Instruction (Streaming Path - Multi-Label)
        emotions = analysis.get("emotions", [])
        for emotion in emotions:
            if emotion in TONE_MAP:
                system_prompt = f"{system_prompt}\n\n{TONE_MAP[emotion]}"
        
        # === Intelligence Layer (Streaming) ===
        # Strategy Memory
        strategy_instruction = None
        if user_id and user_strategy:
            user_strategy = strategy_memory.update_from_query(user_strategy, user_query)
            # Fire and forget strategy save
            asyncio.create_task(strategy_memory.save_strategy(user_id, user_strategy))
            strategy_instruction = strategy_memory.get_instruction(user_strategy)

        # Prompt Optimizer
        prompt_variant_name, prompt_variant_instruction = prompt_optimizer.select_variant(sub_intent, session_id or "")

        # Persist Emotional State (with user_id for skill persistence)
        skill_level = 0.5
        is_debug_mode = False
        frustration_prob = 0.0
        proactive_instruction = None
        if session_id and state:
            state = emotion_engine.update_state(state, emotions, analysis.get("sub_intent", "general"))
            # Async state save
            asyncio.create_task(emotion_engine.save_state(session_id, state, user_id=user_id))
            skill_level = state.skill_level
            
            # Get dynamic instruction
            emotional_instruction = emotion_engine.get_instruction(state)
            if emotional_instruction:
                print(f"[EmotionEngine] Injecting instruction")
                system_prompt = f"{system_prompt}\n\n{emotional_instruction}"

            # === Frustration Prediction (Priority #6) ===
            frustration_prob = emotion_engine.predict_frustration(state, user_query, sub_intent)
            proactive_instruction = emotion_engine.get_proactive_instruction(frustration_prob)
            if proactive_instruction:
                system_prompt = f"{system_prompt}\n\n{proactive_instruction}"

            # === Autonomous Debug Mode (Priority #5) ===
            if debug_agent.should_activate(emotions, state, sub_intent):
                is_debug_mode = True
                debug_prompt = debug_agent.get_debug_system_prompt(sub_intent)
                system_prompt = f"{system_prompt}\n\n{debug_prompt}"
                yield debug_agent.get_info_message()

        # === User Profiler (Priority #2) ===
        if user_id and user_profile:
            personalization = user_profiler.get_personalization_instruction(user_profile)
            if personalization:
                system_prompt = f"{system_prompt}\n\n{personalization}"

        # Inject strategy instruction
        if strategy_instruction:
            system_prompt = f"{system_prompt}\n\n{strategy_instruction}"

        # Inject prompt variant
        if prompt_variant_instruction:
            system_prompt = f"{system_prompt}\n\n{prompt_variant_instruction}"

        # === SAFE CHAT AGENT: Constraint Analysis (Steps 1-11) ===
        if constraint_result and constraint_result.constraint_prompt:
            system_prompt = f"{system_prompt}\n\n{constraint_result.constraint_prompt}"
            if DEBUG_SAFE_AGENT:
                print(f"[SafeAgent] Pipeline complete: {constraint_result.log_summary()}")

        print(f"[LATENCY] Starting Internal Stream ({analysis.get('sub_intent', 'general')}): {time.time() - start_time:.4f}s")

        # Setup citation tracker
        from app.context.citation_engine import CitationTracker, CITATION_INSTRUCTION
        citation_tracker = CitationTracker()
        include_sources = _user_wants_sources(user_query)

        # === Semantic Query Planner (zero-cost) ===
        _query_plan = {}
        try:
            from app.context.query_planner import plan_query
            _query_plan = plan_query(
                user_query,
                sub_intent=sub_intent,
                has_urls=bool(re.search(r'https?://[^\s]+', user_query)),
            )
        except Exception as e:
            print(f"[QueryPlanner] Failed (non-blocking): {e}")
            _query_plan = {"needs_web": True, "needs_memory": True, "needs_chunking": True}
        if intent == "INTERNAL":
            _query_plan = {"needs_web": False, "needs_memory": False, "needs_chunking": False, "task_type": "general_chat"}

        factual_lookup = _is_factual_lookup_query(user_query)
        needs_retrieval = intent in ("DEEP_SEARCH", "EXTERNAL") or factual_lookup
        if needs_retrieval:
            _query_plan["needs_web"] = True
        # Phase 2: Researching
        if needs_retrieval:
            yield f'[INFO]{{"agent_state": "researching", "topic": "Fetching relevant context"}}'

        # === Dynamic Context Router + PARALLEL Retrieval + Packer ===
        _has_web_content = False
        try:
            from app.context.context_router import select_context_layers
            from app.context.context_packer import pack_context
            from app.context.retrieval_intelligence import process_retrieval

            active_layers = select_context_layers(
                user_query,
                sub_intent=sub_intent,
                has_profile=bool(user_profile),
                has_web_content=False,  # Not fetched yet
            )

            _packed_graph = None
            _packed_memories = None
            
            _compacted_graph = None
            _compacted_memory = None

            # === PARALLEL retrieval with safe timeouts ===
            _retrieval_tasks = []
            _retrieval_keys = []

            if "knowledge_graph" in active_layers and _query_plan.get("needs_memory", True):
                async def _fetch_graph():
                    from app.memory.knowledge_graph import retrieve_graph
                    return await retrieve_graph(user_id, user_query, limit=5)
                _retrieval_tasks.append(asyncio.create_task(_fetch_graph()))
                _retrieval_keys.append("graph")

            if "vector_memory" in active_layers and _query_plan.get("needs_memory", True):
                async def _fetch_vector():
                    from app.memory.vector_memory import retrieve_memories
                    return await retrieve_memories(user_id, user_query, top_k=10, final_limit=5)
                _retrieval_tasks.append(asyncio.create_task(_fetch_vector()))
                _retrieval_keys.append("vector")

            # Unified Web Search (Serper)
            if needs_retrieval and (intent == "DEEP_SEARCH" or _query_plan.get("needs_web", False)):
                async def _fetch_web_search():
                    from app.llm.router import execute_serper_batch
                    # Use the correct Search endpoint from SERPER_TOOLS
                    endpoint = SERPER_TOOLS.get("Search", "https://google.serper.dev/search")
                    return await execute_serper_batch(endpoint, [user_query])
                _retrieval_tasks.append(asyncio.create_task(_fetch_web_search()))
                _retrieval_keys.append("web_search")

            # RAG Document Retrieval (uploaded files for this session)
            if needs_retrieval and session_id and user_id:
                async def _fetch_rag():
                    from app.rag.retrieval import retrieve_rag_context
                    return await retrieve_rag_context(user_id=user_id, query=user_query, session_id=session_id, top_k=5)
                _retrieval_tasks.append(asyncio.create_task(_fetch_rag()))
                _retrieval_keys.append("rag")

            if _retrieval_tasks:
                _parallel_start = time.time()
                
                # Extended timeout for web/graph retrieval (5 seconds)
                done, pending = await asyncio.wait(
                    _retrieval_tasks,
                    timeout=5.0
                )
                
                _results = [None] * len(_retrieval_tasks)
                
                for i, task in enumerate(_retrieval_tasks):
                    key = _retrieval_keys[i]
                    if task in done:
                        if not task.cancelled() and task.exception() is None:
                            _results[i] = task.result()
                        elif task.exception():
                            print(f"[{key.capitalize()}Retrieval] failed: {task.exception()}")
                    else:
                        # Pending task timed out
                        task.cancel()
                        _parallel_ms = (time.time() - _parallel_start) * 1000
                        print(f"[RetrievalTimeout] task={key}_search duration={_parallel_ms:.0f}ms trace=sys")
                        
                _parallel_ms = (time.time() - _parallel_start) * 1000
                print(f"[ParallelRetrieval] completed in {_parallel_ms:.0f}ms")

                for key, result in zip(_retrieval_keys, _results):
                    if result is None:
                        continue
                    if key == "graph":
                        _packed_graph = result
                    elif key == "vector":
                        _packed_memories = result
                    elif key == "rag":
                        # result is context string from retrieve_rag_context
                        if result and str(result).strip():
                            citation_tracker.add_rag_results(result)
                            system_prompt += f"\n\n**LOCALLY UPLOADED DOCUMENTS:**\n{result}"
                            system_prompt += f"\n\n{RAG_INSTRUCTION}"
                    elif key == "web_search":
                        # Process Serper results via CitationTracker
                        citation_tracker.add_serper_results(result)
                        
                        # Correctly extract organic results from batch response
                        organic_results = []
                        if isinstance(result, list) and result:
                            # Batch response: result is a list of results for each query
                            first_q = result[0]
                            if isinstance(first_q, dict):
                                organic_results = first_q.get("organic", [])
                        elif isinstance(result, dict):
                            # Non-batch response
                            organic_results = result.get("organic", [])
                            
                        # Inject top results into system prompt
                        if organic_results:
                            search_context = "\n".join([
                                f"- {r.get('title')}: {r.get('snippet')} ({r.get('link')})" 
                                for r in organic_results[:4] if isinstance(r, dict)
                            ])
                            system_prompt += f"\n\n**FRESH WEB CONTEXT (March 2026):**\n{search_context}"

            # === Retrieval Intelligence: deduplicate + filter ===
            if _packed_graph or _packed_memories:
                ril_result = process_retrieval(
                    vector_memories=_packed_memories,
                    graph_triples=_packed_graph,
                    query=user_query,
                )
                _packed_memories = ril_result["vector_memories"]
                _packed_graph = ril_result["graph_triples"]
                
                # Track citations for the filtered context
                citation_tracker.add_vector_memories(_packed_memories)
                citation_tracker.add_graph_triples(_packed_graph)

            # === Context Compactor: dedup + merge + compress ===
            if _packed_graph or _packed_memories:
                try:
                    from app.context.context_compactor import compact_context
                    graph_texts = []
                    if _packed_graph:
                        for t in _packed_graph:
                            subj = getattr(t, 'subject', '') if hasattr(t, 'subject') else ''
                            rel = getattr(t, 'relation', '') if hasattr(t, 'relation') else ''
                            obj = getattr(t, 'object', '') if hasattr(t, 'object') else ''
                            graph_texts.append(f"{subj} {rel} {obj}")
                    mem_texts = []
                    if _packed_memories:
                        for m in _packed_memories:
                            mem_texts.append(getattr(m, 'text', '') if hasattr(m, 'text') else str(m))
                    compacted = compact_context(graph_items=graph_texts, memory_items=mem_texts)
                    # Use compacted text lists for packing
                    _compacted_graph = compacted["graph"]
                    _compacted_memory = compacted["memory"]
                except Exception as e:
                    print(f"[ContextCompactor] Failed (non-blocking): {e}")
                    _compacted_graph = None
                    _compacted_memory = None

            # Pack all context into compact block with semantic IDs
            if _packed_graph or _packed_memories:
                packed_block = pack_context(
                    graph_triples=_compacted_graph if _compacted_graph is not None else _packed_graph,
                    vector_memories=_compacted_memory if _compacted_memory is not None else _packed_memories,
                )
                if packed_block:
                    system_prompt = f"{system_prompt}\n{packed_block}"
                    g_count = len(_packed_graph) if _packed_graph else 0
                    m_count = len(_packed_memories) if _packed_memories else 0
                    print(f"[ContextPacker] Injected: {g_count} triples + {m_count} memories (layers: {active_layers})")

        except Exception as e:
            print(f"[ContextRouter] Failed (non-blocking): {e}")

        # === URL Auto-Fetch (AFTER Router - conditional on planner) ===
        if needs_retrieval and _query_plan.get("needs_web", True):
            try:
                from app.input_processing.web_fetch import detect_urls, fetch_and_extract
                from app.safety.safety_filter import wrap_web_content
                detected_urls = detect_urls(user_query)
                if detected_urls:
                    for url in detected_urls[:2]:
                        fetch_result = await asyncio.wait_for(
                            fetch_and_extract(url), timeout=5.0
                        )
                        if fetch_result.get("status") == "success":
                            page_data = fetch_result["data"]
                            page_content = page_data.get("content", "")
                            page_title = page_data.get("title", "")
                            
                            # Junk page filter
                            if len(page_content) < 200:
                                print(f"[WebFetch] Skipped junk page (length < 200 chars): {url}")
                                continue
                            
                            # Safety Limit: Cap long pages
                            if len(page_content) > 12000:
                                print(f"[WebFetch] Truncating page from {len(page_content)} to 12000 chars: {url}")
                                page_content = page_content[:12000]

                            from app.input_processing.doc_chunker import needs_chunking, analyze_document
                            if needs_chunking(page_content):
                                insights = await asyncio.wait_for(
                                    analyze_document(page_content), timeout=15.0
                                )
                                if insights:
                                    system_prompt += wrap_web_content(insights, url, page_title)
                                    citation_tracker.add_web_content(url, page_title)
                                    _has_web_content = True
                                    print(f"[WebFetch] Chunked analysis for: {url} ({len(page_content)} chars)")
                            else:
                                system_prompt += wrap_web_content(page_content, url, page_title)
                                citation_tracker.add_web_content(url, page_title)
                                _has_web_content = True
                                print(f"[WebFetch] Injected content for: {url} ({len(page_content)} chars)")
                        else:
                            print(f"[WebFetch] Failed for {url}: {fetch_result.get('data', 'unknown')}")
            except asyncio.TimeoutError:
                print("[WebFetch] URL fetch timed out (5s)")
            except Exception as e:
                print(f"[WebFetch] URL auto-fetch failed (non-blocking): {e}")
        else:
            print("[QueryPlanner] Web fetch skipped (not needed)")

        # Append Citation Instructions & Context Footnotes to system prompt
        if include_sources and citation_tracker.citations:
            system_prompt += CITATION_INSTRUCTION
            system_prompt += citation_tracker.get_footnotes()
            print(f"[CitationEngine] Added {len(citation_tracker.citations)} source citations")
            
            # Emit retrieved context IDs precisely for quality scoring
            cited_vectors = [c.reference for c in citation_tracker.citations if c.source_type == "vector"]
            cited_graphs = [c.reference for c in citation_tracker.citations if c.source_type == "graph"]
            if cited_vectors or cited_graphs:
                yield f"[INFO]RETRIEVED_CONTEXT:{json.dumps({'vector': cited_vectors, 'graph': cited_graphs})}"

        # === Structured Reasoning Scaffold (Item #10: polished) ===
        _scaffold = (
            "\n\nBefore answering, think through these steps internally:\n"
            "1. Identify the task type.\n"
            "2. Determine which context sources are relevant.\n"
            "3. Extract key facts from the context.\n"
            "4. Reason carefully before producing the final answer.\n\n"
            "Only output the final answer to the user. "
            "Do not expose your reasoning steps."
        )
        # Phase 3: Synthesizing
        yield f'[INFO]{{"agent_state": "synthesizing", "topic": "Merging & refining knowledge"}}'

        system_prompt = f"{system_prompt}{_scaffold}"

        # === Dynamic Response Formatter (Item #8) ===
        try:
            from app.context.response_formatter import format_instruction
            _task_type = _query_plan.get("task_type", "general_chat")
            _fmt_instruction = format_instruction(_task_type)
            if _fmt_instruction:
                system_prompt = f"{system_prompt}{_fmt_instruction}"
        except Exception:
            pass  # Non-blocking

        # === Context Intelligence (Priority #4) ===
        # Apply smart context compression before building messages
        optimized_context = context_messages
        if context_messages and len(context_messages) > 4:
            pre_count = len(context_messages)
            pre_chars = sum(len(m.get('content', '')) for m in context_messages)
            optimized_context = context_optimizer.optimize(context_messages, keep_last_n=4)
            post_chars = sum(len(m.get('content', '')) for m in optimized_context)
            if pre_chars != post_chars:
                print(f"[ContextOptimizer] Compressed: {pre_count} msgs, {pre_chars} ? {post_chars} chars ({100 - (post_chars/max(pre_chars,1))*100:.0f}% reduction)")
        if _is_casual_query(user_query):
            optimized_context = []

        # Construct messages with Context
        messages = [{"role": "system", "content": system_prompt}]

        if optimized_context:
            messages.extend(optimized_context)

        messages.append({"role": "user", "content": user_query})

        # Cost guard: preserve system + primary skill capsule, trim context if over budget.
        est_tokens = rough_estimate_tokens(messages)
        if est_tokens > 4500 and optimized_context:
            trimmed_context = context_optimizer.optimize(optimized_context, keep_last_n=2)
            messages = [{"role": "system", "content": system_prompt}] + trimmed_context + [{"role": "user", "content": user_query}]

        # === Emit Intelligence Metadata for Frontend ===
        import json as _json
        intel_tags = []
        if user_profile and user_profile.total_interactions >= 5:
            if user_profile.prefers_code_first > 0.7:
                intel_tags.append("code-first")
            if user_profile.prefers_concise > 0.7:
                intel_tags.append("concise")
            if user_profile.prefers_step_by_step > 0.7:
                intel_tags.append("step-by-step")
            if user_profile.prefers_examples > 0.7:
                intel_tags.append("examples")
        
        intel_payload = {
            "mode": sub_intent,
            "debug_active": is_debug_mode,
            "frustration": round(frustration_prob, 2) if session_id else 0,
            "confidence": round(max(0, 1.0 - (frustration_prob if session_id else 0)), 2),
            "skill_level": round(skill_level, 2),
            "personalization": intel_tags,
            "proactive_help": bool(proactive_instruction) if session_id else False
        }
        yield f"[INFO]INTEL:{_json.dumps(intel_payload)}"

        print(f"[LATENCY] Internal: Preparing stream... (Time: {time.time() - start_time:.4f}s)")
        
        # Phase 4: Finalizing
        yield f'[INFO]{{"agent_state": "finalizing", "topic": "Generating response"}}'
        
        t_stream_start = time.time()
        
        # ?? Multi-Model Routing with Trace
        # Initialize trace for this request
        trace = RequestTrace()
        trace.intent = intent
        trace.sub_intent = sub_intent
        trace.log("ROUTE_START", f"mode={mode}, sub_intent={sub_intent}")
        
        # Get routing decision with confidence gating
        client_type, model_to_use, temperature, is_thinking_pass, route_reason = get_model_for_intent(
            mode, sub_intent, personality, user_query
        )

        # Suppress thinking panel when no reasoning is requested for Gemini
        if model_to_use == FAST_MODEL and not _get_reasoning_config(sub_intent, user_settings):
            show_reasoning_panel = False
        
        # Track if gating skipped thinking
        if sub_intent in ["debugging", "system_design", "analysis", "research", "reasoning"] and not is_thinking_pass:
            trace.confidence_gating_skipped = True
            trace.log("GATING", "Skipped thinking pass (query too simple)")
        
        trace.model_chain.append(model_to_use)
        trace.is_thinking_pass = is_thinking_pass
        
        trace.log("ROUTE_DECISION", f"client={client_type}, model={model_to_use}, temp={temperature}, thinking={is_thinking_pass}, reason={route_reason}")
        try:
            log_routing_decision(user_id, {
                "intent": intent,
                "sub_intent": sub_intent,
                "mode": mode,
                "forced_model": model_to_use,
                "reason": route_reason
            })
        except Exception as e:
            print(f"[RoutingLog] Failed: {e}")
        
        # Select the appropriate client
        if client_type == "openai":
            client = get_openrouter_client()
        else:
            client = get_openrouter_client()
        
        # === SAFE CHAT AGENT: Streaming decision ===
        use_streaming = True
        if constraint_result and constraint_result.has_strict():
            use_streaming = True
        create_kwargs = {
            "model": model_to_use,
            "messages": messages,
            "stream": use_streaming
        }
        max_tokens = _get_max_tokens_for_sub_intent(sub_intent)
        if max_tokens:
            create_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        
        _apply_reasoning(create_kwargs, model_to_use, sub_intent, user_settings)
        if is_thinking_pass:
            # Pass 1: Thinking model plans and analyzes (NO code generation)
            trace.log("THINKING_START", f"model={model_to_use}")
            thinking_response = ""
            
            # For coding intents, use a planning-focused prompt
            if sub_intent in LOGIC_CODING_INTENTS:
                planning_messages = [
                    {"role": "system", "content": (
                        "You are an expert software architect, debugger, and technical analyst. "
                        "Your job is to THINK, PLAN, and ANALYZE - NOT to write code.\n\n"
                        "Depending on the request:\n"
                        "- If BUILDING something: Break down requirements, identify tech stack, "
                        "structure, layout, design decisions, colors, fonts, interactions, and best practices.\n"
                        "- If DEBUGGING: Analyze the error/issue, identify root causes, "
                        "explain what's going wrong, and outline the exact fix strategy step by step.\n"
                        "- If EXPLAINING code: Break down the code's purpose, logic flow, "
                        "key functions, data flow, and how each part connects.\n"
                        "- If DESIGNING a system: Identify components, architecture patterns, "
                        "data models, APIs, and trade-offs.\n\n"
                        "DO NOT write any code. Only provide a detailed analysis and plan "
                        "that a developer can use to implement or fix the solution perfectly."
                    )},
                ]
                if context_messages:
                    planning_messages.extend(context_messages)
                planning_messages.append({"role": "user", "content": user_query})
                
                planning_kwargs = {
                    "model": model_to_use,
                    "messages": planning_messages,
                    "stream": True,
                    "temperature": 0.7
                }
                planning_kwargs = self._guard_kwargs(planning_kwargs, user_query)
                stream = await client.chat.completions.create(**planning_kwargs)
            else:
                create_kwargs = self._guard_kwargs(create_kwargs, user_query)
                stream = await client.chat.completions.create(**create_kwargs)
            
            # Signal start of thinking to UI
            yield "[THINKING]"

            async for chunk in stream:
                if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                    r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                    if isinstance(chunk.usage, dict):
                        r_tokens = chunk.usage.get("reasoning_tokens", 0)
                    if r_tokens:
                        yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, "content", None):
                    content_chunk = chunk.choices[0].delta.content
                    thinking_response += content_chunk
                    # Stream the thinking content to the UI
                    yield self._sanitize_output_text(content_chunk, trim_outer=False)
            
            trace.log("THINKING_COMPLETE", f"chars={len(thinking_response)}")
            
            # Signal end of thinking
            yield "[/THINKING]"

            # Pass 2: Generate final output with appropriate model
            if sub_intent in UI_SUB_INTENTS:
                output_model = UI_MODEL
            elif sub_intent in LOGIC_CODING_INTENTS:
                output_model = CODING_MODEL
            else:
                output_model = FAST_MODEL

            trace.model_chain.append(output_model)
            trace.log("OUTPUT_START", f"model={output_model}")
            
            # Construct pass 2 messages with thinking context
            if sub_intent in LOGIC_CODING_INTENTS:
                # For coding: GLM gets the plan and writes production code
                pass2_messages = [
                    {"role": "system", "content": (
                        f"{system_prompt}\n\n"
                        "You are now implementing code based on a detailed technical plan. "
                        "Write clean, complete, production-ready code following the plan exactly. "
                        "Include all styling, interactivity, and details mentioned in the plan. "
                        "Make the output visually stunning and modern."
                    )},
                    {"role": "user", "content": f"TECHNICAL PLAN:\n\n{thinking_response}\n\nORIGINAL REQUEST: {user_query}\n\nNow write the complete, production-ready code based on this plan."}
                ]
            else:
                pass2_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on this analysis:\n\n{thinking_response}\n\nProvide a clear, well-formatted response to: {user_query}"}
                ]
            
            if context_messages:
                pass2_messages = [pass2_messages[0]] + context_messages + [pass2_messages[-1]]
            
            pass2_output = ""
            pass2_followups_preview_sent = False
            pass2_kwargs = {
                "model": output_model,
                "messages": pass2_messages,
                "stream": True,
                "temperature": 0.3  # Lower temp for synthesis
            }
            if max_tokens:
                pass2_kwargs["max_tokens"] = max_tokens
            if sub_intent in UI_SUB_INTENTS:
                pass2_kwargs["stream"] = False
                pass2_kwargs = self._guard_kwargs(pass2_kwargs, user_query)
                response = await get_openrouter_client().chat.completions.create(**pass2_kwargs)
                response_text = (response.choices[0].message.content or "")
                fixed = self._fix_html_css_output(response_text)
                sanitized = self._sanitize_output_text(fixed, allow_double_hyphen=True)
                # === Response Critic (buffered path) ===
                try:
                    from app.safety.response_critic import needs_critic, run_critic
                    if needs_critic(sub_intent, sanitized):
                        correction = await run_critic(user_query, sanitized)
                        if correction:
                            sanitized = self._sanitize_output_text(correction, allow_double_hyphen=True)
                except Exception as crit_e:
                    print(f"[Critic] Error (non-blocking): {crit_e}")
                for i in range(0, len(sanitized), 800):
                    chunk_text = sanitized[i:i+800]
                    pass2_output += chunk_text
                    if not pass2_followups_preview_sent:
                        preview_payload = self._build_followup_payload(pass2_output, mode, session_id, persist=False)
                        if preview_payload.get("followups") or preview_payload.get("action_chips"):
                            yield f"[INFO]{json.dumps(preview_payload)}"
                            pass2_followups_preview_sent = True
                    yield chunk_text
            else:
                try:
                    stream = await get_openrouter_client().chat.completions.create(**pass2_kwargs)
                    async for chunk in stream:
                        if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                            r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                            if isinstance(chunk.usage, dict):
                                r_tokens = chunk.usage.get("reasoning_tokens", 0)
                            if r_tokens:
                                yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                        if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, "content", None):
                            out_chunk = self._sanitize_output_text(chunk.choices[0].delta.content, trim_outer=False)
                            pass2_output += out_chunk
                            if not pass2_followups_preview_sent:
                                preview_payload = self._build_followup_payload(pass2_output, mode, session_id, persist=False)
                                if preview_payload.get("followups") or preview_payload.get("action_chips"):
                                    yield f"[INFO]{json.dumps(preview_payload)}"
                                    pass2_followups_preview_sent = True
                            yield out_chunk
                except Exception as stream_err:
                    print(f"[Processor] Stream failed in pass 2: {stream_err}")
                    yield f"\n\n?? **Stream interrupted:** {str(stream_err)}"
            # End thinking return path
            followup_payload = self._build_followup_payload(pass2_output, mode, session_id)
            yield f"[INFO]{json.dumps(followup_payload)}"
            return

        # === SAFE CHAT AGENT: Single-shot path for STRICT limits ===
        if constraint_result and constraint_result.has_strict():
            create_kwargs["stream"] = False
            create_kwargs = self._guard_kwargs(create_kwargs, user_query)

            import time as _time_mod
            _ss_start = _time_mod.time()
            _max_latency = 30.0  # 2x typical streaming ~15s
            _retry_budget = 2

            response = await client.chat.completions.create(**create_kwargs)
            full_text = self._sanitize_output_text(response.choices[0].message.content or "")

            # Validate
            vresult = output_validator.validate(
                full_text, constraint_result.active_rules,
                intent=constraint_result.intent, query=user_query
            )

            # Retry loop
            retry_count = 0
            while not vresult.passed and output_validator.should_retry(
                vresult, constraint_result.impossibility_tier, retry_count, _retry_budget
            ):
                elapsed = _time_mod.time() - _ss_start
                if elapsed > _max_latency:
                    _retry_budget = min(_retry_budget, 1)  # Reduce budget on timeout
                    if retry_count >= _retry_budget:
                        break

                retry_count += 1
                retry_mode = output_validator.get_retry_mode(retry_count)

                if DEBUG_SAFE_AGENT:
                    print(f"[SafeAgent] Retry {retry_count}: mode={retry_mode}, violations={vresult.violations}")

                retry_prompt = output_validator.build_retry_prompt(
                    user_query, full_text, vresult.violations,
                    constraint_result.active_rules
                )
                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": retry_prompt}
                ]

                if retry_mode == "streaming-trim":
                    # Retry #2: streaming fallback + trim
                    retry_kwargs = {
                        "model": model_to_use, "messages": retry_messages,
                        "stream": True
                    }
                    if temperature is not None:
                        retry_kwargs["temperature"] = max(0.1, (temperature or 0.5) - 0.2)
                    retry_kwargs = self._guard_kwargs(retry_kwargs, user_query)
                    stream_retry = await client.chat.completions.create(**retry_kwargs)
                    streamed_text = ""
                    async for chunk in stream_retry:
                        if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                            r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                            if isinstance(chunk.usage, dict):
                                r_tokens = chunk.usage.get("reasoning_tokens", 0)
                            if r_tokens:
                                yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                        if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, "content", None):
                            streamed_text += chunk.choices[0].delta.content
                    full_text = self._sanitize_output_text(streamed_text)
                else:
                    # Retry #1: single-shot tightened
                    retry_kwargs = {
                        "model": model_to_use, "messages": retry_messages,
                        "stream": False
                    }
                    if temperature is not None:
                        retry_kwargs["temperature"] = max(0.1, (temperature or 0.5) - 0.2)
                    retry_kwargs = self._guard_kwargs(retry_kwargs, user_query)
                    retry_resp = await client.chat.completions.create(**retry_kwargs)
                    full_text = self._sanitize_output_text(retry_resp.choices[0].message.content or "")

                vresult = output_validator.validate(
                    full_text, constraint_result.active_rules,
                    intent=constraint_result.intent, query=user_query
                )
                vresult.retries_used = retry_count

            # Use trimmed version
            final_output = vresult.trimmed_response or full_text
            if vresult.clarity_note:
                final_output += vresult.clarity_note

            if DEBUG_SAFE_AGENT:
                print(f"[SafeAgent] Final: passed={vresult.passed} retries={retry_count} trimmed={vresult.was_trimmed}")

            # Yield as chunks
            for i in range(0, len(final_output), 800):
                chunk_text = final_output[i:i+800]
                streamed_output += chunk_text
                yield chunk_text
            followup_payload = self._build_followup_payload(final_output, mode, session_id)
            yield f"[INFO]{json.dumps(followup_payload)}"
            return

        # Normal streaming path (no STRICT limits or thinking pass)
        create_kwargs = self._guard_kwargs(create_kwargs, user_query)
        
        # ?? PROVIDER ERROR RECOVERY: Retry on 429/timeout
        stream = None
        for _attempt in range(2):
            try:
                stream = await client.chat.completions.create(**create_kwargs)
                break
            except Exception as api_err:
                err_str = str(api_err).lower()
                is_retryable = any(k in err_str for k in ["429", "rate limit", "timeout", "timed out", "502", "503", "overloaded"])
                if is_retryable and _attempt == 0:
                    wait_time = 2.0
                    print(f"[RECOVERY] Provider error (attempt 1): {api_err}. Retrying in {wait_time}s...")
                    yield "[INFO]{\"provider_retry\": true}"
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[RECOVERY] Provider error (final): {api_err}")
                    yield f"?? The AI provider is temporarily unavailable. Please try again in a moment.\n\n*Error: {type(api_err).__name__}*"
                    return
        
        if stream is None:
            yield "?? Failed to connect to AI provider after retries."
            return

        print(f"[LATENCY] Internal: Stream Connection Established: {time.time() - start_time:.4f}s")
        
        emit_raw_reasoning = show_reasoning_panel
        in_reasoning = False
        allow_reasoning_tokens = show_reasoning_panel
        
        # Performance Tracking
        first_token_time = None
        tokens_yielded = 0
        followups_preview_sent = False
        
        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
                print(f"[METRICS] Normal Stream TTFT: {first_token_time - start_time:.4f}s")
            if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                if isinstance(chunk.usage, dict):
                    r_tokens = chunk.usage.get("reasoning_tokens", 0)
                if r_tokens:
                    yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                    
            if not hasattr(chunk, "choices") or not chunk.choices: continue
            delta = chunk.choices[0].delta
            
            # Check for GLM/model built-in reasoning tokens
            if allow_reasoning_tokens:
                reasoning_chunks_to_emit = []
                reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                if reasoning:
                    reasoning_chunks_to_emit.append(reasoning)
                details = getattr(delta, 'reasoning_details', None)
                if details:
                    for item in details:
                        if isinstance(item, dict):
                            text_part = item.get("text") or item.get("summary")
                        else:
                            text_part = getattr(item, "text", None) or getattr(item, "summary", None)
                        if text_part:
                            reasoning_chunks_to_emit.append(text_part)
                if reasoning_chunks_to_emit:
                    for reasoning in reasoning_chunks_to_emit:
                        if emit_raw_reasoning:
                            if not in_reasoning:
                                in_reasoning = True
                                trace.log("REASONING_START", f"model={model_to_use} (built-in)")
                                yield "[THINKING]"
                            yield self._sanitize_output_text(reasoning, trim_outer=False)
                        else:
                            if not in_reasoning:
                                in_reasoning = True
                                trace.log("REASONING_START", f"model={model_to_use} (built-in)")
            
            # Regular content
            if delta.content:
                if in_reasoning:
                    in_reasoning = False
                    trace.log("REASONING_COMPLETE", "built-in reasoning done")
                    if emit_raw_reasoning:
                        yield "[/THINKING]"
                sanitized_chunk = self._sanitize_output_text(delta.content, trim_outer=False)
                streamed_output += sanitized_chunk
                if not followups_preview_sent:
                    preview_payload = self._build_followup_payload(streamed_output, mode, session_id, persist=False)
                    if preview_payload.get("followups") or preview_payload.get("action_chips"):
                        yield f"[INFO]{json.dumps(preview_payload)}"
                        followups_preview_sent = True
                yield sanitized_chunk
                tokens_yielded += 1
                
        # Close reasoning if stream ended during reasoning
        if in_reasoning:
            trace.log("REASONING_COMPLETE", "built-in reasoning done")
            if emit_raw_reasoning:
                yield "[/THINKING]"
                
        if first_token_time:
            end_time = time.time()
            duration = end_time - first_token_time
            tps = tokens_yielded / duration if duration > 0 else 0
            print(f"[METRICS] Normal Stream Speed: {tps:.1f} tokens/sec ({tokens_yielded} tokens in {duration:.2f}s)")
        
        followup_payload = self._build_followup_payload(streamed_output, mode, session_id)
        yield f"[INFO]{json.dumps(followup_payload)}"

        # End of Unified Normal Mode
        return


    async def _process_agent_mode(
        self,
        user_query: str,
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        start_time: float = 0.0,
        task_id: Optional[str] = None,
        resume_graph: Optional[Any] = None,
        mode: str = "normal",
        intent: str = "AGENT",
        sub_intent: str = "general"
    ) -> AsyncGenerator[str, None]:
        """
        Structured Action Agent mode.
        Runs the 16-layer decision pipeline, then generates via LLM.
        Includes tool execution interceptor, completion guard, and failure transparency.
        """
        import time as _time
        import json as _json
        from app.agent.tool_executor import (
            parse_tool_calls, execute_tool, format_tool_result,
            ExecutionContext,
        )
        MAX_AGENT_STEPS = 50  # Completion Guard: generous cognitive cycles
        MAX_TOTAL_TOOL_CALLS = 100  # Effectively unlimited tool invocations per turn
        is_casual_query = _is_casual_query(user_query)
        is_profile_query = _is_profile_query(user_query)

        # Adaptive verification depth to avoid meta-verification loops on simple prompts.
        MIN_AGENT_STEPS = 1
        if _is_factual_lookup_query(user_query) or self._estimate_query_complexity(user_query) > 0.58:
            MIN_AGENT_STEPS = 3
        if is_casual_query or is_profile_query:
            MIN_AGENT_STEPS = 1

        if _is_recall_query(user_query):
            recall = _build_recall_response(context_messages, user_query)
            if recall:
                yield recall
                return

        if is_casual_query:
            q = (user_query or "").strip().lower()
            if "how are" in q or "how r u" in q:
                yield "Hey macha! I'm good. Sollu, what do you need?"
            else:
                yield "Hey macha! Sollu, what do you need?"
            return

        # --- Run agent pipeline (classify ? time ? autonomy ? orchestrate) ---
        pipeline_start = _time.time()
        agent_result = await run_agent_pipeline(
            user_query,
            context_messages=context_messages,
            intent=intent,
            sub_intent=sub_intent,
            user_id=user_id,
            session_id=session_id,
            session_start_time=start_time,
        )

        # Tool access policy by mode
        # - Agent mode: full tool access (as decided by pipeline)
        # - Normal/Business: limit to essential tools only
        if mode == "normal" and agent_result.tool_allowed:
            normal_tools = {
                "get_current_time",
                "search_web",
                "calculate",
            }
            agent_result.allowed_tools = [
                t for t in agent_result.allowed_tools if t in normal_tools
            ]
            if not agent_result.allowed_tools:
                agent_result.tool_allowed = False

        if mode == "business" and agent_result.tool_allowed:
            business_tools = {
                "get_current_time",
                "search_web",
                "search_news",
                "search_weather",
                "search_finance",
                "search_currency",
                "search_company",
                "search_documents",
                "web_fetch",
                "calculate",
                "extract_tables",
                "summarize_url",
                "document_compare",
                "faq_builder",
            }
            agent_result.allowed_tools = [
                t for t in agent_result.allowed_tools if t in business_tools
            ]
            if not agent_result.allowed_tools:
                agent_result.tool_allowed = False
        # Force tool availability for factual lookups in agent execution.
        if _is_factual_lookup_query(user_query) and not is_casual_query:
            agent_result.tool_allowed = True
            for _tool in ["search_web", "search_company", "search_legal", "search_documents", "get_current_time"]:
                if _tool not in agent_result.allowed_tools:
                    agent_result.allowed_tools.append(_tool)
        # Override tool classification if we are resuming a deterministic graph node
        if resume_graph:
            agent_result.tool_allowed = True
            if not agent_result.allowed_tools:
                agent_result.allowed_tools = ["search_web", "read_file", "calculate", "get_current_time"] 
            if agent_result.action_decision:
                agent_result.action_decision.action_type = "ACTION"
            
        print(f"[Agent] Pipeline complete in {_time.time() - pipeline_start:.4f}s | "
              f"type={agent_result.action_decision.action_type if agent_result.action_decision else 'N/A'} | "
              f"decision={agent_result.decision} | "
              f"delegation={agent_result.delegation_active} | "
              f"tools={'ENABLED' if agent_result.tool_allowed else 'DISABLED'}")

        # --- Register Execution Context early (needed by metadata + tool dispatch) ---
        exec_ctx = agent_result.execution_context
        exec_ctx.user_id = user_id or ""  # PH 4: Pass user context to tools
        exec_ctx.session_id = session_id or ""

        # --- Emit agent intel metadata ---
        # Map action_type to IntelligenceBar-compatible mode

        action_type = agent_result.action_decision.action_type if agent_result.action_decision else "QUESTION"
        _mode_map = {
            "TASK": "reasoning",
            "ACTION": "system_design",
            "QUESTION": "general",
            "CODING": "coding",
            "DEBUGGING": "debugging",
            "RESEARCH": "research",
            "ANALYSIS": "analysis",
            "CREATIVE": "creative",
        }
        ui_mode = _mode_map.get(action_type, "general")
        
        # Derive confidence from strategy or autonomy
        _confidence = 0.85
        _strategy = getattr(agent_result, "strategy", None)
        if _strategy and hasattr(_strategy, 'research_confidence'):
            _confidence = _strategy.research_confidence
        elif agent_result.autonomy and agent_result.autonomy.risk_tier == "high":
            _confidence = 0.5
        elif agent_result.autonomy and agent_result.autonomy.risk_tier == "medium":
            _confidence = 0.7
        
        intel_payload = {
            "mode": ui_mode,
            "action_type": action_type,
            "decision": agent_result.decision,
            "delegation": agent_result.delegation_active,
            "tool_allowed": agent_result.tool_allowed,
            "risk_tier": agent_result.autonomy.risk_tier if agent_result.autonomy else "low",
            "reversible": agent_result.autonomy.reversible if agent_result.autonomy else True,
            "time_sensitive": agent_result.temporal.is_time_sensitive if agent_result.temporal else False,
            # IntelligenceBar-compatible fields
            "confidence": _confidence,
            "debug_active": action_type == "DEBUGGING",
            "skill_level": 0.8 if action_type in ["TASK", "CODING", "DEBUGGING"] else 0.5,
            "proactive_help": action_type == "TASK" and bool(agent_result.action_decision and agent_result.action_decision.subtasks),
        }
        yield f"[INFO]INTEL:{_json.dumps(intel_payload)}"

        # --- Register Execution Profile ---
        import uuid
        execution_id = str(uuid.uuid4())
        active_executions[execution_id] = {"ctx": exec_ctx, "created_at": _time.time()}
        yield f'[INFO]{_json.dumps({"execution_id": execution_id})}'

        if getattr(exec_ctx, 'memory_hits', 0) > 0:
            yield '[INFO]{"memory_used": true}'
        
        # State: Initializing
        yield f"[INFO]{_json.dumps({'agent_state': 'initializing', 'topic': 'Configuring agent execution context'})}"

        # State: Planning
        if agent_result.action_decision and agent_result.action_decision.action_type in ["ACTION", "TASK"]:
            yield f"[INFO]{_json.dumps({'agent_state': 'planning', 'topic': f'Developing strategy for {action_type.lower()} task'})}"
            
            # Plan Preview
            if agent_result.action_decision.action_type == "TASK" and agent_result.action_decision.subtasks:
                exec_ctx.internal_plan = agent_result.action_decision.subtasks
                yield f"[INFO]{_json.dumps({'agent_state': 'plan_preview', 'plan': agent_result.action_decision.subtasks})}"

        # --- Handle confirm / ask decisions (short-circuit) ---
        if agent_result.decision == "confirm":
            yield agent_result.message
            return

        if agent_result.decision == "ask":
            yield agent_result.message
            return

        # --- Build system prompt with runtime context + tool mode ---
        system_prompt = build_agent_system_prompt(
            agent_result,
            mode=mode,
            sub_intent=sub_intent,
            user_settings=user_settings,
            user_id=user_id,
            user_query=user_query,
            personality=personality,
            session_id=session_id,
        )

        # --- Prepare messages ---
        optimized_context = context_messages
        if context_messages and len(context_messages) > 4:
            optimized_context = context_optimizer.optimize(context_messages, keep_last_n=4)
        if is_casual_query:
            optimized_context = []

        messages = [{"role": "system", "content": system_prompt}]
        if optimized_context:
            messages.extend(optimized_context)
        messages.append({"role": "user", "content": user_query})

        # Cost guard: preserve system + primary skill capsule, trim context if over budget.
        est_tokens = rough_estimate_tokens(messages)
        if est_tokens > 4500 and optimized_context:
            trimmed_context = context_optimizer.optimize(optimized_context, keep_last_n=2)
            messages = [{"role": "system", "content": system_prompt}] + trimmed_context + [{"role": "user", "content": user_query}]

        # --- Hybrid Strategy: Inject reasoning / emit metadata ---
        strategy = getattr(agent_result, "strategy", None)
        if strategy:
            yield f'[INFO]{_json.dumps({"strategy_mode": strategy.planning_mode})}'

            # PARALLEL_REASONING: inject structured reasoning context
            if strategy.planning_mode == "PARALLEL_REASONING" and strategy.reasoning_context:
                rc = strategy.reasoning_context
                messages.append({"role": "system", "content": (
                    "Planning Insight:\n"
                    f"- Performance: {rc.get('performance', 'N/A')}\n"
                    f"- Structure: {rc.get('structure', 'N/A')}\n"
                    f"- Risk: {rc.get('risk', 'N/A')}\n"
                    f"- Alternatives: {rc.get('alternatives', 'N/A')}\n"
                    "Use this context while forming your answer."
                )})

        # --- INIT PHASE 4B MEMORY OUTCOME TRACKER ---
        memory_outcome = {
            "strategy_mode": strategy.planning_mode if strategy else None,
            "confidence_drifted": False,
            "research_used": False,
            "research_text": "",
            "output_text": "",
            "repair_attempts": []
        }

        # --- NEW CONTEXT FOR TASK_STATE (PHASE 4C) ---
        # task_id is already passed as an argument
        
        # Initialize task tracking for multi-step strategies
        if session_id and task_id and strategy and strategy.planning_mode in ["ADAPTIVE_CODE_PLAN", "SEQUENTIAL_PLAN"]:
            from app.state.task_state_engine import create_task_state
            # Only create if it doesn't already exist (e.g. not a resume)
            create_task_state(session_id, task_id, strategy.planning_mode)
            
            # --- PHASE 4D: Transaction Start ---
            from app.state.transaction_manager import begin_transaction
            try:
                begin_transaction(session_id, task_id)
            except Exception as e:
                print(f"[Transaction] Warning: {e}")

        # --- RESEARCH PHASE: Execute prior to LLM generation ---
        # Gating: Only auto-execute if research confidence > 0.7
        if agent_result.tool_allowed and strategy and strategy.research_needed and strategy.research_confidence > 0.7:
            yield f'[INFO]{_json.dumps({"agent_state": "researching", "topic": strategy.research_needed.get("mode")})}'
            
            from app.agent.tool_executor import ToolCall, execute_tool, format_tool_result
            from app.research.cache import get_cached, set_cache
            from app.research.source_ranker import compute_trust_score, enforce_domain_diversity
            from app.research.recency import compute_recency_score, final_source_score, parse_date_heuristic
            from app.research.chunker import chunk_text
            from app.research.conflict_detector import detect_conflicts, adjust_confidence_for_conflicts

            r_tool = strategy.research_needed.get("tool", "search_web")
            r_prefix = strategy.research_needed.get("query_prefix", "")
            r_query = f"{r_prefix} {user_query}".strip()
            
            cached = get_cached(r_query)
            if cached:
                formatted_research = cached
                yield f'[INFO]{_json.dumps({"agent_state": "research_cache_hit"})}'
            else:
                research_call = ToolCall(name=r_tool, args=r_query, raw=f'{r_tool}("{r_query}")')
                research_result = await execute_tool(research_call, agent_result.execution_context)
                
                if research_result.success and isinstance(research_result.data, list):
                    results = research_result.data
                    
                    # Score results
                    for idx, r in enumerate(results):
                        link = r.get("url", r.get("link", ""))
                        trust = compute_trust_score(link)
                        
                        pub_date = parse_date_heuristic(r.get("date", ""))
                        recency = compute_recency_score(pub_date)
                        
                        # Proxy for relevance (search engine implicitly sorts by relevance)
                        similarity = max(0.2, 1.0 - (idx * 0.1))
                        r["_score"] = final_source_score(similarity, trust, recency)
                        
                    # Sort internally by our trust+recency+similarity score
                    results.sort(key=lambda x: x.get("_score", 0), reverse=True)
                    
                    # Diversity and Top N
                    top_results = enforce_domain_diversity(results, max_results=3)
                    
                    # Chunk and aggregate
                    synthesized_chunks = []
                    for r in top_results:
                        title = r.get("title", "")
                        snippet = r.get("snippet", "")
                        chunks = chunk_text(snippet, max_chars=1000)
                        chunked_snippet = " ".join(chunks)
                        synthesized_chunks.append(f"Source: {title} ({r.get('link', '')})\nSnippet: {chunked_snippet}")
                        
                    # Conflict Check
                    conflicts = detect_conflicts(synthesized_chunks)
                    if conflicts:
                        strategy.research_confidence = adjust_confidence_for_conflicts(strategy.research_confidence, conflicts)
                        print("[Agent] Conflict detected in research chunks. Lowering research confidence.")
                        yield f'[INFO]{_json.dumps({"agent_state": "research_conflict_detected"})}'
                        
                    formatted_research = "\\n\\n".join(synthesized_chunks)
                    set_cache(r_query, formatted_research)
                    memory_outcome["research_used"] = True
                    memory_outcome["research_text"] = formatted_research
                else:
                    formatted_research = None
                    yield f'[INFO]{_json.dumps({"agent_state": "research_failed"})}'

            if formatted_research:
                messages.append({
                    "role": "system",
                    "content": f"Research Data Acquired:\\n{formatted_research}\\n\\nUse this data to fulfill the user's request."
                })

        # --- LLM generation with Tool Execution Interceptor ---
        client_type, model_to_use, temperature, _, route_reason = get_model_for_intent(
            "normal", "general", personality, user_query
        )
        client = get_openrouter_client() if client_type == "openai" else get_openrouter_client()

        full_response = ""
        
        # --- PHASE 5: Graph Scheduler Interception ---
        # If the strategy implies a multi-step deterministic workflow, route to DAG engine
        use_graph_engine = False
        if agent_result.tool_allowed and (resume_graph or (strategy and strategy.planning_mode in ["ADAPTIVE_CODE_PLAN", "SEQUENTIAL_PLAN"])):
            use_graph_engine = True
            from app.agent.graph_builder import compile_plan_graph
            from app.agent.graph_scheduler import run_plan_graph
            from app.state.task_state_engine import update_task_graph
            
            # Map original query/plan into a graph
            if resume_graph:
                plan_graph = resume_graph
            else:
                plan_graph = compile_plan_graph(session_id, task_id, user_query)
            
            # Seed state with new graph snapshot
            update_task_graph(session_id, task_id, plan_graph.serialize(), new_status="RUNNING")
            create_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }

            graph_messages_start_len = len(messages)
            # Graph prompt dump disabled in production logs.
            
            async for token in run_plan_graph(
                graph=plan_graph,
                strategy=strategy,
                user_query=user_query,
                messages=messages,
                agent_result=agent_result,
                client=client,
                model_to_use=model_to_use,
                create_kwargs=create_kwargs
            ):
                yield self._sanitize_output_text(token, trim_outer=False)
            
            # --- Synthesis Pass: Convert tool results into readable output ---
            # Extract tool results from messages (added by graph scheduler)
            tool_result_messages = [
                m for m in messages
                if m.get("role") == "system" and "successfully completed" in m.get("content", "")
            ]
            
            print(f"[Synthesis] Tool result messages found: {len(tool_result_messages)}")
            
            synthesis_output = ""
            synthesis_followups_preview_sent = False
            if tool_result_messages:
                # Build a CLEAN context - strip all old refusal messages to prevent context poisoning
                # Only include: system prompt (first msg), user query (last user msg), tool results, and synthesis instruction
                tool_data = "\n\n".join(m.get("content", "") for m in tool_result_messages)
                
                synthesis_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a real-time data synthesis engine. Your ONLY job is to take search results "
                            "and present them in a clear, detailed, well-formatted response.\n\n"
                            "RULES:\n- Answer the user\'s question directly first, then add short supporting bullets if needed.\n- Include specific details (numbers, names, dates, locations) only when relevant.\n- Do NOT output labels like \'TOOL#\', \'WEB SEARCH\', \'Search Result\', or raw titles/snippets.\n- Do NOT dump raw search result lists or metadata unless the user explicitly asks.\n- Only include sources when the user requests them; otherwise focus on the answer.\n- NEVER say \'I cannot access real-time data\' - the data is provided below.\n- NEVER apologize or mention knowledge cutoffs.\n- NEVER refuse. Your only purpose is to format and present the data.\n"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Original question: {user_query}\n\n--- LIVE SEARCH DATA ---\n{tool_data}\n--- END DATA ---\n\nUse the data above to answer the question. Do not list raw search results or metadata."
                    }
                ]
                
                print(f"[Synthesis] Starting synthesis LLM call with {len(tool_data)} chars of data")
                
                synthesis_kwargs = {
                    "model": model_to_use,
                    "messages": synthesis_messages,
                    "temperature": temperature,
                    "stream": True
                }
                
                try:
                    synthesis_stream = await client.chat.completions.create(**synthesis_kwargs)
                    token_count = 0
                    async for chunk in synthesis_stream:
                        if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                            r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                            if isinstance(chunk.usage, dict):
                                r_tokens = chunk.usage.get("reasoning_tokens", 0)
                            if r_tokens:
                                yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                        if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, 'content', None):
                            token_count += 1
                            out_chunk = self._sanitize_output_text(chunk.choices[0].delta.content, trim_outer=False)
                            synthesis_output += out_chunk
                            if not synthesis_followups_preview_sent:
                                preview_payload = self._build_followup_payload(synthesis_output, mode, session_id, persist=False)
                                if preview_payload.get("followups") or preview_payload.get("action_chips"):
                                    yield f"[INFO]{_json.dumps(preview_payload)}"
                                    synthesis_followups_preview_sent = True
                            yield out_chunk
                    print(f"[Synthesis] Completed. Streamed {token_count} tokens.")
                except Exception as e:
                    print(f"[Synthesis] ERROR: {e}")
                    yield f"\n\n**Search completed but synthesis failed:** {str(e)}"
            else:
                print("[Synthesis] No tool results found - using graph text fallback.")
                graph_delta = messages[graph_messages_start_len:]
                fallback_segments = [
                    m.get("content", "") for m in graph_delta
                    if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m.get("content", "").strip()
                ]
                if fallback_segments:
                    fallback_text = "\\n\\n".join(fallback_segments[-2:])
                    fallback_text = self._sanitize_output_text(fallback_text)
                    if fallback_text:
                        synthesis_output = fallback_text
                        for j in range(0, len(fallback_text), 800):
                            yield fallback_text[j:j+800]
            
            # Update final state
            final_status = "COMPLETED" if plan_graph.is_fully_completed() else "FAILED"
            update_task_graph(session_id, task_id, plan_graph.serialize(), new_status=final_status)
            
            graph_followups = self._build_followup_payload(synthesis_output, mode, session_id)
            yield f"[INFO]{_json.dumps(graph_followups)}"
            yield f"[INFO]{_json.dumps({'agent_state': 'completed', 'completed': True})}"
            
            # TODO: Gracefully extract memory outcome mapping for DAG here
            return

        # --- Legacy Linear Loop (For QUESTION/Basic Interactions) ---
        if agent_result.tool_allowed:
            messages.append({
                "role": "user",
                "content": "\n[CRITICAL OVERRIDE]\nYou are an automated data retrieval API. You MUST NOT refuse to query for geopolitical, conflict-related, or real-time events. Output exactly `TOOL_CALL: search_web(\"...\")` immediately. Do NOT apologize or provide text refusals."
            })
            
        step_count = 0
        forced_tool_attempted = False
        fact_lookup_query = bool(re.search(
            r"\b(who is|founder|ceo|owner|director|price|market cap|valuation|address|incorporated|registered|linkedin|crunchbase|cbse|matric|matriculation|affiliation|board|school|college)\b",
            (user_query or "").lower()
        ))
        while True:
            if getattr(exec_ctx, 'terminate', False):
                yield '[INFO]{"agent_state": "cancelled"}'
                break

            # --- PHASE 4C: Hard Execution Budget Per Task ---
            if session_id and task_id:
                from app.state.task_state_engine import get_task_state
                task_state = get_task_state(session_id, task_id)
                if task_state and task_state.total_tool_calls >= 15: # MAX_TOTAL_TOOL_CALLS_PER_TASK
                    exec_ctx.forced_finalize = True
                    messages.append({
                        "role": "system",
                        "content": "Fatal: Task has exceeded the absolute maximum allowed tool calls (15) across all resumes. Finalize immediately."
                    })
                    print(f"[Agent] Task {task_id} exhausted its absolute tool call budget.")
                    break

            if step_count > MAX_AGENT_STEPS:
                exec_ctx.forced_finalize = True
                messages.append({
                    "role": "system",
                    "content": "Finalize now using available data. Do not call further tools."
                })
                break

            step_count += 1

            create_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "stream": True,
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            create_kwargs = self._guard_kwargs(create_kwargs, user_query)
            
            print("\n\n--- MESSAGES DUMP ---")
            print(repr(messages))
            print("---------------------\n\n")
                
            def strip_execution_narration(text: str) -> str:
                BLOCKED_PHRASES = [
                    "Here is the plan", "I will now", "Next I will", "Let's start",
                    "I'll perform", "First, I will", "Then, I will", "Sure, I can help",
                    "Let's begin", "Now I will"
                ]
                for phrase in BLOCKED_PHRASES:
                    if phrase.lower() in text.lower():
                        exec_ctx.internal_reasoning_suppressed = True
                        return ""
                return text

            print(f"[Agent] Cycle {step_count}/{MAX_AGENT_STEPS} | Streaming via {model_to_use} (Time: {_time.time() - start_time:.4f}s)")
            yield f"[INFO]{_json.dumps({'agent_state': 'reasoning', 'topic': f'Reasoning cycle {step_count}'})}"

            step_output = ""
            tool_detected = False
            suppress_step_stream = agent_result.tool_allowed or step_count < MIN_AGENT_STEPS
            try:
                stream = await client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                        r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                        if isinstance(chunk.usage, dict):
                            r_tokens = chunk.usage.get("reasoning_tokens", 0)
                        if r_tokens:
                            yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                    if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, 'content', None):
                        token = self._sanitize_output_text(chunk.choices[0].delta.content, trim_outer=False)
                        step_output += token
                        if not suppress_step_stream:
                            clean_chunk = strip_execution_narration(token)
                            if clean_chunk:
                                yield clean_chunk
            except Exception as step_err:
                print(f"[Agent] Step {step_count} failed: {step_err}")
                yield f"\n\n?? **Agent step failed:** {str(step_err)}"
                break # exit current loop to either finalize or retry

            # --- Confidence Drift & Misclassification Detection ---
            if step_count == 1 and strategy and strategy.strategy_confidence >= 0.5:
                # If we expected code but got pure theory (no code blocks), intent drifted
                if strategy.planning_mode == "ADAPTIVE_CODE_PLAN" and "```" not in step_output:
                    print("[Agent] Confidence Drift: Expected code but received theory. Downgrading strategy.")
                    yield f'[INFO]{_json.dumps({"confidence_drift": True})}'
                    strategy.strategy_confidence = 0.3
                    strategy.validation_required = False
                    if "enabled" in strategy.repair_policy:
                        strategy.repair_policy["enabled"] = False
                    memory_outcome["confidence_drifted"] = True
                    yield f'[INFO]{_json.dumps({"strategy_drift": "theory_fallback"})}'

            # --- Post-stream Tool Interceptor (Parallel Execution) ---
            if "TOOL_CALL:" in step_output:
                # Parse ALL tool calls from the step output
                tool_calls = parse_tool_calls(step_output)
                
                if not tool_calls:
                    full_response += step_output
                    break

                # Filter out disallowed tools
                valid_calls = []
                for tc in tool_calls:
                    if not agent_result.tool_allowed or tc.name not in agent_result.allowed_tools:
                        print(f"[Agent] TOOL_CALL ignored - not in allowed list (tried: {tc.name})")
                    else:
                        tc.user_id = user_id or "" # PH 4: Pass context to each call
                        tc.session_id = session_id or ""
                        valid_calls.append(tc)


                if not valid_calls:
                    full_response += step_output
                    break

                tool_detected = True
                if mode == "normal" and len(valid_calls) > 1:
                    valid_calls = [valid_calls[0]]

                # Record the pre-tool text into full_response
                first_call_idx = step_output.find("TOOL_CALL:")
                before_tool = step_output[:first_call_idx].strip()
                full_response += before_tool
                truncated_step_output = step_output.strip()

                # Budget check
                exec_ctx.tool_calls_made += len(valid_calls)
                if exec_ctx.tool_calls_made > MAX_TOTAL_TOOL_CALLS:
                    exec_ctx.forced_finalize = True
                    exec_ctx.degraded = True
                    exec_ctx.degradation_reasons.append("Max total tool calls reached")
                    print(f"[Agent] Max tool calls ({MAX_TOTAL_TOOL_CALLS}) reached, forcing finalize")
                    yield f"[INFO]{_json.dumps({'agent_state': 'finalizing'})}"
                    messages.append({"role": "assistant", "content": truncated_step_output})
                    messages.append({"role": "user", "content": "You have reached the maximum tool call budget. Finalize the best possible answer now."})
                    continue

                # Same-tool Loop Breaker (per-tool, still 3 max)
                _loop_broken = False
                for tc in valid_calls:
                    same_call_count = sum(1 for r in exec_ctx.tool_results if r.tool_name == tc.name)
                    if same_call_count >= 3:
                        exec_ctx.forced_finalize = True
                        exec_ctx.degraded = True
                        exec_ctx.degradation_reasons.append(f"Tool loop blocked: {tc.name}")
                        print(f"[Agent] Tool loop breaker triggered for {tc.name}")
                        yield f"[INFO]{_json.dumps({'agent_state': 'finalizing'})}"
                        messages.append({"role": "assistant", "content": truncated_step_output})
                        messages.append({"role": "user", "content": f"You are repeating the {tc.name} tool uselessly. Finalize the best possible answer now."})
                        _loop_broken = True
                        break
                if _loop_broken:
                    continue

                print(f"[Agent] Executing {len(valid_calls)} tools in PARALLEL: {[tc.name for tc in valid_calls]} "
                      f"[total calls: {exec_ctx.tool_calls_made}/{MAX_TOTAL_TOOL_CALLS}]")

                # Emit info for each tool
                for tc in valid_calls:
                    yield f"[INFO]{_json.dumps({'agent_state': 'using_tool', 'tool': tc.name})}"

                # --- STATIC VALIDATION: Pre-execution safety gate ---
                if strategy and strategy.validation_required:
                    val_context = {"generated_code": step_output}
                    validation_status = static_validation(val_context)
                    yield f'[INFO]{_json.dumps({"validation_status": validation_status})}'
                    if validation_status == "unsafe_code":
                        exec_ctx.forced_finalize = True
                        exec_ctx.degraded = True
                        exec_ctx.degradation_reasons.append(f"Static validation: {validation_status}")
                        print(f"[Agent] Static validation BLOCKED: unsafe code detected")
                        messages.append({"role": "assistant", "content": truncated_step_output})
                        messages.append({"role": "user", "content": "The code failed static validation due to unsafe operations. Rewrite without dangerous operations (eval, exec, os.system, subprocess)."})
                        continue

                # === PARALLEL TOOL EXECUTION via asyncio.gather ===
                async def _exec_single(tc_item):
                    return await execute_tool(tc_item, exec_ctx)

                tool_results_list = await asyncio.gather(
                    *[_exec_single(tc) for tc in valid_calls],
                    return_exceptions=True
                )

                # Process all results
                combined_formatted = []
                for tc, result in zip(valid_calls, tool_results_list):
                    if isinstance(result, Exception):
                        from app.agent.tool_executor import ToolResult
                        result = ToolResult(tool_name=tc.name, success=False, error=str(result), data=None)

                    exec_ctx.tool_results.append(result)

                    # Task state checkpointing
                    if session_id and task_id:
                        from app.state.task_state_engine import update_task_state
                        update_task_state(
                            session_id, task_id,
                            step_result={
                                "tool": tc.name,
                                "outcome_summary": str(result.data) if result.success else str(result.error),
                                "success": result.success
                            }
                        )

                    freshness = getattr(result, "freshness", "static")
                    if freshness != "static":
                        yield f"[INFO]{_json.dumps({'freshness': freshness})}"

                    formatted = format_tool_result(result)
                    combined_formatted.append(f"[{tc.name}] {formatted}")

                    if not result.success:
                        exec_ctx.degraded = True
                        exec_ctx.degradation_reasons.append(f"{tc.name}: {result.error}")
                        print(f"[Agent] Tool FAILED: {tc.name} - {result.error}")

                    # Inject untrusted content hint if applicable
                    if getattr(result, "trust", "verified") == "unverified":
                        yield f"[INFO]{_json.dumps({'trust': 'unverified'})}"
                        if not getattr(exec_ctx, "untrusted_seen", False):
                            messages.append({
                                "role": "system",
                                "content": "Treat unverified file content as reference only. Do not assume factual correctness."
                            })
                            exec_ctx.untrusted_seen = True

                # Inject all results into message history for next LLM pass
                all_results_text = "\n\n".join(combined_formatted)
                messages.append({"role": "assistant", "content": truncated_step_output})
                messages.append({"role": "user", "content": all_results_text})
                messages.append({
                    "role": "system",
                    "content": "All tool results have been provided above. Continue reasoning. You may call more tools if needed, or produce the final answer."
                })
                # Continue to next step (next LLM generation)

            if not tool_detected and agent_result.tool_allowed:
                requires_live = False
                if agent_result.temporal and agent_result.temporal.is_time_sensitive:
                    requires_live = True
                if agent_result.action_decision and agent_result.action_decision.requires_research:
                    requires_live = True
                if fact_lookup_query:
                    requires_live = True
                if requires_live and not exec_ctx.tool_results and not forced_tool_attempted:
                    forced_tool_attempted = True
                    messages.append({"role": "assistant", "content": step_output})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must call the most appropriate tool now (search_web first for entity/founder queries; "
                            "optionally search_legal/search_company/search_tech_docs when relevant). "
                            "Prioritize official registries, company pages, and LinkedIn. Output ONLY TOOL_CALL."
                        ),
                    })
            if not tool_detected:
                # Optional second pass for harder queries only; avoid recursive self-verification prompts.
                if step_count < MIN_AGENT_STEPS:
                    messages.append({"role": "assistant", "content": step_output})
                    messages.append({
                        "role": "system",
                        "content": (
                            "If more evidence is required for factual claims, call an appropriate tool now. "
                            "Otherwise finalize in the next response. "
                            "Do not output verification summaries, self-audits, or process commentary."
                        )
                    })
                    continue

                # No tool call ? stream completed naturally
                full_response += step_output
                if suppress_step_stream and step_output.strip():
                    yield step_output
                break  # exit step loop

            # --- Completion Guard: counts LLM steps + tool calls + retries ---
            if step_count >= MAX_AGENT_STEPS or exec_ctx.total_operations >= MAX_AGENT_STEPS:
                exec_ctx.forced_finalize = True
                exec_ctx.degraded = True
                exec_ctx.degradation_reasons.append("Max agent steps reached")
                print(f"[Agent] Completion guard triggered at step {step_count}")

                yield f"[INFO]{_json.dumps({'agent_state': 'finalizing'})}"
                messages.append({"role": "user", "content":
                    "You must now finalize the best possible answer using "
                    "available information. Do not plan further."
                })

                # Final forced generation
                create_kwargs["messages"] = messages
                create_kwargs = self._guard_kwargs(create_kwargs, user_query)
                stream = await client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                        r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                        if isinstance(chunk.usage, dict):
                            r_tokens = chunk.usage.get("reasoning_tokens", 0)
                        if r_tokens:
                            yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                    if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, 'content', None):
                        token = self._sanitize_output_text(chunk.choices[0].delta.content, trim_outer=False)
                        full_response += token
                        yield token
                break

        # --- FIX 3/7: Final Merge Response Pass (Silent Execution Mode) ---
        if not exec_ctx.forced_finalize and not exec_ctx.final_delivery:
            exec_ctx.final_delivery = True
            messages.append({
                "role": "system",
                "content": """
You have completed all required execution steps.

Now produce the FINAL STRUCTURED ANSWER.

Rules:
- Default to concise structure: short heading + bullet points (3-7 bullets) when informative.
- Use paragraphs only for explicitly narrative requests (essay/story/long-form).
- Do NOT describe your steps or reasoning.
- Do NOT mention tool usage.
- Do NOT narrate what you did.
- Merge all outputs logically.
- Do NOT include raw tool output or JSON; synthesize only.
- Only include sources if the user explicitly asked for them.
- Deliver one clean final response.
"""
            })

            create_kwargs["messages"] = messages
            create_kwargs = self._guard_kwargs(create_kwargs, user_query)
            try:
                final_stream = await client.chat.completions.create(**create_kwargs)
                async for chunk in final_stream:
                    if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                        r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                        if isinstance(chunk.usage, dict):
                            r_tokens = chunk.usage.get("reasoning_tokens", 0)
                        if r_tokens:
                            yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                    if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, 'content', None):
                        token = self._sanitize_output_text(chunk.choices[0].delta.content, trim_outer=False)
                        full_response += token
                        yield token
            except Exception as final_err:
                print(f"[Agent] Final pass failed: {final_err}")
                yield f"\n\n?? **Response generation failed:** {str(final_err)}"

        # --- Failure Transparency: Internal-First ---
        # Inject degradation note into prompt, let LLM phrase it naturally
        # Do not emit an additional second answer pass to users; it causes duplicated/confusing output.
        if False and exec_ctx.degraded:
            degradation_note = "[INTERNAL] Execution note: "
            if exec_ctx.forced_finalize:
                degradation_note += "Response was finalized before all steps could complete. "
            if any("failure" in r.lower() or "empty" in r.lower() or "timeout" in r.lower() for r in exec_ctx.degradation_reasons):
                degradation_note += "Some data could not be verified live. Answer synthesized from prior knowledge. "
            degradation_note += "Communicate this honestly to the user in your own words."

            print(f"[Agent] Degradation detected: {exec_ctx.degradation_reasons}")

            # Final LLM pass with degradation context
            messages.append({"role": "user", "content": degradation_note})
            create_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "stream": True,
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            create_kwargs = self._guard_kwargs(create_kwargs, user_query)
            stream = await client.chat.completions.create(**create_kwargs)
            async for chunk in stream:
                if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                    r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                    if isinstance(chunk.usage, dict):
                        r_tokens = chunk.usage.get("reasoning_tokens", 0)
                    if r_tokens:
                        yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
                if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, 'content', None):
                    token = self._sanitize_output_text(chunk.choices[0].delta.content, trim_outer=False)
                    full_response += token
                    yield token

        # --- Layer 14: Self Monitor (passive - record only) ---
        try:
            monitor_report = monitor_evaluate_response(
                user_query, full_response,
                intent=agent_result.action_decision.action_type.lower() if agent_result.action_decision else "explain",
            )
            if monitor_report.adjustments:
                print(f"[Agent] Monitor observations: {monitor_report.adjustments}")

            feedback_engine.log_interaction(
                session_id=session_id or "",
                user_id=user_id or "",
                query=user_query,
                intent="AGENT",
                sub_intent=agent_result.action_decision.action_type if agent_result.action_decision else "QUESTION",
                model_used=model_to_use,
                emotions=[],
                skill_level=0.5,
                response_length=len(full_response),
                latency_ms=(_time.time() - start_time) * 1000,
                prompt_variant="agent"
            )
        except Exception as e:
            print(f"[Agent] Monitor/feedback failed (non-blocking): {e}")

        # Signal completed state for UI execution feedback layering
        confidence_payload = {
            "confidence_metrics": {
                "tool_failures": len(exec_ctx.degradation_reasons),
                "forced_finalize": exec_ctx.forced_finalize,
                "loop_break": exec_ctx.loop_break_triggered,
                "step_count": step_count
            }
        }
        yield f'[INFO]{_json.dumps(confidence_payload)}'
        
        # --- PHASE 4B: UPDATE STRATEGY MEMORY ---
        if session_id:
            from app.memory.strategy_memory import update_strategy_memory
            memory_outcome["output_text"] = full_response
            # Inject repair history mapped from execution context if tracked
            # Note: The repair engine itself tracks the failure types into its own context, 
            # we just need to ensure `run_repair_cycle` pushes into `exec_ctx.repair_history` if available
            if hasattr(exec_ctx, "repair_history"):
                memory_outcome["repair_attempts"] = exec_ctx.repair_history
                
            update_strategy_memory(session_id, memory_outcome)
            
            # --- PHASE 4C: SET TASK STATUS ---
            if task_id:
                from app.state.task_state_engine import set_task_status
                if getattr(exec_ctx, 'terminate', False):
                    set_task_status(session_id, task_id, "PAUSED")
                elif exec_ctx.forced_finalize:
                    set_task_status(session_id, task_id, "FAILED")
                else:
                    set_task_status(session_id, task_id, "COMPLETED")
                    # --- PHASE 4D: Commit Transaction on Success ---
                    from app.state.transaction_manager import commit_transaction
                    commit_transaction(session_id, task_id)
        
        final_followups = self._build_followup_payload(full_response, mode, session_id)
        yield f"[INFO]{_json.dumps(final_followups)}"
        active_executions.pop(execution_id, None)
        yield f"[INFO]{_json.dumps({'agent_state': 'completed', 'completed': True})}"

# Global processor instance
llm_processor = LLMProcessor()







