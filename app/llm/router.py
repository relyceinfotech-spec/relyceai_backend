"""
Relyce AI - LLM Router
Routes queries to appropriate mode (Normal, Business, DeepSearch)
Imports logic from existing Python files
"""
import json
import re
import requests
import time
import asyncio
import httpx
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI
from app.llm.guards import normalize_user_query, build_guard_system_messages
from app.chat.mode_mapper import normalize_chat_mode
from app.config import (
    OPENAI_API_KEY, SERPER_API_KEY, LLM_MODEL, SERPER_TOOLS,
    OPENROUTER_API_KEY, FAST_MODEL, ERNIE_THINKING_MODEL,
    SERPER_CONNECT_TIMEOUT, SERPER_READ_TIMEOUT, SERPER_MAX_RETRIES, SERPER_RETRY_BACKOFF,
    TIEBREAKER_MODEL
)
from app.llm.embeddings import classify_emotion
from app.llm.skill_router_runtime import inject_skill_capsules, select_skills
from app.llm.prompts import (
    CORE_POLICY, BASE_FORMATTING_RULES,
    BASE_LANGUAGE_RULES,
    DEFAULT_PERSONA,
    NORMAL_SYSTEM_PROMPT, BUSINESS_SYSTEM_PROMPT, DEEPSEARCH_SYSTEM_PROMPT,
    AGENT_FORMAT_PROMPT, INTERNAL_SYSTEM_PROMPT, INTERNAL_MODE_PROMPTS
)

# Compatibility exports consumed by processor/processor_parts.
# Keep these in router to avoid startup import failures after refactors.
EMOTIONAL_BLOCK = """
Emotional adaptation:
- Detect user emotional state and adapt tone while preserving accuracy.
- If the user is frustrated/confused, simplify and prioritize actionable steps.
- If urgent, keep the response short and immediately useful.
"""

TONE_MAP = {
    "frustrated": "Tone: Calm, empathetic, and solution-focused. Avoid blame.",
    "confused": "Tone: Clarify with simple language and step-by-step guidance.",
    "excited": "Tone: Positive and energetic while keeping technical accuracy.",
    "urgent": "Tone: Brief and direct. Prioritize next best action first.",
    "curious": "Tone: Exploratory and explanatory. Include concise reasoning.",
    "casual": "Tone: Friendly and natural; avoid excessive formal structure.",
    "professional": "Tone: Professional, precise, and concise.",
    "neutral": "Tone: Clear, balanced, and practical.",
}

NORMAL_MARKDOWN_POLISH = """
Formatting polish:
- Start with a direct answer for straightforward questions.
- Use short bullets/sections only when they improve clarity.
- Avoid over-formatting short conversational replies.
"""

# Initialize clients lazily
_client: Optional[AsyncOpenAI] = None
_openrouter_client: Optional[AsyncOpenAI] = None


def _normalize_runtime_mode(mode: str) -> str:
    raw = str(mode or "").strip().lower()
    normalized = normalize_chat_mode(raw or "smart")
    if raw in {"normal", "business", "hybrid_main", "deepsearch", "research"}:
        print(f"[ModeCompat] legacy mode '{raw}' mapped to '{normalized}'")
    return normalized

def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
             raise RuntimeError("OPENAI_API_KEY not set")
        
        http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=60.0)
        )
        _client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            timeout=60.0,
            http_client=http_client
        )
    return _client

def get_openrouter_client() -> AsyncOpenAI:
    """Get OpenRouter client (uses OpenAI SDK with custom base_url)"""
    global _openrouter_client
    if _openrouter_client is None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set")
            
        http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0),
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=60.0)
        )
        _openrouter_client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://relyce.ai", 
                "X-Title": "Relyce AI",
                "Connection": "keep-alive"
            },
            timeout=60.0,
            http_client=http_client
        )
    return _openrouter_client

# ============================================
# TOOLS CONFIGURATION (from existing files)
# ============================================

# Normal mode tools
NORMAL_TOOLS = {
    "Search": SERPER_TOOLS["Search"],
    "Images": SERPER_TOOLS["Images"],
    "Videos": SERPER_TOOLS["Videos"],
    "Places": SERPER_TOOLS["Places"],
    "Maps": SERPER_TOOLS["Maps"],
    "Reviews": SERPER_TOOLS["Reviews"],
    "News": SERPER_TOOLS["News"],
    "Shopping": SERPER_TOOLS["Shopping"],
    "Scholar": SERPER_TOOLS["Scholar"],
    "Patents": SERPER_TOOLS["Patents"]
}

# Business mode tools
BUSINESS_TOOLS = {
    "Search": SERPER_TOOLS["Search"],
    "News": SERPER_TOOLS["News"],
    "Reviews": SERPER_TOOLS["Reviews"],
    "Places": SERPER_TOOLS["Places"],
    "Maps": SERPER_TOOLS["Maps"],
    "Shopping": SERPER_TOOLS["Shopping"],
    "Scholar": SERPER_TOOLS["Scholar"],
    "Patents": SERPER_TOOLS["Patents"],
    "Videos": SERPER_TOOLS["Videos"],
    "Images": SERPER_TOOLS["Images"]
}

# DeepSearch mode tools (all tools)
DEEPSEARCH_TOOLS = SERPER_TOOLS.copy()

def get_tools_for_mode(mode: str) -> Dict[str, Any]:
    """Helper to get tools for a specific mode."""
    mode = _normalize_runtime_mode(mode)
    if mode in {"agent", "research_pro"}:
        return DEEPSEARCH_TOOLS
    return NORMAL_TOOLS

# ============================================
# SYSTEM PROMPTS (from existing files)
# ============================================

# System prompts are now imported from app.llm.prompts


def get_headers() -> Dict[str, str]:
    """Get Serper API headers"""
    return {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

def _get_current_time_context() -> str:
    """Helper to get standardized time context"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def _get_user_facts(user_query: str, user_id: str) -> str:
    """Centralized helper for user memory loading (Synchronous)"""
    try:
        from app.chat.smart_memory import format_memories_for_prompt
        return format_memories_for_prompt(user_id, user_query)
    except Exception as e:
        print(f"[Router] Memory fetch error: {e}")
    return "No prior user facts found."

def _is_tanglish_query(query: str) -> bool:
    """High-speed Tanglish detection using marker list"""
    markers = ["macha", "dei", "da", "seri", "bro", "enna", "epdi", "va", "poda", "na", "than"]
    q = f" {query.lower()} "
    return any(f" {m} " in q for m in markers)

def should_use_thinking_model(query: str, sub_intent: str) -> bool:
    """
    Confidence gating: Decide if query needs Ernie thinking pass.
    Disabling 2-pass thinking to rely on OpenRouter's native reasoning feature.
    """
    return False

def _get_tech_sub_intent(user_query: str) -> Optional[str]:
    """Robust technical sub-intent extraction"""
    q = user_query.lower()
    
    # Coding Simple vs Complex
    if any(k in q for k in ["how", "explain", "why", "logic", "architecture"]):
        return "coding_complex"
    
    # SQL
    if any(k in q for k in ["sql", "query", "database", "schema", "table"]):
        return "sql"
    
    # Debugging
    if any(k in q for k in ["error", "debug", "fix", "issue", "exception", "crash"]):
        return "debugging"
        
    return "coding_simple"





async def select_tools_for_mode(user_query: str, mode: str) -> List[str]:
    """
    Select relevant tools based on chat mode.
    Imported from existing files.
    """
    user_query = normalize_user_query(user_query)
    mode = _normalize_runtime_mode(mode)
    q = user_query.lower()
    
    # âš¡ FAST PATH: Obvious tools based on keywords or intent clues
    if any(k in q for k in ["news", "latest", "today", "current", "update"]):
        return ["Search", "News"] if mode == "research_pro" else ["Search"]
    
    if any(k in q for k in ["place", "location", "near me", "map", "direction"]):
        return ["Places", "Search"] if mode == "research_pro" else ["Search"]

    guard_messages = build_guard_system_messages(user_query)

    if mode == "smart":
        tools = NORMAL_TOOLS
        mode_descriptor = "relevant tools"
    else:  # agent/research_pro
        tools = DEEPSEARCH_TOOLS
        mode_descriptor = "top 3-5 tools for comprehensive research"

    tools_list = ", ".join(tools.keys())

    if mode == "research_pro":
        system_prompt = f"""You are a Senior Research Architect.
Available Tools: [{tools_list}]
Select the top 3-5 tools that will provide the most comprehensive, detailed, and varied data.
Rules:
- ALWAYS include 'Search'.
- Include 'News' for current events.
- Include 'Scholar' or 'Patents' ONLY for technical or academic topics.
- Include 'Places' or 'Maps' for locations.
Return the tool names as a comma-separated list (e.g., 'Search, News, Videos')."""
    else:
        system_prompt = f"Select {mode_descriptor} from [{tools_list}]. Return comma-separated list."

    messages = [{"role": "system", "content": system_prompt}]
    for guard in guard_messages:
        messages.append({"role": "system", "content": guard})
    messages.append({"role": "user", "content": user_query})

    # Switch to faster model for tool selection to reduce TTFT
    response = await get_openrouter_client().chat.completions.create(
        model=TIEBREAKER_MODEL, # Use gpt-4o-mini
        messages=messages
    )

    selected_str = response.choices[0].message.content.strip()
    # Extract tokens with regex for words/phrases robustly
    cand = re.findall(r"[A-Za-z0-9\-\_ ]+", selected_str)
    selected_tools = [t.strip() for t in cand if t.strip() in tools]

    if not selected_tools:
        return ["Search", "News"] if mode == "research_pro" else ["Search"]

    return selected_tools


# ============================================
# Internal response helpers

def _select_ui_implementation_sub_intent(user_query: str) -> str:
    q = user_query.lower()
    react_keywords = [
        "react", "jsx", "tsx", "next", "next.js", "nextjs"
    ]
    if any(k in q for k in react_keywords):
        return "ui_react"
    return "ui_demo_html"

async def analyze_and_route_query(
    user_query: str, 
    mode: str, 
    context_messages: Optional[List[Dict]] = None,
    personality: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Combined Intent + Tool Selection + Sub-Intent Detection.
    Now Context-Aware: Uses recent chat history to deterime intent (Sticky Mode).
    Returns: {"intent": "INTERNAL", "sub_intent": "sql", "tools": []}
    """
    user_query = normalize_user_query(user_query)
    mode = _normalize_runtime_mode(mode)
    # 0. PERSONALITY CONTENT MODE OVERRIDE (Only in Normal Mode)
    # If a personality forces a specific behavior, we obey it immediately.
    import time
    t_start = time.time()
    
    # Robust Initialization
    needs_reasoning = False
    needs_web = False

    # Relyce AI: personalities no longer override logic branches
    # Normal chat always flows to automatic detection
    
    # ============================================
    # PARALLEL ANALYSIS: Emotion + Intent
    # ============================================
    from app.llm.embeddings import classify_emotion, classify_intent_with_timeout

    # Run in parallel for speed (with strict timeouts)
    emotion_task = asyncio.wait_for(classify_emotion(user_query), timeout=2.0)
    intent_task = classify_intent_with_timeout(user_query)

    try:
        detected_emotions, emb_result = await asyncio.gather(emotion_task, intent_task)
    except asyncio.TimeoutError:
        print("[Router] Emotion or Intent task timed out, using defaults")
        detected_emotions = ["neutral"]
        emb_result = None
    except Exception as e:
        print(f"[Router] Parallel analysis error: {e}")
        detected_emotions = ["neutral"]
        emb_result = None
    
    print(f"[Router] Emotions detected: {detected_emotions}")

    # Initial Tone Flags (Separate from Intent)
    q_lower = user_query.lower()
    is_tanglish = _is_tanglish_query(user_query)
    is_factual_q = any(m in q_lower for m in ["who is", "founder", "ceo", "owner", "board", "cbse", "matric", "price", "latest", "when was"])
    is_structured_q = any(m in q_lower for m in ["explain", "difference", "compare", "guide", "steps", "why", "quantum", "how does"])
    force_non_casual_q = (mode == "smart" and any(m in q_lower for m in ["what is", "explain", "how", "why", "difference", "tell me", "overview", "comparison"]))

    tone_flags = {"casual": False}
    if is_tanglish and not is_factual_q and not is_structured_q and not force_non_casual_q:
        tone_flags["casual"] = True
    
    try:
        if emb_result and emb_result.get("confidence", 0) >= 0.60:
            # Use embedding result
            intent = emb_result["intent"]
            needs_web = emb_result.get("needs_web", False)
            if needs_web:
                selected_tools = await select_tools_for_mode(user_query, mode)
                return {
                    "intent": "DEEP_SEARCH",
                    "sub_intent": intent,
                    "tools": selected_tools,
                    "needs_reasoning": emb_result.get("needs_reasoning", needs_reasoning),
                    "embedding_confidence": emb_result["confidence"],
                    "emotions": detected_emotions,
                    "tone_flags": tone_flags
                }

            # Map intent to sub_intent for internal routing
            sub_intent_map = {
                "casual": "casual_chat",
                "coding_simple": "code_explanation",
                "coding_complex": "debugging",
                "analysis_internal": "reasoning",
                "ui_strategy": "ui_strategy",
                "ui_implementation": "ui_implementation",
                "ui_design": "ui_strategy",
                "business": "general"
            }
            sub = sub_intent_map.get(intent, "general")
            if sub == "ui_implementation":
                sub = _select_ui_implementation_sub_intent(user_query)
            return {
                "intent": "AGENT",
                "sub_intent": sub,
                "tools": [],
                "needs_reasoning": emb_result.get("needs_reasoning", needs_reasoning),
                "embedding_confidence": emb_result["confidence"],
                "emotions": detected_emotions,
                "tone_flags": tone_flags
            }

        print(f"[Router] Embedding low confidence ({emb_result.get('confidence', 0):.2f}), using keyword fallback")
    except Exception as e:
        print(f"[Router] Embedding error, using keyword fallback: {e}")
        
    # âš¡ FAST PATH: Check for technical/simple queries to skip LLM entirely (<0.01s)
        
    # âš¡ FAST PATH: Check for technical/simple queries to skip LLM entirely (<0.01s)
    # 1.0 Robust Classifications
    q = user_query.lower().strip()
    q_compact = re.sub(r"[!?.,;:]+$", "", q).strip()
    is_factual = any(m in q for m in ["who is", "founder", "ceo", "owner", "board", "cbse", "matric", "price", "latest", "when was"])
    is_structured = any(m in q for m in ["explain", "difference", "compare", "guide", "steps", "why", "quantum", "how does"])
    
    force_non_casual = (mode == "smart" and any(m in q for m in ["what is", "explain", "how", "why", "difference", "tell me", "overview", "comparison"]))

    if force_non_casual or is_structured:
        return {"intent": "INTERNAL", "sub_intent": "general", "emotions": [e for e in detected_emotions if e != "casual"], "tone_flags": tone_flags}

    # 1.1 Greetings & Personal questions (FAST PATH)
    greeting_only = {"hi", "hello", "hey", "sup", "macha", "bro", "vanakkam", "namaste"}
    if q_compact in greeting_only:
        return {"intent": "INTERNAL", "sub_intent": "casual_chat", "emotions": detected_emotions, "tone_flags": tone_flags}
    
    # Personal IDENTITY query
    if any(p in q for p in ["who created you", "who made you", "who are you", "your name"]):
        return {"intent": "INTERNAL", "sub_intent": "casual_chat", "emotions": detected_emotions, "tone_flags": tone_flags}

    # 1.2 Technical intent detection
    def _has_tech_intent(query: str) -> bool:
        tech_keywords = [
            "html", "css", "tailwind", "bootstrap", "react", "jsx", "tsx",
            "frontend", "front end", "landing page", "hero section", "navbar", "cta",
            "website", "web design", "homepage", "ui", "ux", "wireframe", "mockup",
            "code", "build", "implement", "javascript", "js", "component",
            "fix this error", "debug this", "why is this failing", "exception:", "error:", "traceback", "stack trace",
            "select *", "join table", "sql query", "write sql", "database schema", "insert into",
            "explain this code", "how does this work", "walk me through",
            "design", "layout", "ui", "ux", "wireframe", "mockup",
            "write", "blog", "post", "email", "article",
            "mkdir", "terminal", "npm", "git", "python", "javascript", "bash", "linux"
        ]
        return any(k in query for k in tech_keywords)

    if _has_tech_intent(q):
        sub = _get_tech_sub_intent(q)
        if "ui" in q or "design" in q or "layout" in q:
            sub = _select_ui_implementation_sub_intent(q) if any(k in q for k in ["html", "css", "react", "build", "implement"]) else "ui_strategy"
        
        needs_reasoning = should_use_thinking_model(user_query, sub)
        return {"intent": "INTERNAL", "sub_intent": sub, "needs_reasoning": needs_reasoning, "emotions": detected_emotions, "tone_flags": tone_flags}

    # 1.3 Business Mode Logic
    if mode in {"agent", "research_pro"}:
        if any(t in q for t in ["code", "math", "calculate"]):
             return {"intent": "INTERNAL", "sub_intent": "general", "emotions": detected_emotions, "tone_flags": tone_flags}
        
        selected_tools = await select_tools_for_mode(user_query, mode)
        return {"intent": "AGENT", "sub_intent": "research", "tools": selected_tools, "emotions": detected_emotions, "tone_flags": tone_flags}

    # 2.0 Web Search vs Internal fallback
    search_keywords = ["search", "latest", "news", "current", "today", "price", "weather", "match", "score"]
    if any(sk in q for sk in search_keywords) or is_factual:
        selected_tools = await select_tools_for_mode(user_query, mode)
        return {"intent": "AGENT", "sub_intent": "general", "tools": selected_tools, "emotions": detected_emotions, "tone_flags": tone_flags}

    return {"intent": "AGENT", "sub_intent": "general", "emotions": detected_emotions, "tone_flags": tone_flags}


import asyncio
import functools

def execute_serper_batch_sync(endpoint_url: str, queries: List[str], param_key: str = "q") -> Dict[str, Any]:
    """
    Synchronous implementation of Serper API batch request.
    """
    payload_queries = [{param_key: q} for q in queries] if isinstance(queries, list) else [{param_key: queries}]
    payload = json.dumps(payload_queries)

    last_error = None
    for attempt in range(SERPER_MAX_RETRIES + 1):
        try:
            response = requests.post(
                endpoint_url,
                headers=get_headers(),
                data=payload,
                timeout=(SERPER_CONNECT_TIMEOUT, SERPER_READ_TIMEOUT)
            )
            if response.status_code == 200:
                return response.json()
            last_error = f"API Error {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

        if attempt < SERPER_MAX_RETRIES:
            time.sleep(SERPER_RETRY_BACKOFF * (2 ** attempt))

    return {"error": last_error or "Serper request failed"}

async def execute_serper_batch(endpoint_url: str, queries: List[str], param_key: str = "q") -> Dict[str, Any]:
    """
    Async wrapper for Serper API batch request to prevent blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, 
        functools.partial(execute_serper_batch_sync, endpoint_url, queries, param_key)
    )


def _build_user_context_string(user_settings: Optional[Dict]) -> str:
    """
    Build context string from user settings.
    Implements 'Hard Authority' for stylistic preferences while respecting safety.
    """
    if not user_settings:
        return ""
    
    # Check if settings are nested under 'personalization' or flat
    p = user_settings.get("personalization", user_settings)
    if not p:
        return ""
    
    context = []
    
    # Tone/Emoji (Strong Instructions)
    if p.get("tone") and p.get("tone") != "Default":
        context.append(f"Tone: {p['tone']}")
    
    if p.get("emoji") == "More":
        context.append(
            "Emoji usage: Default none. Use one emoji only if the user used emoji or uses slang/very casual tone. "
            "Never use emoji for technical/code/structured/instructional responses. "
            "Tone resets each message; do not carry emoji tone forward."
        )
    elif p.get("emoji") == "Less":
        context.append("Emoji usage: None, unless user explicitly asks for emojis.")

    # User Info
    if p.get("nickname"):
        context.append(f"User Nickname: {p['nickname']}")
    if p.get("occupation"):
        context.append(f"User Occupation: {p['occupation']}")
    if p.get("aboutMe"):
        context.append(f"User Bio: {p['aboutMe']}")
    
    # Imported Memories (from other AI platforms)
    memories = p.get("memories", [])
    if memories:
        mem_block = "\n".join(f"- {m}" for m in memories[:50])  # Cap at 50 to avoid prompt bloat
        context.append(f"User Memories (imported context from other AI tools):\n{mem_block}")
        
    if not context:
        return ""
        
    # The Safe Hard Authority Block
    return (
        "\n\n--- USER PREFERENCES (HIGHEST PRIORITY) ---\n" +
        "\n".join(context) + "\n\n" +
        "These preferences override any stylistic or tone instructions above,\n" +
        "as long as they remain within safety and policy boundaries.\n" +
        "-----------------------------------------\n"
    )


_SIMPLE_QUERY_KEYWORDS = (
    "what is",
    "define",
    "meaning",
    "difference",
    "example",
)


def is_simple_query(text: str, mode: str = "smart") -> bool:
    """Heuristic: classify short/straightforward prompts for compact formatting."""
    if not text:
        return True

    cleaned = text.strip().lower()
    if len(cleaned.split()) < 12:
        return True

    return any(k in cleaned for k in _SIMPLE_QUERY_KEYWORDS)


def _build_format_complexity_hint(user_query: str, mode: str = "smart") -> str:
    if is_simple_query(user_query, mode=mode):
        return (
            "\n\n**FORMAT COMPLEXITY MODE: SIMPLE**\n"
            "Use compact mode: `## Direct Answer` + up to 3 concise bullets. "
            "Do not force full templates.\n"
        )

    return (
        "\n\n**FORMAT COMPLEXITY MODE: DETAILED**\n"
        "Use the full mode template for this response. "
        "Skip any section that has no meaningful content.\n"
    )



def _build_followup_hint(mode: str, user_query: str) -> str:
    mode = _normalize_runtime_mode(mode)
    q = (user_query or "").strip().lower()
    if not q:
        return "\n\nSkip follow-ups for this reply.\n"

    # Skip on conversational one-liners to avoid UI clutter.
    skip_markers = ("hi", "hello", "hey", "thanks", "thank you", "bye")
    if len(q.split()) <= 4 and any(m in q for m in skip_markers):
        return "\n\nSkip follow-ups for this reply.\n"

    if mode == "research_pro":
        pattern = "Use Research pattern: Evidence -> Validation -> Caveat."
    elif mode == "agent":
        pattern = "Use Agent pattern: Data -> Execution -> Validation."
    else:
        pattern = "Use Smart pattern: Clarify -> Implement -> Pitfall."

    return (
        "\n\nGenerate up to 3 distinct follow-up questions.\n"
        + pattern + "\n"
        + "Do not repeat the same angle.\n"
        + "Also include up to 3 optional action chips ONLY if helpful: "
        + "[Compare options], [Show example], [Generate code].\n"
    )

def get_system_prompt_for_mode(mode: str, user_settings: Optional[Dict] = None, user_id: Optional[str] = None, user_query: str = "", session_id: Optional[str] = None) -> str:
    """Get system prompt for mode with centralized context helpers"""
    mode = _normalize_runtime_mode(mode)
    
    # 1. Select base prompt
    mode_prompts = {
        "smart": NORMAL_SYSTEM_PROMPT,
        "agent": AGENT_FORMAT_PROMPT,
        "research_pro": DEEPSEARCH_SYSTEM_PROMPT,
    }
    base_sys = mode_prompts.get(mode, NORMAL_SYSTEM_PROMPT)

    # 2. Inject Skills
    skill_selection = select_skills(user_query, mode, session_id=session_id)
    base = inject_skill_capsules(base_sys, skill_selection, token_budget=4500)

    # 3. Build Context
    time_ctx = f"\n\n**CURRENT DATE & TIME:** {_get_current_time_context()}\n"
    user_facts = f"\n\n**USER CONTEXT & MEMORY:**\n{_get_user_facts(user_query, user_id)}\n" if user_id else ""
    pref_ctx = _build_user_context_string(user_settings)
    
    # 4. Final Composition
    return base + time_ctx + user_facts + pref_ctx


def get_system_prompt_for_personality(personality: Dict[str, Any], user_settings: Optional[Dict] = None, user_id: Optional[str] = None, user_query: str = "", session_id: Optional[str] = None) -> str:
    """
    Combined BASE formatting/language rules with custom personality prompt.
    Includes specialty context and runtime-selected skill capsules.
    """
    p_id = personality.get("id")
    p_name = personality.get("name")
    
    # 1. Base Logic
    if p_id == "default_relyce" or p_name == "Relyce AI":
        return get_system_prompt_for_mode("smart", user_settings, user_id, user_query, session_id=session_id)

    custom_prompt = personality.get("prompt") or DEFAULT_PERSONA
    specialty = personality.get("specialty", "general")
    
    # 2. Inject Skills
    skill_selection = select_skills(user_query, "smart", session_id=session_id)
    custom_prompt = inject_skill_capsules(custom_prompt, skill_selection, token_budget=4500)

    # 3. Build Context
    time_ctx = f"\n\n**CURRENT DATE & TIME:** {_get_current_time_context()}\n"
    user_facts = f"\n\n**USER CONTEXT & MEMORY:**\n{_get_user_facts(user_query, user_id)}\n" if user_id else ""
    pref_ctx = _build_user_context_string(user_settings)

    # 4. Specialty Contexts
    specialty_contexts = {
        "coding": "\n**EXPERTISE: Coding & Technology**\n- You are an expert programmer. Provide clean, production-ready code.",
        "business": "\n**EXPERTISE: Business & Strategy**\n- Focus on ROI, metrics, and actionable insights.",
        "ecommerce": "\n**EXPERTISE: E-Commerce & Retail**\n- Focus on conversion and fulfillment.",
        "creative": "\n**EXPERTISE: Creative & Design**\n- Balance aesthetics with practical implementation.",
        "legal": "\n**EXPERTISE: Legal & Compliance**\n- Flag risk clearly and recommend legal counsel.",
        "health": "\n**EXPERTISE: Health & Wellness**\n- Provide evidence-informed wellness guidance.",
        "education": "\n**EXPERTISE: Education & Learning**\n- Teach step-by-step with examples."
    }
    spec_ctx = specialty_contexts.get(specialty, "")

    return f"""{CORE_POLICY}
{custom_prompt}
{spec_ctx}
{user_facts}
{time_ctx}
{BASE_LANGUAGE_RULES}
{BASE_FORMATTING_RULES}
{pref_ctx}"""

def get_internal_system_prompt_for_personality(personality: Dict[str, Any], user_settings: Optional[Dict] = None, user_id: Optional[str] = None, mode: str = "smart") -> str:
    """
    Get system prompt for INTERNAL queries with PERSONALITY.
    """
    p_id = personality.get("id")
    p_name = personality.get("name")

    # Default Relyce
    if p_id == "default_relyce" or p_name == "Relyce AI":
        base_prompt = INTERNAL_SYSTEM_PROMPT + "\n\nResponse style: keep answers concise and adaptive to user intent."
        user_facts = _get_user_facts("", user_id) if user_id else ""
        return f"{CORE_POLICY}\n{base_prompt}\n{user_facts}\n{_build_user_context_string(user_settings)}"

    # Custom Personalized
    custom_prompt = personality.get("prompt") or DEFAULT_PERSONA
    user_facts = _get_user_facts("", user_id) if user_id else ""
    
    return f"""{CORE_POLICY}
{custom_prompt}
{user_facts}
{BASE_LANGUAGE_RULES}
{_build_user_context_string(user_settings)}
**RESPONSE STYLE:**
1. Be brief and natural for casual chat.
2. For technical questions, provide clear code blocks.
3. Keep it concise."""

