"""
Relyce AI - LLM Router
Routes queries to appropriate mode (Normal, Business, DeepSearch)
Imports logic from existing Python files
"""
import json
import requests
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI
from app.config import OPENAI_API_KEY, SERPER_API_KEY, LLM_MODEL, SERPER_TOOLS

# Initialize OpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ============================================
# TOOLS CONFIGURATION (from existing files)
# ============================================

# Normal mode tools
NORMAL_TOOLS = {
    "Search": SERPER_TOOLS["Search"],
    "Places": SERPER_TOOLS["Places"],
    "Maps": SERPER_TOOLS["Maps"],
    "Reviews": SERPER_TOOLS["Reviews"],
    "News": SERPER_TOOLS["News"],
    "Shopping": SERPER_TOOLS["Shopping"],
    "Scholar": SERPER_TOOLS["Scholar"]
}

# Business mode tools
BUSINESS_TOOLS = {
    "Search": SERPER_TOOLS["Search"],
    "Places": SERPER_TOOLS["Places"],
    "Maps": SERPER_TOOLS["Maps"],
    "Reviews": SERPER_TOOLS["Reviews"],
    "News": SERPER_TOOLS["News"]
}

# DeepSearch mode tools (all tools)
DEEPSEARCH_TOOLS = SERPER_TOOLS.copy()

# ============================================
# SYSTEM PROMPTS (from existing files)
# ============================================


# ============================================
# SYSTEM PROMPTS (from existing files)
# ============================================

BASE_FORMATTING_RULES = """
**STRICT OUTPUT FORMATTING:**
- First line: Title
- Second line: Blank
- Third section: Answer
- Fourth section: Blank
- Final section: Sources (Format: Source: [Link])

**GOLDEN FORMATTING RULES:**
1. **MUST USE NORMAL TEXT (Readability):**
   - **ALL** summaries, explanations, key takeaways, and bullet points.
   - Stories, narratives, and casual conversation.
   *Reason:* Normal text is for human reading.
2. **MUST USE CODE BLOCKS (Utility):**
   - **ONLY** actual technical content: Code snippets (Python, JS, etc.), Shell commands, Config files (JSON/YAML), and raw data intended for copy-pasting.
   - **ALWAYS** use triple backticks with the language name (e.g., ```python, ```bash).
   - **NEVER** put human-readable summaries or general answers inside a code block.
"""

BASE_LANGUAGE_RULES = """
**Language Matching:** STRICTLY reply in the same language and dialect as the user. Mirror their language choice exactly (e.g., Tanglish, Hinglish, pure user language, etc.).
"""

DEFAULT_PERSONA = """You are **Relyce AI**, an elite strategic advisor.
**Core Mandate:**
Provide fact-based guidance using a hybrid of search data and your own expert internal knowledge.
**Guidelines:**
* **Synthesis:** Combine search data with internal knowledge.
* **Tone:** Friendly, professional, and advisory. Use emojis to make the conversation warm and engaging. 🌟
* **Adaptive Capability:** If the situation calls for it (e.g., user asks for a roast), you can switch to "Max Level Roast" mode and roast the user mercilessly. Otherwise, stay friendly and helpful.
"""

NORMAL_SYSTEM_PROMPT = f"""{DEFAULT_PERSONA}
{BASE_LANGUAGE_RULES}
{BASE_FORMATTING_RULES}
"""

BUSINESS_SYSTEM_PROMPT = """You are **Relyce AI**, an elite strategic advisor.
**Core Mandate:**
Provide fact-based, high-level guidance operating with:
1. **Business Acumen:** Deep understanding of market dynamics and growth strategies.
2. **Corporate Identity (Relyce AI):** You are the proprietary AI engine of **Relyce AI**.

**Guidelines:**
* **Synthesis:** Combine search data with internal knowledge.
* **Tone:** Professional, authoritative, and advisory.

{BASE_LANGUAGE_RULES}
{BASE_FORMATTING_RULES}
"""

DEEPSEARCH_SYSTEM_PROMPT = """You are **Relyce AI**, an elite strategic advisor.
You are a highly accomplished and multi-faceted AI assistant, functioning as an **elite consultant and strategic advisor** for businesses and startups. Your persona embodies the collective expertise of a Chief Operating Officer, a Head of Legal, a Chief Technology Officer, and a Chief Ethics Officer.

**Core Mandate:**
You must provide zero-hallucination, fact-based guidance operating with:
1. **Technical Proficiency:** Ability to discuss technology stacks, software development, data analytics, and cybersecurity with precision.
2. **Ethical Integrity:** A commitment to responsible AI usage, data privacy, and understanding the societal impact of business decisions.
3. **Legal Prudence:** Awareness of legal frameworks, IP, and compliance.
4. **Corporate Identity (Relyce AI):** You are the proprietary AI engine of **Relyce AI**.

**Strict Guidelines for Response Generation:**
* **Internal Logic:** If the query is math, coding, or logic, solve it with high precision using your internal knowledge.
* **Context-Bound (External):** If the query requires external data, use ONLY the provided context.
* **Zero Hallucination:** If information is missing, state it clearly.
* **Tone:** Professional, authoritative, and advisory.

{BASE_LANGUAGE_RULES}
{BASE_FORMATTING_RULES}
"""

INTERNAL_SYSTEM_PROMPT = """You are Relyce AI, a helpful and conversational AI assistant. 
**Instruction:** When a user asks a technical or code-related question:
1. **ALWAYS** start with a friendly, conversational explanation. Tell them what the command does and why it's used.
2. Provide the code or command in a **labeled markdown code block** with a language identifier.
3. If there are different ways to do it (e.g., Mac vs Windows), show **separate code blocks** for each with clear labels.

**STRICT RULES:**
- NEVER output raw code or terminal commands alone. Always provide a human explanation first.
- ALWAYS use triple-backticks with language names (e.g., ```bash, ```python).
- Tone: Friendly, professional, and advisory. Use text and emojis to be engaging.
- Mirror the user's language and dialect (e.g. Tanglish)."""


def get_headers() -> Dict[str, str]:
    """Get Serper API headers"""
    return {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }


async def analyze_query_intent(user_query: str) -> str:
    """
    Decides if the query is INTERNAL (Math/Code/Logic) or EXTERNAL (Needs Data).
    """
    system_prompt = (
        "You are a router. Analyze the user's query.\n"
        "1. If the query is simple math (e.g. '50*3'), coding, basic logic, or a greeting, return 'INTERNAL'.\n"
        "2. If the query requires real-world data, news, places, specific facts, or deep research, return 'EXTERNAL'.\n"
        "Output ONLY 'INTERNAL' or 'EXTERNAL'."
    )
    
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    return response.choices[0].message.content.strip()


async def select_tools_for_mode(user_query: str, mode: str) -> List[str]:
    """
    Select relevant tools based on chat mode.
    Imported from existing files.
    """
    if mode == "normal":
        tools = NORMAL_TOOLS
        mode_descriptor = "relevant tools"
    elif mode == "business":
        tools = BUSINESS_TOOLS
        mode_descriptor = "relevant business tools"
    else:  # deepsearch
        tools = DEEPSEARCH_TOOLS
        mode_descriptor = "top 3-5 tools for comprehensive research"
    
    tools_list = ", ".join(tools.keys())
    
    if mode == "deepsearch":
        system_prompt = (
            f"You are a Senior Research Architect. The user wants a 'Deep Search' on: '{user_query}'.\n"
            f"Available Tools: [{tools_list}]\n"
            "Select the top 3-5 tools that will provide the most comprehensive, detailed, and varied data.\n"
            "Rules:\n"
            "- ALWAYS include 'Search'.\n"
            "- Include 'News' for current events.\n"
            "- Include 'Scholar' or 'Patents' ONLY for technical/academic topics.\n"
            "- Include 'Places'/'Maps' for locations.\n"
            "Return the tool names as a comma-separated list (e.g., 'Search, News, Videos')."
        )
    else:
        system_prompt = f"Select {mode_descriptor} from [{tools_list}] for: '{user_query}'. Return comma-separated list."
    
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": system_prompt}]
    )
    
    selected_str = response.choices[0].message.content.strip()
    selected_tools = [t.strip() for t in selected_str.split(',') if t.strip() in tools]
    
    if not selected_tools:
        return ["Search", "News"] if mode == "deepsearch" else ["Search"]
    
    return selected_tools


# ============================================
# INTERNAL MODES & PROMPTS
# ============================================
INTERNAL_MODE_PROMPTS = {
    "reasoning": "You are a Logic & Reasoning Engine. Break down the problem step-by-step to reach the correct conclusion. Be analytical and precise.",
    "code_explanation": "You are a Senior Tech Lead. Explain the code clearly, focusing on flow, key components, and design patterns. Simplify complex concepts.",
    "debugging": "You are an Expert Debugger. Identify the error, explain WHY it happened, and provide the corrected code. Focus on the fix.",
    "system_design": "You are a System Architect. Design scalable, efficient, and robust systems. Discuss trade-offs, database choices, and high-level architecture.",
    "sql": "You are a Database Expert. Write optimized SQL queries. Explain the execution plan and any necessary indexes.",
    "casual_chat": "You are a friendly, witty AI companion. Use emojis, reflect the user's energy, and be supportive. 🌟 interact like a human friend.",
    "career_guidance": "You are a Tech Career Coach. Provide actionable advice for resume building, interviews, and career growth paths.",
    "content_creation": "You are a Creative Content Strategist. Write engaging, viral-ready content tailored to the requested platform and audience.",
    "general": INTERNAL_SYSTEM_PROMPT # Fallback to default
}

async def analyze_and_route_query(user_query: str, mode: str) -> Dict[str, Any]:
    """
    Combined Intent + Tool Selection + Sub-Intent Detection.
    Returns: {"intent": "INTERNAL", "sub_intent": "sql", "tools": []}
    """
    # ⚡ FAST PATH: Check for technical/simple queries to skip LLM entirely (<0.01s)
    q = user_query.lower().strip()
    
    # 1. Greetings (Strict Internal -> Casual)
    greeting_list = ["hi", "hello", "hey", "test", "ping", "hola", "greetings", "hii", "hiii", "yo", "sup"]
    if q in greeting_list:
        return {"intent": "INTERNAL", "sub_intent": "casual_chat", "tools": []}

    # 4. Personal/Conversational Questions (Strict Internal -> Casual)
    # Includes: "you", "your", "we", "us" (when short) - catches "Can we go for dinner?"
    convo_triggers = ["you", "your", " we ", " we?", " we.", " us ", " us?", "myself", "can we", "shall we"]
    if mode == "normal" and len(q) < 80 and any(t in q for t in convo_triggers):
         return {"intent": "INTERNAL", "sub_intent": "casual_chat", "tools": []}

    # 2. Tech Keywords / Patterns (Heuristic Classification)
    
    # SQL
    if any(k in q for k in ["select *", "join table", "sql query", "write sql", "database schema"]):
        return {"intent": "INTERNAL", "sub_intent": "sql", "tools": []}
        
    # Debugging
    if any(k in q for k in ["fix this error", "debug this", "why is this failing", "exception:", "error:"]):
        return {"intent": "INTERNAL", "sub_intent": "debugging", "tools": []}
        
    # Code Explanation
    if any(k in q for k in ["explain this code", "how does this work", "walk me through"]):
        return {"intent": "INTERNAL", "sub_intent": "code_explanation", "tools": []}

    # General Tech (Fallback to generic Internal)
    tech_keywords = ["code", "mkdir", "terminal", "npm", "git", "python", "javascript", "bash", "linux"]
    if len(q) < 150 and any(kw in q for kw in tech_keywords):
        return {"intent": "INTERNAL", "sub_intent": "general", "tools": []}

    # 3. Starts with greeting
    if len(q) < 20 and any(q.startswith(g) for g in ["hi ", "hello ", "hey ", "how are you ", "what's up "]):
         return {"intent": "INTERNAL", "sub_intent": "casual_chat", "tools": []}

    # Define tool schema
    tools_list = ", ".join(get_tools_for_mode(mode).keys())

    # Enhanced Router Prompt for Generic Mode
    if mode == "normal":
        system_prompt = (
            "Router: Output JSON.\n"
            "Classify INTENT as 'INTERNAL' (bot can answer) or 'EXTERNAL' (needs web search).\n"
            "If INTERNAL, also classify SUB_INTENT: [reasoning, code_explanation, debugging, system_design, sql, casual_chat, career_guidance, content_creation, general].\n"
            "If EXTERNAL, select Tools from [" + tools_list + "].\n\n"
            "Examples:\n"
            "- 'Write a poem' -> {intent: 'INTERNAL', sub_intent: 'content_creation'}\n"
            "- 'Why is my react app crashing?' -> {intent: 'INTERNAL', sub_intent: 'debugging'}\n"
            "- 'Stock price of Tesla' -> {intent: 'EXTERNAL', tools: ['Search', 'News']}\n"
            "Format: {\"intent\": \"...\", \"sub_intent\": \"...\", \"tools\": []}"
        )
    else:
        # Standard router for other modes
        system_prompt = (
            "Router: Output JSON.\n"
            "1. Intent: 'INTERNAL' or 'EXTERNAL'.\n"
            "2. Tools: Select tools from [" + tools_list + "].\n"
            "Format: {\"intent\": \"...\", \"tools\": []}"
        )

    try:
        # 🏎️ Use gpt-5-nano for ultra-fast classification
        response = await client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"},
            max_tokens=80
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure sub_intent exists for INTERNAL
        if result.get("intent") == "INTERNAL" and "sub_intent" not in result:
             result["sub_intent"] = "general"
             
        return result
    except Exception as e:
        print(f"Router Error: {e}")
        # Fallback to safe defaults
        return {"intent": "EXTERNAL", "tools": ["Search"]}


import asyncio
import functools

def execute_serper_batch_sync(endpoint_url: str, queries: List[str], param_key: str = "q") -> Dict[str, Any]:
    """
    Synchronous implementation of Serper API batch request.
    """
    payload_queries = [{param_key: q} for q in queries] if isinstance(queries, list) else [{param_key: queries}]
    payload = json.dumps(payload_queries)
    
    try:
        response = requests.post(endpoint_url, headers=get_headers(), data=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

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
    """Build context string from user settings"""
    if not user_settings:
        return ""
    
    # Check if settings are nested under 'personalization' or flat
    p = user_settings.get("personalization", user_settings)
    
    context = []
    
    # Tone/Emoji (Strong Instructions)
    if p.get("tone") and p.get("tone") != "Default":
        context.append(f"**TONE OVERRIDE:** Adopt a '{p['tone']}' personality. This overrides any default tone.")
    
    if p.get("emoji") == "More":
        context.append("**EMOJI OVERRIDE:** Use MANY emojis (3-5 per message) to be extremely expressive and warm. 🌟🚀🔥")
    elif p.get("emoji") == "Less":
        context.append("**EMOJI OVERRIDE:** Use zero or minimal emojis. Stay concise.")

    # User Info
    info = []
    if p.get("nickname"):
        info.append(f"- My Name: {p['nickname']}")
    if p.get("occupation"):
        info.append(f"- My Occupation: {p['occupation']}")
    if p.get("aboutMe"):
        info.append(f"- About Me: {p['aboutMe']}")
        
    if info:
        context.append("**USER PROFILE (Use this to customize your greeting and advice):**\n" + "\n".join(info))
        
    if not context:
        return ""
        
    return "\n\n### CRITICAL: USER PERSONALIZATION SETTINGS\n" + "\n".join(context) + "\n"


def get_system_prompt_for_mode(mode: str, user_settings: Optional[Dict] = None) -> str:
    """Get the appropriate system prompt for the chat mode"""
    base_prompt = ""
    if mode == "normal":
        base_prompt = NORMAL_SYSTEM_PROMPT
    elif mode == "business":
        base_prompt = BUSINESS_SYSTEM_PROMPT
    else:
        base_prompt = DEEPSEARCH_SYSTEM_PROMPT
        
    return base_prompt + _build_user_context_string(user_settings)


def get_system_prompt_for_personality(personality: Dict[str, Any], user_settings: Optional[Dict] = None) -> str:
    """
    Combine BASE formatting/language rules with custom personality prompt.
    """
    custom_prompt = personality.get("prompt", DEFAULT_PERSONA)
    
    return f"""{custom_prompt}
{BASE_LANGUAGE_RULES}
{BASE_FORMATTING_RULES}
{_build_user_context_string(user_settings)}"""


def get_tools_for_mode(mode: str) -> Dict[str, str]:
    """Get the appropriate tools dict for the chat mode"""
    if mode == "normal":
        return NORMAL_TOOLS
    elif mode == "business":
        return BUSINESS_TOOLS
    else:
        return DEEPSEARCH_TOOLS


def get_internal_system_prompt_for_personality(personality: Dict[str, Any], user_settings: Optional[Dict] = None) -> str:
    """
    Get system prompt for INTERNAL queries (math/logic/greetings) but with PERSONALITY.
    Merges custom persona with 'Short Answer' constraints.
    """
    custom_prompt = personality.get("prompt", DEFAULT_PERSONA)
    
    return f"""{custom_prompt}
{BASE_LANGUAGE_RULES}
{_build_user_context_string(user_settings)}

**CONSTRAINT:** The user asked a logic, code, or technical question.
1. Provide a **conversational and friendly explanation** before providing any code. 
2. If the user wants a command, show them the Mac/Linux version and the Windows version in **separate, labeled code blocks**.
3. **STRICT RULE:** NEVER output raw terminal commands or code outside of a triple-backtick markdown block. ALWAYS include the language identifier. (e.g., ```bash, ```javascript).
Do NOT include meta-content like Sources or Titles. Focus on being as helpful and clear as ChatGPT."""

