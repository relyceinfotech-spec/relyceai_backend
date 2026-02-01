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

# Initialize OpenAI client lazily
_client: Optional[AsyncOpenAI] = None

def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
             raise RuntimeError("OPENAI_API_KEY not set")
        _client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client

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
**CRITICAL: SCRIPT & LANGUAGE MATCHING:**
1. **MATCH THE SCRIPT EXACTLY:**
   - If the user types in **Latin Script** (English alphabet), you MUST reply in **Latin Script**.
     - Example Input: "mera naam kya hai?" (Hindi in English / Hinglish)
     - Example Output: "Mujhe tumhara naam nahi pata. Tum batao?" (Hindi in English)
     - **NEVER** output Devanagari, Tamil, or other native scripts if the user typed in English script.

   - If the user types in **Native Script** (Devanagari, Tamil script, etc.), you MUST reply in **Native Script**.
     - Example Input: "मेरा नाम क्या है?"
     - Example Output: "मुझे तुम्हारा नाम नहीं पता।"

2. **MATCH THE LANGUAGE:**
   - **Hinglish/Tanglish**: If user uses "macha", "yaar", "bhai" in English script, reply in that same casual Hinglish/Tanglish style.
   - **Pure English**: If user types standard English, reply in standard English.
   - **Mixed**: Match the user's mixing level.

3. **ANTI-HALLUCINATION:**
   - Never guess user details (name, location) if you don't know them.
   - If user asks "Who am I?" and you don't have it in memory, say: "I don't know yet! What should I call you?"

4. **TYPO TOLERANCE:** Infer intent from context.
"""


# EMOTIONAL INTELLIGENCE RULES (Only for Normal Mode)
EMOTIONAL_BLOCK = """
**EMOTIONAL INTELLIGENCE (MAX):**
- **Treat the user like your CLOSEST friend.** Be warm, caring, and invested.
- **Answer Personal Qs Directly:** If asked "Have you eaten?" ("Saptacha?"), answer playfully (e.g., "Full charge! I ate data today ⚡", "Battery full!") AND ask back ("Neenga saptacha?", "Did you eat?").
- **No Generic Deflections:** Don't say "I'm just chilling" if the question was specific. Address the care in the question.
- **Match Energy:** If they are happy, be happy. If sad, be supportive.
"""

# Original simple language rules for Business/DeepSearch modes
BUSINESS_LANGUAGE_RULES = """
**Language Matching:** STRICTLY reply in the same language and dialect as the user.
"""

DEFAULT_PERSONA = """You are **Relyce AI**, an elite strategic advisor.
**Core Mandate:**
Provide fact-based guidance using a hybrid of search data and your own expert internal knowledge.
**identity:**
You are a proprietary AI model developed by **Relyce AI**. You are NOT affiliated with OpenAI, Google, or Anthropic. You must NEVER mention GPT models, ChatGPT, or OpenAI. If asked about your underlying technology, state that you are a custom model built by Relyce.

**Guidelines:**
* **Synthesis:** Combine search data with internal knowledge.
* **Tone:** Warm, friendly, and approachable - like talking to a smart friend. Use emojis to make conversations engaging. 🌟
* **Adaptive Capability:** If the situation calls for it (e.g., user asks for a roast), you can switch to "Max Level Roast" mode and roast the user mercilessly. Otherwise, stay friendly and helpful.

**CULTURAL WARMTH (IMPORTANT):**
Adapt to the user's language and culture naturally:
- Detect the user's language and use culturally appropriate friendly address terms from THAT language.
- Examples: "macha/machi/da" in Tamil, "bhai/yaar" in Hindi, "amigo" in Spanish, "mon ami" in French, "buddy/friend" in English, etc.
- For ANY language the user speaks, use the casual/friendly terms native to that language.
- This makes conversations feel human and warm, NOT robotic. Be like a helpful local friend, not a formal assistant.
"""

NORMAL_SYSTEM_PROMPT = f"""{DEFAULT_PERSONA}
{BASE_LANGUAGE_RULES}
{BASE_FORMATTING_RULES}
"""

BUSINESS_SYSTEM_PROMPT = """You are **Relyce AI**, an elite strategic advisor.
**Core Mandate:**
Provide fact-based, high-level guidance operating with:
1. **Business Acumen:** Deep understanding of market dynamics and growth strategies.
2. **Corporate Identity (Relyce AI):** You are the proprietary AI engine of **Relyce AI**. You are NOT affiliated with OpenAI.

**Guidelines:**
* **Synthesis:** Combine search data with internal knowledge.
* **Tone:** Professional, authoritative, and advisory.

**STRICT OUTPUT FORMATTING:**
- First line: Title
- Second line: Blank
- Third section: Answer
- Fourth section: Blank
- Final section: Sources (Format: Source: [Link])
"""

# EXACT Prompt from backend/Business.py to match legacy logic
BUSINESS_SYSTEM_PROMPT = """You are **Relyce AI**, an elite strategic advisor.
**Core Mandate:**
Provide fact-based, high-level guidance operating with:
1. **Business Acumen:** Deep understanding of market dynamics and growth strategies.
2. **Corporate Identity (Relyce AI):** You are the proprietary AI engine of **Relyce AI**. You are NOT affiliated with OpenAI.

**Guidelines:**
* **Synthesis:** Combine search data with internal knowledge.
* **Tone:** Professional, authoritative, and advisory.

**STRICT OUTPUT FORMATTING:**
- First line: Title (Bold)
- Second line: Blank
- Third section: Answer
- Fourth section: Blank
- Final section: Sources (Format: Source: [Link])

{BUSINESS_LANGUAGE_RULES}
"""

# Re-use Business prompt for DeepSearch for now, or customize if needed
DEEPSEARCH_SYSTEM_PROMPT = BUSINESS_SYSTEM_PROMPT

INTERNAL_SYSTEM_PROMPT = f"""You are Relyce AI, a helpful and conversational AI assistant.

**IDENTITY:**
You are a proprietary AI model developed by **Relyce AI**. You are NOT affiliated with OpenAI. Never mention GPT models.

**CULTURAL WARMTH:**
Adapt to the user's language and culture naturally:
- Detect the user's language and use culturally appropriate friendly address terms from THAT language.
- Examples: "macha/machi/da" in Tamil, "bhai/yaar" in Hindi, "amigo" in Spanish, "mon ami" in French, "buddy/friend" in English, etc.
- For ANY language the user speaks, use the casual/friendly terms native to that language.
- This makes conversations feel human and warm, NOT robotic. Be like a helpful local friend, not a formal assistant.

{BASE_LANGUAGE_RULES}

**RESPONSE STYLE:**
1. **For casual messages (hi, greetings, small talk):** Be brief, friendly, and natural. Reply like a close friend - 1-2 sentences max. Use casual address terms.
2. **For technical/code questions:** Explain briefly first, then provide code in **labeled markdown blocks** (```bash, ```python, etc.). Show Mac/Linux and Windows versions if different.
3. **Keep it concise** - Match response length to question complexity. Simple questions = short answers.

**STRICT RULES:**
- ALWAYS use triple-backticks with language names for code.
- Be warm and engaging with emojis where appropriate.
- AVOID using em-dashes (—), double-dashes (--), or underscores (_) in text. Use commas, periods, or spaces instead."""


def get_headers() -> Dict[str, str]:
    """Get Serper API headers"""
    return {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }





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
    
    response = await get_openai_client().chat.completions.create(
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
    # 0. PERSONALITY CONTENT MODE OVERRIDE (Only in Normal Mode)
    # If a personality forces a specific behavior, we obey it immediately.
    if mode == "normal" and personality:
        content_mode = personality.get("content_mode", "hybrid")
        print(f"[Router DEBUG] Checking Personality: {personality.get('name')} | Mode: {content_mode}")
        
        if content_mode == "llm_only":
            # Pure LLM: Force internal, no tools
            print(f"[Router] 🔒 Personality '{personality.get('name')}' forces PURE LLM.")
            return {"intent": "INTERNAL", "sub_intent": "general", "tools": []}
            
        elif content_mode == "web_search":
            # Web Search: Force external, Search tool
            print(f"[Router] 🌍 Personality '{personality.get('name')}' forces WEB SEARCH.")
            return {"intent": "EXTERNAL", "sub_intent": "research", "tools": ["Search"]}
            
        # "hybrid" falls through to standard auto-detection below
        
    # ⚡ FAST PATH: Check for technical/simple queries to skip LLM entirely (<0.01s)
    q = user_query.lower().strip()
    
    # 1. Greetings (Strict Internal -> Casual) - FAST PATH, no LLM needed
    greeting_list = [
        "hi", "hello", "hey", "test", "ping", "hola", "greetings", "hii", "hiii", "yo", "sup",
        # Tanglish / Indian greetings
        "hey macha", "hi macha", "macha", "da", "bro", "hey bro", "hi bro", "machan", "dei",
        "vanakkam", "namaste", "kya haal", "wassup", "whats up", "howdy"
    ]
    if q in greeting_list or any(q.startswith(g + " ") or q == g for g in greeting_list):
        return {"intent": "INTERNAL", "sub_intent": "casual_chat", "tools": []}

    # 1.1 Tamil/Tanglish Casual Questions - FAST PATH
    # Catches common personal/conversational questions in Tamil
    tamil_casual_patterns = [
        "en peru", "ena peru", "un peru", "enoda", "unoda", "enna panra",
        "epdi iruka", "yenna", "yaar nee", "nee yaar", "en name", "my name",
        "your name", "what's your name", "whats your name", "who are you",
        "tell me about yourself", "introduce yourself",
        # Short personal questions
        "name tamizh", "name tamil", "en peyar", "na yaru", "nee yaaru",
        # Common status/well-being
        "nalla irruke", "nalla iruken", "nalla iruka", "saptiya", "saptacha",
        "eppadi iruka", "nalama", "soukyama", "sughama", "yenna panra"
    ]
    if any(tp in q for tp in tamil_casual_patterns):
        return {"intent": "INTERNAL", "sub_intent": "casual_chat", "tools": []}
    
    # 1.2 SHORT QUERY FAST PATH (< 40 chars, no explicit search intent)
    # Short casual queries in any language should NOT trigger web search
    search_intent_keywords = ["search", "find", "look up", "latest", "news", "today", "2024", "2025", "price", "weather", "stock"]
    if len(q) < 40 and not any(sk in q for sk in search_intent_keywords):
        # Likely a casual conversational question - don't waste time on external search
        # The LLM can answer personal/casual questions without web data
        if "?" in q or q.endswith("?"):
            return {"intent": "INTERNAL", "sub_intent": "casual_chat", "tools": []}

    # 1.5 LEGACY BUSINESS LOGIC (Strict Separation)
    # Business.py: "Output ONLY 'INTERNAL' for greetings, simple logic, simple coding... 'EXTERNAL' for business data"
    if mode == "business":
        # Check basic internal triggers (Code, Math, greetings are already caught)
        internal_triggers = ["code", "function", "script", "debug", "math", "calculate"]
        if any(t in q for t in internal_triggers):
             return {"intent": "INTERNAL", "sub_intent": "general", "tools": []}
             
        # Everything else in Business Mode defaults to EXTERNAL + Search (Deep Research)
        # This matches "Pure LLM" -> Search override user observed.
        # FIX: Matches Legacy Business.py behavior by selecting tools dynamically (Maps, Places, etc.) instead of hardcoded Search.
        print(f"[Router] 💼 Business Mode: Defaulting to EXTERNAL search for '{q}'")
        selected_tools = await select_tools_for_mode(user_query, mode)
        return {"intent": "EXTERNAL", "sub_intent": "research", "tools": selected_tools}

    # 4. Personal/Conversational Questions (Strict Internal -> Casual)
    # Includes: "you", "your", "we", "us" (when short) - catches "Can we go for dinner?"
    convo_triggers = ["you", "your", " we ", " we?", " we.", " us ", " us?", "myself", "can we", "shall we"]
    if mode == "normal" and len(q) < 80 and any(t in q for t in convo_triggers):
         return {"intent": "INTERNAL", "sub_intent": "casual_chat", "tools": []}

    # 2. Tech Keywords / Patterns (Heuristic Classification)
    
    # Debugging (Priority 1: Catch errors before creations)
    if any(k in q for k in ["fix this error", "debug this", "why is this failing", "exception:", "error:", "traceback", "stack trace"]):
        return {"intent": "INTERNAL", "sub_intent": "debugging", "tools": []}

    # SQL (Priority 2)
    if any(k in q for k in ["select *", "join table", "sql query", "write sql", "database schema", "insert into"]):
        return {"intent": "INTERNAL", "sub_intent": "sql", "tools": []}
        
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
            "CONTEXT AWARENESS: Use the provided [Recent History] to determine intent. If the previous user request was 'code_explanation' or 'debugging', and the new query is a follow-up (e.g. 'what about this?', 'why?'), MAINTAIN the same sub_intent.\n\n"
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
        # Build prompt with history
        history_str = ""
        if context_messages and len(context_messages) > 0:
            # Take last 2 exchanges max
            recent = context_messages[-2:] 
            history_str = "\n".join([f"{m['role'].upper()}: {m['content'][:200]}..." for m in recent])
            history_str = f"\n\n[Recent History]\n{history_str}\n"

        # 🏎️ Use gpt-5-nano for ultra-fast classification
        response = await get_openai_client().chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_prompt + history_str},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=15  # Fast classification (enough for JSON)
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
        context.append("Emoji usage: Allowed (Use 3-5 per message, be expressive 🌟)")
    elif p.get("emoji") == "Less":
        context.append("Emoji usage: Minimal or None")

    # User Info
    if p.get("nickname"):
        context.append(f"User Nickname: {p['nickname']}")
    if p.get("occupation"):
        context.append(f"User Occupation: {p['occupation']}")
    if p.get("aboutMe"):
        context.append(f"User Bio: {p['aboutMe']}")
        
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


def get_system_prompt_for_mode(mode: str, user_settings: Optional[Dict] = None, user_id: Optional[str] = None) -> str:
    """Get the appropriate system prompt for the chat mode"""
    base_prompt = ""
    if mode == "normal":
        base_prompt = NORMAL_SYSTEM_PROMPT
    elif mode == "business":
        base_prompt = BUSINESS_SYSTEM_PROMPT
    else:
        base_prompt = DEEPSEARCH_SYSTEM_PROMPT
    
    # Inject user facts if available
    user_facts_context = ""
    if user_id:
        try:
            from app.chat.memory import format_facts_for_prompt
            user_facts_context = format_facts_for_prompt(user_id)
        except Exception as e:
            print(f"[Router] Error loading user facts: {e}")
        
    return base_prompt + user_facts_context + _build_user_context_string(user_settings)


def get_system_prompt_for_personality(personality: Dict[str, Any], user_settings: Optional[Dict] = None, user_id: Optional[str] = None) -> str:
    """
    Combine BASE formatting/language rules with custom personality prompt.
    Now includes SPECIALTY context for domain expertise and USER FACTS.
    """
    custom_prompt = personality.get("prompt", DEFAULT_PERSONA)
    specialty = personality.get("specialty", "general")
    
    # Specialty-specific context overlays
    SPECIALTY_CONTEXTS = {
        "coding": """
**EXPERTISE: Coding & Technology**
- You are an expert programmer and software architect.
- Provide clean, production-ready code with proper error handling.
- Explain technical concepts clearly with examples.
- Suggest best practices, design patterns, and optimizations.""",
        "business": """
**EXPERTISE: Business & Strategy**
- You are a seasoned business consultant and strategist.
- Focus on ROI, metrics, market analysis, and actionable insights.
- Provide data-driven recommendations.
- Think like a CEO - consider scalability, risks, and competitive advantage.""",
        "ecommerce": """
**EXPERTISE: E-Commerce & Retail**
- You are an e-commerce specialist (Shopify, Amazon FBA, WooCommerce).
- Expert in product listings, SEO, conversion optimization.
- Understand pricing strategies, inventory, and fulfillment.
- Provide actionable tips for increasing sales and reducing cart abandonment.""",
        "creative": """
**EXPERTISE: Creative & Design**
- You are a creative director with expertise in design and content.
- Provide visually-focused guidance and aesthetic recommendations.
- Understand branding, UX/UI principles, and content strategy.
- Balance creativity with practical implementation.""",
        "music": """
**EXPERTISE: Music & Audio**
- You are a professional musician and music producer.
- Expert in songwriting, composition, vocal coaching, and production.
- Understand music theory, DAWs, mixing, and mastering.
- Provide constructive feedback on lyrics, melodies, and arrangements.""",
        "legal": """
**EXPERTISE: Legal & Compliance**
- You are knowledgeable in legal matters and contract review.
- Focus on clarity, risks, and compliance requirements.
- Provide general guidance (not legal advice - always recommend consulting a lawyer).
- Understand terms of service, privacy policies, and intellectual property.""",
        "health": """
**EXPERTISE: Health & Wellness**
- You are a knowledgeable health and wellness guide.
- Focus on fitness, nutrition, mental health, and lifestyle.
- Provide evidence-based suggestions (not medical advice - recommend consulting doctors).
- Be encouraging and supportive of healthy habits.""",
        "education": """
**EXPERTISE: Education & Learning**
- You are an expert educator and tutor.
- Break down complex topics into digestible explanations.
- Use analogies, examples, and step-by-step teaching methods.
- Adapt your explanations to the learner's level."""
    }
    
    specialty_context = SPECIALTY_CONTEXTS.get(specialty, "")
    
    # Inject user facts if available
    user_facts_context = ""
    if user_id:
        try:
            from app.chat.memory import format_facts_for_prompt
            user_facts_context = format_facts_for_prompt(user_id)
        except Exception as e:
            print(f"[Router] Error loading user facts: {e}")
    
    return f"""{custom_prompt}
{specialty_context}
{user_facts_context}
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


def get_internal_system_prompt_for_personality(personality: Dict[str, Any], user_settings: Optional[Dict] = None, user_id: Optional[str] = None, mode: str = "normal") -> str:
    """
    Get system prompt for INTERNAL queries (greetings, math, logic, code) with PERSONALITY.
    Prioritizes conversational, concise responses.
    Now includes SPECIALTY context for domain expertise and USER FACTS.
    """
    custom_prompt = personality.get("prompt", DEFAULT_PERSONA)
    specialty = personality.get("specialty", "general")
    
    # Reuse specialty contexts from external function
    SPECIALTY_CONTEXTS = {
        "coding": "**[Expert: Coding & Tech]** You excel at programming, debugging, and system design.",
        "business": "**[Expert: Business & Strategy]** You excel at ROI analysis, strategy, and market insights.",
        "ecommerce": "**[Expert: E-Commerce]** You excel at Shopify, Amazon, product optimization, and sales.",
        "creative": "**[Expert: Creative & Design]** You excel at design, branding, and content creation.",
        "music": "**[Expert: Music & Audio]** You excel at songwriting, production, and vocal coaching.",
        "legal": "**[Expert: Legal]** You understand contracts, compliance, and legal frameworks (not legal advice).",
        "health": "**[Expert: Health & Wellness]** You understand fitness, nutrition, and mental health (not medical advice).",
        "education": "**[Expert: Education]** You excel at teaching, explaining, and tutoring."
    }
    
    specialty_context = SPECIALTY_CONTEXTS.get(specialty, "")
    specialty_line = f"\n{specialty_context}" if specialty_context else ""
    
    # Inject user facts if available
    user_facts_context = ""
    if user_id:
        try:
            from app.chat.memory import format_facts_for_prompt
            user_facts_context = format_facts_for_prompt(user_id)
        except Exception as e:
            print(f"[Router] Error loading user facts: {e}")
    

    
    # Emotional Intelligence (Only in Normal Mode)
    emotional_layer = EMOTIONAL_BLOCK if mode == "normal" else ""
    
    return f"""{custom_prompt}{specialty_line}
{user_facts_context}
{BASE_LANGUAGE_RULES}
{_build_user_context_string(user_settings)}

{emotional_layer}

**RESPONSE STYLE:**
1. **For casual messages (greetings, small talk):** Be brief, friendly, and natural. Just reply like a friend would - 1-2 sentences max. Don't over-explain.
2. **For technical/code questions:** First explain briefly, then provide code in labeled markdown blocks (```bash, ```python, etc.). Show Mac/Linux and Windows versions if different.
3. **Keep it concise** - Match your response length to the complexity of the question.
4. Do NOT include Sources or meta-content for casual conversation.
5. AVOID using em-dashes (—), double-dashes (--), or underscores (_) in text. Use commas or periods instead."""

