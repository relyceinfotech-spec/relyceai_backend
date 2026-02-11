"""
Relyce AI - LLM Processor
Handles message processing with streaming support
Includes legacy prompts and routing logic consolidated into app/llm
"""
import json
import re
from typing import AsyncGenerator, List, Dict, Any, Optional
from openai import OpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL, GEMINI_MODEL, CODING_MODEL, ERNIE_THINKING_MODEL
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
)
from app.llm.routing_log import log_routing_decision

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

CREATIVE_INTENTS = {
    "content_creation",
    "ui_design",
    "ui_strategy",
}

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
    
    Logic: needsReasoning = (containsReasoningKeywords OR multiSentence)
    Length is just an optimization hint, NOT a hard skip rule.
    """
    query_lower = query.lower()
    
    # Simple intents NEVER need thinking (hard rule)
    simple_intents = ["casual_chat", "general", "code_explanation", "sql", "content_creation", "ui_design", "ui_demo_html", "ui_react"]
    if sub_intent in simple_intents:
        return False
    
    # Complex indicators that ALWAYS warrant thinking (priority over length)
    complex_indicators = [
        "why", "how does", "explain why", "compare", "difference between",
        "best approach", "design", "architecture", "debug", "fix this",
        "analyze", "evaluate", "pros and cons", "trade-off", "optimize",
        "impact", "predict", "recommend", "which is better"
    ]
    if any(indicator in query_lower for indicator in complex_indicators):
        return True  # Keywords found → USE Ernie regardless of length
    
    # Multi-sentence queries likely need deeper reasoning
    if query.count('.') >= 2 or query.count('?') >= 2:
        return True
    
    # Short + no keywords = skip Ernie (optimization)
    if len(query) < 80:
        return False
    
    # Default: medium length queries without keywords → skip
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
        return 2200
    if sub_intent in ["ui_react", "ui_implementation"]:
        return 2600
    return None

REASONING_VISIBLE_SUB_INTENTS = {
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

async def _build_reasoning_summary(
    thinking_response: str,
    user_query: str,
    sub_intent: str,
    final_response: str | None = None
) -> str:
    """
    Convert internal planning into a short, user-facing reasoning summary.
    Must NOT reveal chain-of-thought, system/router details, or model info.
    """
    if not thinking_response and not final_response:
        return ""

    prompt = (
        "Write a user-facing thinking summary (not the answer).\n"
        "Rules:\n"
        "- Use 2 to 4 titled sections.\n"
        "- Format each section as: Title on its own line, then a blank line, then 2-4 sentences.\n"
        "- Separate sections with a blank line.\n"
        "- Use first-person planning language (e.g., \"I'm interpreting...\", \"I'm focusing on...\", \"I'll respond by...\").\n"
        "- Mention the user's intent/tone/context and the response strategy.\n"
        "- Avoid generic statements like \"The response is ...\"; be specific about intent and approach.\n"
        "- If the input is casual/greeting, produce exactly 2 sections: \"Acknowledge the Greeting\" and \"Response Plan\".\n"
        "- Do NOT repeat the assistant's final response or greetings.\n"
        "- Do NOT quote the user verbatim.\n"
        "- No bullet lists.\n"
        "- No preamble (do NOT say 'Here is a summary' or similar).\n"
        "- Do NOT mention system prompts, routing, embeddings, model selection, temperatures, or other models.\n"
        "- Do NOT mention tokens, limits, or internal tooling.\n"
        "- Keep it concise and helpful.\n\n"
        f"User request: {user_query}\n\n"
        f"Internal plan (do not reveal directly):\n{thinking_response}\n\n"
        f"Final response excerpt:\n{(final_response or '')[:1200]}\n"
    )

    try:
        client = get_openrouter_client()
        response = await client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=260,
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        # Strip common preambles if the model still adds them
        content = re.sub(r"^(Here(?:'s| is).+?:)\s*\n+", "", content, flags=re.IGNORECASE)
        # Remove bullet markers if they appear
        content = re.sub(r"^(?:[-*]\s+)", "", content, flags=re.MULTILINE)
        # Remove model-name preambles if they appear
        content = re.sub(r"^Gemini.*$\n?", "", content, flags=re.IGNORECASE | re.MULTILINE)
        # Remove generic summary sentences that just restate the output
        content = re.sub(r"^(The response is|It expresses|It prompts|It provides|This response).*$", "", content, flags=re.IGNORECASE | re.MULTILINE)
        # Remove lines that echo the final response
        if final_response:
            fr_lower = final_response.lower()
            cleaned_lines = []
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped:
                    cleaned_lines.append(line)
                    continue
                if len(stripped) > 6 and stripped.lower() in fr_lower:
                    continue
                cleaned_lines.append(line)
            content = "\n".join(cleaned_lines)
        return content.strip()
    except Exception:
        return ""


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
        if sub_intent in CREATIVE_INTENTS:
            return ("openrouter", CODING_MODEL, None, False, "coding_buddy_override")
        if sub_intent in ["debugging", "system_design", "coding_complex"] and needs_thinking:
            return ("openrouter", CODING_MODEL, None, True, "coding_buddy_override_thinking")
        temp = _resolve_temperature(personality, "coding")
        return ("openrouter", CODING_MODEL, temp, False, "coding_buddy_override")

    # System-design override (explicit priority)
    if sub_intent == "system_design":
        return ("openrouter", CODING_MODEL, None, False, "system_design_override")

    # Coding intents: use coding model, allow thinking for complex debugging/system design
    if sub_intent in LOGIC_CODING_INTENTS:
        if sub_intent in ["debugging", "system_design"] and needs_thinking:
            return ("openrouter", CODING_MODEL, None, True, "coding_override_thinking")
        return ("openrouter", CODING_MODEL, None, False, "coding_override")

    # Business mode always uses OpenAI (except coding intents above)
    if mode == "business":
        return ("openai", LLM_MODEL, 0.4, False, "mode_override")

    # UI/creative requests should stay creative (no coding clamp)
    if sub_intent in CREATIVE_INTENTS:
        return ("openrouter", GEMINI_MODEL, None, False, "creative_override")
    
    # Analysis/Research with thinking gate
    if sub_intent in ["analysis", "research", "reasoning"]:
        if needs_thinking:
            return ("openrouter", ERNIE_THINKING_MODEL, 0.2, True, "reasoning_override")
        # Skip thinking, go direct to Gemini
        return ("openrouter", GEMINI_MODEL, 0.4, False, "reasoning_override")
    
    # Default: General/Personality uses Gemini Flash Lite
    fallback_specialty = personality.get("specialty", "general") if personality else "general"
    # Coding-specialty personalities use GLM4.7 Flash directly
    if fallback_specialty == "coding":
        return ("openrouter", CODING_MODEL, None, False, "persona_specialty_override")
    temp = _resolve_temperature(personality, fallback_specialty)
    return ("openrouter", GEMINI_MODEL, temp, False, "default_persona_model")

class LLMProcessor:
    """
    Main LLM processor class that handles all chat modes.
    Consolidated logic from legacy mode scripts
    """
    
    def __init__(self):
        self.model = LLM_MODEL

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

        # Hard caps for safety-critical domains
        if specialty == "legal":
            temp = min(temp, 0.3)
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
    
    def _sanitize_output_text(self, text: str) -> str:
        """
        Normalize punctuation to avoid em-dashes/double-hyphen in outputs.
        """
        if not text:
            return text
        # Replace em dash and en dash with a simple hyphen
        sanitized = text.replace("—", " - ").replace("–", " - ")
        # Reduce double-hyphen to single hyphen spacing
        sanitized = sanitized.replace("--", " - ")
        return sanitized
    
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
            "Update the summary to include the new messages. \n"
            "Summarize the conversation into 5–7 bullet points.\n"
            "Preserve:\n"
            "- Technical decisions\n"
            "- User intent\n"
            "- Constraints (privacy, admin rules)\n"
            "- Important entities (Firestore, Redis, etc.)\n"
            "Do NOT add new info. Keep it concise."
        )
        
        try:
            response = await get_openai_client().chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception:
            return existing_summary # Return old summary if update fails

    
    async def process_internal_query(
        self,
        user_query: str,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        mode: str = "normal",
        sub_intent: str = "general",
        user_id: Optional[str] = None
    ) -> str:
        """
        Handle internal queries (greetings, math, code, logic).
        Now supports PERSONALITY customization.
        """
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
            
        # Check for Coding Buddy override
        model_to_use = self.model
        if sub_intent in LOGIC_CODING_INTENTS:
            model_to_use = CODING_MODEL
        if personality and personality.get("id") == "coding_buddy":
            model_to_use = CODING_MODEL
            print(f"[LLM] ⚡ Switching to {model_to_use} for Coding Buddy")

        try:
            reason = "coding_override" if (sub_intent in LOGIC_CODING_INTENTS or (personality and personality.get("id") == "coding_buddy")) else "default_persona_model"
            log_routing_decision(user_id, {
                "intent": "INTERNAL",
                "sub_intent": sub_intent,
                "mode": mode,
                "forced_model": model_to_use,
                "reason": reason
            })
        except Exception as e:
            print(f"[RoutingLog] Failed: {e}")

        client = get_openrouter_client() if model_to_use == CODING_MODEL else get_openai_client()
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
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = await client.chat.completions.create(**create_kwargs)
        return self._sanitize_output_text(response.choices[0].message.content)
    
    async def process_external_query(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        sub_intent: str = "general",
        user_id: Optional[str] = None
    ) -> tuple[str, List[str]]:
        """
        Handle external queries that need web search.
        
        Returns: (response, tools_activated)
        """
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
            # 🎭 Personalities are ONLY honored in Normal Mode
            from app.llm.router import get_system_prompt_for_personality
            system_prompt = get_system_prompt_for_personality(personality, user_settings)
        else:
            # Business / Deep Search / Normal (without persona) -> Use default System Prompt defined in router.py
            # This ensures Business Mode uses the EXACT "Elite Strategic Advisor" prompt from Business.py
            system_prompt = get_system_prompt_for_mode(mode, user_settings)

        # Apply specialized internal prompt overlays for coding/technical intents
        from app.llm.router import INTERNAL_MODE_PROMPTS
        if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
            system_prompt = f"{system_prompt}\n\n**MODE SWITCH: {sub_intent.upper()}**\n{INTERNAL_MODE_PROMPTS[sub_intent]}"

        
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

        client = get_openai_client() if client_type == "openai" else get_openrouter_client()
        create_kwargs = {
            "model": model_to_use,
            "messages": messages
        }
        create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = await client.chat.completions.create(**create_kwargs)
        
        return self._sanitize_output_text(response.choices[0].message.content), selected_tools
    
    async def process_message(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - routes to appropriate handler.
        Returns complete response dict.
        """
        # Analyze intent using the consolidated router
        analysis = await analyze_and_route_query(user_query, mode, personality=personality)
        intent = analysis.get("intent", "EXTERNAL")
        
        if intent == "INTERNAL":
            response_text = await self.process_internal_query(
                user_query,
                personality,
                user_settings,
                mode=mode,
                sub_intent=analysis.get("sub_intent", "general"),
                user_id=user_id
            )
            return {
                "success": True,
                "response": self._sanitize_output_text(response_text),
                "mode_used": "internal",
                "tools_activated": []
            }
        else:
            response_text, tools = await self.process_external_query(
                user_query, mode, context_messages, personality, user_settings, analysis.get("sub_intent", "general"), user_id
            )
            return {
                "success": True,
                "response": self._sanitize_output_text(response_text),
                "mode_used": mode,
                "tools_activated": tools
            }
    
    async def process_message_stream(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version of process_message.
        Yields tokens as they're generated.
        """
        import time
        start_time = time.time()
        print(f"[LATENCY] Processing Stream Request... (Time: 0.0000s)")

        # Combined Analysis (Intent + Tools)
        t_analysis_start = time.time()
        # Pass context_messages for sticky routing (history awareness)
        # Pass personality for Content Mode overrides (Pure LLM / Web Search)
        analysis = await analyze_and_route_query(user_query, mode, context_messages, personality=personality)
        intent = analysis.get("intent", "EXTERNAL")
        selected_tools = analysis.get("tools", [])
        sub_intent = analysis.get("sub_intent", "general")
        show_reasoning_panel = _should_show_reasoning_panel(mode, sub_intent, user_settings)
        
        print(f"[LATENCY] Analysis/Router Complete ({intent}): {time.time() - start_time:.4f}s (Analysis took: {time.time() - t_analysis_start:.4f}s)")
        
        if intent == "INTERNAL" or not selected_tools:
            # Stream internal response
            if personality:
                base_prompt = get_internal_system_prompt_for_personality(personality, user_settings, user_id, mode=mode)
            else:
                from app.llm.router import _build_user_context_string
                base_prompt = INTERNAL_SYSTEM_PROMPT + _build_user_context_string(user_settings)

            from app.llm.router import INTERNAL_MODE_PROMPTS
            if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
                specialized_prompt = INTERNAL_MODE_PROMPTS[sub_intent]
                system_prompt = f"{base_prompt}\n\n**MODE SWITCH: {sub_intent.upper()}**\n{specialized_prompt}"
            else:
                system_prompt = base_prompt
            print(f"[LATENCY] Starting Internal Stream ({analysis.get('sub_intent', 'general')}): {time.time() - start_time:.4f}s")

            # Construct messages with Context (Option 2 Support)
            messages = [{"role": "system", "content": system_prompt}]
            
            # Inject context messages (including potential Summary system message)
            if context_messages:
                # If we have a summary system message injected by websocket.py, it will be first
                # We simply append all context messages. 
                # Note: Logic to prune/summarize is handled in websocket.py/main.py before calling this.
                messages.extend(context_messages)
            
            messages.append({"role": "user", "content": user_query})

            print(f"[LATENCY] Internal: Preparing stream... (Time: {time.time() - start_time:.4f}s)")
            t_stream_start = time.time()
            
            # 🔀 Multi-Model Routing with Trace
            # Initialize trace for this request
            trace = RequestTrace()
            trace.intent = intent
            trace.sub_intent = sub_intent
            trace.log("ROUTE_START", f"mode={mode}, sub_intent={sub_intent}")
            
            # Get routing decision with confidence gating
            client_type, model_to_use, temperature, is_thinking_pass, route_reason = get_model_for_intent(
                mode, sub_intent, personality, user_query
            )

            # Gemini does not emit reasoning tokens; suppress thinking panel for Gemini outputs
            if model_to_use == GEMINI_MODEL:
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
                client = get_openai_client()
            else:
                client = get_openrouter_client()
            
            create_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "stream": True
            }
            max_tokens = _get_max_tokens_for_sub_intent(sub_intent)
            if max_tokens:
                create_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            
            # For thinking models, we need to collect the full response for 2-pass
            if is_thinking_pass:
                # Pass 1: Thinking model plans and analyzes (NO code generation)
                trace.log("THINKING_START", f"model={model_to_use}")
                thinking_response = ""
                
                # For coding intents, use a planning-focused prompt
                if sub_intent in LOGIC_CODING_INTENTS:
                    planning_messages = [
                        {"role": "system", "content": (
                            "You are an expert software architect, debugger, and technical analyst. "
                            "Your job is to THINK, PLAN, and ANALYZE — NOT to write code.\n\n"
                            "Depending on the request:\n"
                            "• If BUILDING something: Break down requirements, identify tech stack, "
                            "structure, layout, design decisions, colors, fonts, interactions, and best practices.\n"
                            "• If DEBUGGING: Analyze the error/issue, identify root causes, "
                            "explain what's going wrong, and outline the exact fix strategy step by step.\n"
                            "• If EXPLAINING code: Break down the code's purpose, logic flow, "
                            "key functions, data flow, and how each part connects.\n"
                            "• If DESIGNING a system: Identify components, architecture patterns, "
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
                    stream = await client.chat.completions.create(**planning_kwargs)
                else:
                    stream = await client.chat.completions.create(**create_kwargs)
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        thinking_response += content_chunk
                trace.log("THINKING_COMPLETE", f"chars={len(thinking_response)}")

                # Pass 2: Generate final output with appropriate model
                if sub_intent in LOGIC_CODING_INTENTS:
                    output_model = CODING_MODEL
                else:
                    output_model = GEMINI_MODEL

                reasoning_summary = ""
                if show_reasoning_panel and output_model != GEMINI_MODEL:
                    reasoning_summary = await _build_reasoning_summary(thinking_response, user_query, sub_intent)
                
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
                
                pass2_kwargs = {
                    "model": output_model,
                    "messages": pass2_messages,
                    "stream": True,
                    "temperature": 0.3  # Lower temp for synthesis
                }
                if max_tokens:
                    pass2_kwargs["max_tokens"] = max_tokens
                
                stream = await get_openrouter_client().chat.completions.create(**pass2_kwargs)
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
                if reasoning_summary:
                    yield "[THINKING]"
                    yield self._sanitize_output_text(reasoning_summary)
                    yield "[/THINKING]"
            else:
                # Single-pass: Direct streaming (captures reasoning tokens from models like GLM)
                stream = await client.chat.completions.create(**create_kwargs)
                print(f"[LATENCY] Internal: Stream Connection Established: {time.time() - start_time:.4f}s (Waited: {time.time() - t_stream_start:.4f}s)")
                
                emit_raw_reasoning = show_reasoning_panel
                collect_for_summary = show_reasoning_panel and not emit_raw_reasoning
                response_chunks: List[str] = [] if collect_for_summary else []
                reasoning_chunks: List[str] = [] if collect_for_summary else []
                in_reasoning = False
                allow_reasoning_tokens = show_reasoning_panel
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    
                    # Check for GLM/model built-in reasoning tokens
                    if allow_reasoning_tokens:
                        reasoning = getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None)
                        if reasoning:
                            if emit_raw_reasoning:
                                if not in_reasoning:
                                    in_reasoning = True
                                    trace.log("REASONING_START", f"model={model_to_use} (built-in)")
                                    yield "[THINKING]"
                                yield self._sanitize_output_text(reasoning)
                            else:
                                reasoning_chunks.append(reasoning)
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
                        sanitized_chunk = self._sanitize_output_text(delta.content)
                        if collect_for_summary:
                            response_chunks.append(sanitized_chunk)
                        yield sanitized_chunk
                
                # Close reasoning if stream ended during reasoning
                if in_reasoning:
                    trace.log("REASONING_COMPLETE", "built-in reasoning done")
                    if emit_raw_reasoning:
                        yield "[/THINKING]"

                if collect_for_summary and response_chunks:
                    final_response = "".join(response_chunks).strip()
                    if final_response:
                        reasoning_text = "".join(reasoning_chunks).strip()
                        if reasoning_text:
                            reasoning_summary = await _build_reasoning_summary(
                                reasoning_text, user_query, sub_intent, final_response
                            )
                        else:
                            reasoning_summary = await _build_reasoning_summary(
                                "", user_query, sub_intent, final_response
                            )
                        if reasoning_summary:
                            yield "[THINKING]"
                            yield self._sanitize_output_text(reasoning_summary)
                            yield "[/THINKING]"
        else:
            # External Mode with Tools
            tools_dict = get_tools_for_mode(mode)
            
            # Yield info about tools being used
            yield f"[INFO] Searching with: {', '.join(selected_tools)}\n\n"
            
            t2 = time.time()
            aggregated_context = {}
            for tool in selected_tools:
                if tool in tools_dict:
                    endpoint = tools_dict[tool]
                    param_key = "url" if tool == "Webpage" else "q"
                    result = await execute_serper_batch(endpoint, [user_query], param_key=param_key)
                    aggregated_context[tool] = result
            print(f"[LATENCY] Tool Execution (Serper): {time.time() - t2:.4f}s")
            
            # 🔧 PRODUCTION HARDENING: Mini validation pass for search results
            def validate_search_results(context: dict) -> dict:
                """
                Lightweight validation/deduplication of search results.
                - Remove duplicate URLs
                - Strip ad content
                - Rank by freshness (if date available)
                """
                validated = {}
                seen_urls = set()
                
                for tool, results in context.items():
                    if isinstance(results, list):
                        cleaned = []
                        for item in results:
                            if isinstance(item, dict):
                                url = item.get("link", item.get("url", ""))
                                # Skip duplicates
                                if url and url in seen_urls:
                                    continue
                                seen_urls.add(url)
                                
                                # Skip obvious ad/sponsored content
                                title = item.get("title", "").lower()
                                if any(ad in title for ad in ["sponsored", "ad:", "[ad]"]):
                                    continue
                                    
                                cleaned.append(item)
                        validated[tool] = cleaned
                    else:
                        validated[tool] = results
                        
                return validated
            
            aggregated_context = validate_search_results(aggregated_context)
            print(f"[SEARCH] Validated {sum(len(v) if isinstance(v, list) else 1 for v in aggregated_context.values())} results")
            
            # Now stream the synthesis
            context_str = json.dumps(aggregated_context, indent=2)
            
            # Determine system prompt
            if personality and mode == "normal":
                # 🎭 Personalities are now honored ONLY in Normal/Generic mode
                # This prevents "Hello macha" leaking into Business Mode
                from app.llm.router import get_system_prompt_for_personality
                system_prompt = get_system_prompt_for_personality(personality, user_settings, user_id)
            else:
                from app.llm.router import get_system_prompt_for_mode
                system_prompt = get_system_prompt_for_mode(mode, user_settings, user_id)
            
            messages = [{"role": "system", "content": system_prompt}]
            
            if context_messages:
                messages.extend(context_messages[-6:])
            
            messages.append({
                "role": "user",
                "content": f"Search Data:\n{context_str}\n\nUser Query: {user_query}"
            })
            
            print(f"[LATENCY] Ready to Stream Synthesis: {time.time() - start_time:.4f}s")
            
            # 🔀 HYBRID MODE: Web Search + Conditional Reasoning
            trace_ext = RequestTrace()
            trace_ext.intent = "EXTERNAL"
            trace_ext.sub_intent = "web_search"
            
            # Check if query needs deep reasoning after web search
            needs_reasoning = should_use_thinking_model(user_query, "web_search")

            final_client_type, final_model, final_temp, _, route_reason = get_model_for_intent(
                mode, sub_intent, personality, user_query
            )
            final_client = get_openai_client() if final_client_type == "openai" else get_openrouter_client()

            # Preserve legacy behavior: Business mode never uses the reasoning pass
            if final_client_type == "openai":
                needs_reasoning = False

            trace_ext.log(
                "ROUTE_DECISION",
                f"client={final_client_type}, model={final_model}, temp={final_temp}, reason={route_reason}"
            )
            try:
                log_routing_decision(user_id, {
                    "intent": "EXTERNAL",
                    "sub_intent": sub_intent,
                    "mode": mode,
                    "forced_model": final_model,
                    "reason": route_reason
                })
            except Exception as e:
                print(f"[RoutingLog] Failed: {e}")
            
            if final_client_type == "openai":
                # OpenAI direct (no reasoning step)
                client = final_client
                model_to_use = final_model
                trace_ext.model_chain.append(model_to_use)
                trace_ext.log("EXTERNAL_ROUTE", f"direct({route_reason}) -> {model_to_use}")
                
                create_kwargs = {
                    "model": model_to_use,
                    "messages": messages,
                    "stream": True
                }
                create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
                if final_temp is not None:
                    create_kwargs["temperature"] = final_temp
                stream = await client.chat.completions.create(**create_kwargs)
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
                        
            elif needs_reasoning:
                # 🧠 HYBRID: Web → Ernie (thinking) → Gemini (synthesis)
                trace_ext.is_thinking_pass = True
                trace_ext.model_chain.append(ERNIE_THINKING_MODEL)
                trace_ext.log("HYBRID_START", f"hybrid(reasoning, final={route_reason}) -> Ernie")
                
                # Pass 1: Ernie thinks about the search results
                thinking_messages = [
                    {"role": "system", "content": "You are an expert analyst. Analyze the search results thoroughly. Identify key facts, compare sources, resolve contradictions, and extract insights. Think step by step."},
                    {"role": "user", "content": f"Search Results:\n{context_str}\n\nUser Query: {user_query}\n\nAnalyze these results and provide your reasoning."}
                ]
                
                thinking_kwargs = {
                    "model": ERNIE_THINKING_MODEL,
                    "messages": thinking_messages,
                    "stream": True,
                    "temperature": 0.2
                }

                thinking_response = ""
                thinking_stream = await get_openrouter_client().chat.completions.create(**thinking_kwargs)
                async for chunk in thinking_stream:
                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        thinking_response += content_chunk
                trace_ext.log("THINKING_COMPLETE", f"chars={len(thinking_response)}")

                reasoning_summary = ""
                if show_reasoning_panel and final_model != GEMINI_MODEL:
                    reasoning_summary = await _build_reasoning_summary(thinking_response, user_query, sub_intent)
                
                # Pass 2: Final model synthesizes the answer
                trace_ext.model_chain.append(final_model)
                trace_ext.log("SYNTHESIS_START", f"Final model synthesizing")
                
                synthesis_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on this analysis:\n\n{thinking_response}\n\nProvide a clear, well-structured answer to: {user_query}"}
                ]
                
                if context_messages:
                    synthesis_messages = [{"role": "system", "content": system_prompt}] + context_messages[-4:] + [synthesis_messages[-1]]
                
                synthesis_kwargs = {
                    "model": final_model,
                    "messages": synthesis_messages,
                    "stream": True
                }
                synthesis_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
                if final_temp is not None:
                    synthesis_kwargs["temperature"] = final_temp
                
                stream = await final_client.chat.completions.create(**synthesis_kwargs)
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
                if reasoning_summary:
                    yield "[THINKING]"
                    yield self._sanitize_output_text(reasoning_summary)
                    yield "[/THINKING]"
            else:
                # Simple web search: direct synthesis (no reasoning needed)
                trace_ext.confidence_gating_skipped = True
                trace_ext.model_chain.append(final_model)
                trace_ext.log("EXTERNAL_ROUTE", f"direct({route_reason}) -> {final_model}")
                
                create_kwargs = {
                    "model": final_model,
                    "messages": messages,
                    "stream": True
                }
                create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
                if final_temp is not None:
                    create_kwargs["temperature"] = final_temp
                stream = await final_client.chat.completions.create(**create_kwargs)
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
            
            trace_ext.log("COMPLETE", f"models={trace_ext.model_chain}")

# Global processor instance
llm_processor = LLMProcessor()
