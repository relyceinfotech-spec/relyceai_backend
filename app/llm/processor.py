"""
Relyce AI - LLM Processor
Handles message processing with streaming support
Includes legacy prompts and routing logic consolidated into app/llm
"""
import json
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
}

CREATIVE_INTENTS = {
    "content_creation",
    "ui_design",
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
    simple_intents = ["casual_chat", "general", "code_explanation", "sql", "content_creation", "ui_design"]
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


def get_model_for_intent(mode: str, sub_intent: str, personality: Optional[Dict] = None, query: str = "") -> Tuple[str, str, float, bool]:
    """
    Determine which client/model to use based on mode, sub_intent, and personality.
    Now includes confidence gating for thinking model.
    
    Returns: (client_type, model_id, temperature, needs_thinking)
    - client_type: "openai" or "openrouter"
    - model_id: The model to use
    - temperature: Suggested temperature
    - needs_thinking: Whether to use 2-pass pipeline
    """
    # Business mode always uses OpenAI
    if mode == "business":
        return ("openai", LLM_MODEL, 0.4, False)
    
    # Check if thinking is warranted (confidence gating)
    needs_thinking = should_use_thinking_model(query, sub_intent)

    # Always use thinking pass for any coding-related intent
    if sub_intent in LOGIC_CODING_INTENTS:
        return ("openrouter", ERNIE_THINKING_MODEL, 0.2, True)

    # UI/creative requests should stay creative (no coding clamp)
    if sub_intent in CREATIVE_INTENTS:
        temp = _resolve_temperature(personality, "creative")
        if temp < 0.7:
            temp = 0.7
        return ("openrouter", GEMINI_MODEL, temp, False)
    
    # Analysis/Research with thinking gate
    if sub_intent in ["analysis", "research", "reasoning"]:
        if needs_thinking:
            return ("openrouter", ERNIE_THINKING_MODEL, 0.2, True)
        else:
            # Skip thinking, go direct to Gemini
            return ("openrouter", GEMINI_MODEL, 0.4, False)
    
    # Default: General/Personality uses Gemini Flash Lite
    fallback_specialty = personality.get("specialty", "general") if personality else "general"
    temp = _resolve_temperature(personality, fallback_specialty)
    return ("openrouter", GEMINI_MODEL, temp, False)

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
        elif is_creative_intent and temp < 0.7:
            temp = 0.7

        sampling: Dict[str, Any] = {"temperature": temp}

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
        sub_intent: str = "general"
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
            
        # Check for Coding Buddy override
        model_to_use = self.model
        if sub_intent in LOGIC_CODING_INTENTS:
            model_to_use = CODING_MODEL
        if personality and personality.get("id") == "coding_buddy":
            model_to_use = CODING_MODEL
            print(f"[LLM] ⚡ Switching to {model_to_use} for Coding Buddy")

        client = get_openrouter_client() if model_to_use == CODING_MODEL else get_openai_client()
        create_kwargs = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        }
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
        sub_intent: str = "general"
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
        # Determine system prompt
        if personality and mode == "normal":
            # 🎭 Personalities are ONLY honored in Normal Mode
            from app.llm.router import get_system_prompt_for_personality
            system_prompt = get_system_prompt_for_personality(personality, user_settings)
        else:
            # Business / Deep Search / Normal (without persona) -> Use default System Prompt defined in router.py
            # This ensures Business Mode uses the EXACT "Elite Strategic Advisor" prompt from Business.py
            system_prompt = get_system_prompt_for_mode(mode, user_settings)

        
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_messages:
            messages.extend(context_messages[-6:])
        
        messages.append({
            "role": "user", 
            "content": f"Search Data:\n{context_str}\n\nUser Query: {user_query}"
        })
        
        # Check for Coding Buddy override
        model_to_use = self.model
        if personality and personality.get("id") == "coding_buddy":
            model_to_use = CODING_MODEL
            print(f"[LLM] ⚡ Switching to {model_to_use} for Coding Buddy")

        client = get_openrouter_client() if model_to_use == CODING_MODEL else get_openai_client()
        create_kwargs = {
            "model": model_to_use,
            "messages": messages
        }
        create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
        response = await client.chat.completions.create(**create_kwargs)
        
        return self._sanitize_output_text(response.choices[0].message.content), selected_tools
    
    async def process_message(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None
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
                sub_intent=analysis.get("sub_intent", "general")
            )
            return {
                "success": True,
                "response": self._sanitize_output_text(response_text),
                "mode_used": "internal",
                "tools_activated": []
            }
        else:
            response_text, tools = await self.process_external_query(
                user_query, mode, context_messages, personality, user_settings
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
        
        print(f"[LATENCY] Analysis/Router Complete ({intent}): {time.time() - start_time:.4f}s (Analysis took: {time.time() - t_analysis_start:.4f}s)")
        
        if intent == "INTERNAL" or not selected_tools:
            # Stream internal response
            if personality:
                 # 🎭 Advanced Routing: Dynamic Persona Switching (ONLY for Generic/Normal + Relyce AI)
                 # If sub_intent is detected (debugging, sql, etc.), use that specific prompt overlay
                 from app.llm.router import INTERNAL_MODE_PROMPTS
                 
                 base_prompt = get_internal_system_prompt_for_personality(personality, user_settings, user_id, mode=mode)
                 
                 # Dynamic Overlay: If mode is generic/normal and we have a specialized sub-intent
                 if mode == "normal" and sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
                     specialized_prompt = INTERNAL_MODE_PROMPTS[sub_intent]
                     system_prompt = f"{base_prompt}\n\n**MODE SWITCH: {sub_intent.upper()}**\n{specialized_prompt}"
                 else:
                     system_prompt = base_prompt
            else:
                 from app.llm.router import _build_user_context_string
                 system_prompt = INTERNAL_SYSTEM_PROMPT + _build_user_context_string(user_settings)

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
            client_type, model_to_use, temperature, is_thinking_pass = get_model_for_intent(mode, sub_intent, personality, user_query)
            
            # Track if gating skipped thinking
            if sub_intent in ["debugging", "system_design", "analysis", "research", "reasoning"] and not is_thinking_pass:
                trace.confidence_gating_skipped = True
                trace.log("GATING", "Skipped thinking pass (query too simple)")
            
            trace.model_chain.append(model_to_use)
            trace.is_thinking_pass = is_thinking_pass
            
            trace.log("ROUTE_DECISION", f"client={client_type}, model={model_to_use}, temp={temperature}, thinking={is_thinking_pass}")
            
            # Select the appropriate client
            if client_type == "openai":
                client = get_openai_client()
            else:
                client = get_openrouter_client()
            
            create_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "stream": True,
                "temperature": temperature
            }
            
            # For thinking models, we need to collect the full response for 2-pass
            if is_thinking_pass:
                # Pass 1: Get thinking from Ernie
                trace.log("THINKING_START", f"model={model_to_use}")
                yield "[THINKING]"
                thinking_response = ""
                stream = await client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        thinking_response += content_chunk
                        yield self._sanitize_output_text(content_chunk)
                
                yield "[/THINKING]"
                trace.log("THINKING_COMPLETE", f"chars={len(thinking_response)}")
                
                # Pass 2: Generate final output with appropriate model
                if sub_intent in LOGIC_CODING_INTENTS:
                    output_model = CODING_MODEL
                else:
                    output_model = GEMINI_MODEL
                
                trace.model_chain.append(output_model)
                trace.log("OUTPUT_START", f"model={output_model}")
                
                # Construct pass 2 messages with thinking context
                pass2_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on this analysis:\n\n{thinking_response}\n\nProvide a clear, well-formatted response to: {user_query}"}
                ]
                
                if context_messages:
                    pass2_messages = [{"role": "system", "content": system_prompt}] + context_messages + [pass2_messages[-1]]
                
                pass2_kwargs = {
                    "model": output_model,
                    "messages": pass2_messages,
                    "stream": True,
                    "temperature": 0.3  # Lower temp for synthesis
                }
                
                stream = await get_openrouter_client().chat.completions.create(**pass2_kwargs)
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
            else:
                # Single-pass: Direct streaming
                stream = await client.chat.completions.create(**create_kwargs)
                print(f"[LATENCY] Internal: Stream Connection Established: {time.time() - start_time:.4f}s (Waited: {time.time() - t_stream_start:.4f}s)")
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
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
            
            if mode == "business":
                # Business mode: OpenAI direct (no reasoning step)
                client = get_openai_client()
                model_to_use = self.model
                trace_ext.model_chain.append(model_to_use)
                trace_ext.log("EXTERNAL_ROUTE", f"Business → OpenAI ({model_to_use})")
                
                create_kwargs = {
                    "model": model_to_use,
                    "messages": messages,
                    "stream": True
                }
                create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
                stream = await client.chat.completions.create(**create_kwargs)
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
                        
            elif needs_reasoning:
                # 🧠 HYBRID: Web → Ernie (thinking) → Gemini (synthesis)
                trace_ext.is_thinking_pass = True
                trace_ext.model_chain.append(ERNIE_THINKING_MODEL)
                trace_ext.log("HYBRID_START", f"Reasoning needed → Ernie first")
                
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
                
                yield "[THINKING]"
                thinking_response = ""
                thinking_stream = await get_openrouter_client().chat.completions.create(**thinking_kwargs)
                async for chunk in thinking_stream:
                    if chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        thinking_response += content_chunk
                        yield self._sanitize_output_text(content_chunk)
                
                yield "[/THINKING]"
                trace_ext.log("THINKING_COMPLETE", f"chars={len(thinking_response)}")
                
                # Pass 2: Gemini synthesizes the final answer
                trace_ext.model_chain.append(GEMINI_MODEL)
                trace_ext.log("SYNTHESIS_START", f"Gemini synthesizing")
                
                synthesis_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Based on this analysis:\n\n{thinking_response}\n\nProvide a clear, well-structured answer to: {user_query}"}
                ]
                
                if context_messages:
                    synthesis_messages = [{"role": "system", "content": system_prompt}] + context_messages[-4:] + [synthesis_messages[-1]]
                
                synthesis_kwargs = {
                    "model": GEMINI_MODEL,
                    "messages": synthesis_messages,
                    "stream": True,
                    "temperature": 0.4
                }
                
                stream = await get_openrouter_client().chat.completions.create(**synthesis_kwargs)
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
            else:
                # Simple web search: Gemini direct (no reasoning needed)
                trace_ext.confidence_gating_skipped = True
                trace_ext.model_chain.append(GEMINI_MODEL)
                trace_ext.log("EXTERNAL_ROUTE", f"Simple query → Gemini direct")
                
                create_kwargs = {
                    "model": GEMINI_MODEL,
                    "messages": messages,
                    "stream": True
                }
                create_kwargs.update(self._get_sampling(personality, sub_intent=sub_intent))
                stream = await get_openrouter_client().chat.completions.create(**create_kwargs)
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield self._sanitize_output_text(chunk.choices[0].delta.content)
            
            trace_ext.log("COMPLETE", f"models={trace_ext.model_chain}")

# Global processor instance
llm_processor = LLMProcessor()
