"""
Relyce AI - LLM Processor
Handles message processing with streaming support
Includes legacy prompts and routing logic consolidated into app/llm
"""
import json
from typing import AsyncGenerator, List, Dict, Any, Optional
from openai import OpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL
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

class LLMProcessor:
    """
    Main LLM processor class that handles all chat modes.
    Consolidated logic from legacy mode scripts
    """
    
    def __init__(self):
        self.model = LLM_MODEL

    def _get_sampling(self, personality: Optional[Dict]) -> Dict[str, Any]:
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

        # Hard caps for safety-critical domains
        if specialty in {"coding", "legal"}:
            temp = min(temp, 0.3)

        sampling: Dict[str, Any] = {"temperature": temp}

        # Specialty-specific knobs
        if specialty == "coding":
            sampling.update({"top_p": 0.9, "frequency_penalty": 0, "presence_penalty": 0})
        elif specialty == "creative":
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
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception:
            return existing_summary # Return old summary if update fails

    
    async def process_internal_query(self, user_query: str, personality: Optional[Dict] = None, user_settings: Optional[Dict] = None, mode: str = "normal") -> str:
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
        if personality and personality.get("id") == "coding_buddy":
            model_to_use = "gpt-5-nano"
            print(f"[LLM] ⚡ Switching to {model_to_use} for Coding Buddy")

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
        response = await get_openai_client().chat.completions.create(**create_kwargs)
        return self._sanitize_output_text(response.choices[0].message.content)
    
    async def process_external_query(
        self, 
        user_query: str, 
        mode: str = "normal",
        context_messages: Optional[List[Dict]] = None,
        personality: Optional[Dict] = None,
        user_settings: Optional[Dict] = None
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
            model_to_use = "gpt-5-nano"
            print(f"[LLM] ⚡ Switching to {model_to_use} for Coding Buddy")

        create_kwargs = {
            "model": model_to_use,
            "messages": messages
        }
        create_kwargs.update(self._get_sampling(personality))
        response = await get_openai_client().chat.completions.create(**create_kwargs)
        
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
            response_text = await self.process_internal_query(user_query, personality, user_settings, mode=mode)
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
        
        print(f"[LATENCY] Analysis/Router Complete ({intent}): {time.time() - start_time:.4f}s (Analysis took: {time.time() - t_analysis_start:.4f}s)")
        
        if intent == "INTERNAL" or not selected_tools:
            # Stream internal response
            if personality:
                 # 🎭 Advanced Routing: Dynamic Persona Switching (ONLY for Generic/Normal + Relyce AI)
                 # If sub_intent is detected (debugging, sql, etc.), use that specific prompt overlay
                 sub_intent = analysis.get("sub_intent", "general")
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

            print(f"[LATENCY] Internal: Calling OpenAI Stream (Model: {self.model})... (Time: {time.time() - start_time:.4f}s)")
            t_stream_start = time.time()
            
            # Check for Coding Buddy override
            model_to_use = self.model
            if personality and personality.get("id") == "coding_buddy":
                model_to_use = "gpt-5-nano"
                print(f"[LLM] ⚡ Switching to {model_to_use} for Coding Buddy")
            
            create_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "stream": True
            }
            create_kwargs.update(self._get_sampling(personality))
            stream = await get_openai_client().chat.completions.create(**create_kwargs)
            print(f"[LATENCY] Internal: OpenAI Connection Established: {time.time() - start_time:.4f}s (Waited: {time.time() - t_stream_start:.4f}s)")
            
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
            
            print(f"[LATENCY] Ready to Stream Synthesis: {time.time() - start_time:.4f}s")
            
            # Removed premature signal token to keep Search Loader active until first real token
            # yield " "

            # Check for Coding Buddy override
            model_to_use = self.model
            if personality and personality.get("id") == "coding_buddy":
                model_to_use = "gpt-5-nano"
                print(f"[LLM] ⚡ Switching to {model_to_use} for Coding Buddy")

            create_kwargs = {
                "model": model_to_use,
                "messages": messages,
                "stream": True
            }
            create_kwargs.update(self._get_sampling(personality))
            stream = await get_openai_client().chat.completions.create(**create_kwargs)
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield self._sanitize_output_text(chunk.choices[0].delta.content)

# Global processor instance
llm_processor = LLMProcessor()
