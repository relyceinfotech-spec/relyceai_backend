"""
Relyce AI - LLM Processor
Handles message processing with streaming support
Uses logic from existing Normal.py, Business.py, Deepsearch.py
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
    INTERNAL_SYSTEM_PROMPT,
    get_openai_client
)

class LLMProcessor:
    """
    Main LLM processor class that handles all chat modes.
    Imports and uses logic from existing Normal.py, Business.py, Deepsearch.py
    """
    
    def __init__(self):
        self.model = LLM_MODEL
    
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
                model="gpt-3.5-turbo", # Use cheaper model for summary
                messages=[{"role": "user", "content": prompt_content}],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception:
            return existing_summary # Return old summary if update fails

    
    async def process_internal_query(self, user_query: str, personality: Optional[Dict] = None, user_settings: Optional[Dict] = None) -> str:
        """
        Handle internal queries (greetings, math, code, logic).
        Now supports PERSONALITY customization.
        """
        if personality:
            system_prompt = get_internal_system_prompt_for_personality(personality, user_settings)
        else:
            from app.llm.router import _build_user_context_string
            system_prompt = INTERNAL_SYSTEM_PROMPT + _build_user_context_string(user_settings)
            
        response = await get_openai_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content
    
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
        
        response = await get_openai_client().chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        return response.choices[0].message.content, selected_tools
    
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
            response_text = await self.process_internal_query(user_query, personality, user_settings)
            return {
                "success": True,
                "response": response_text,
                "mode_used": "internal",
                "tools_activated": []
            }
        else:
            response_text, tools = await self.process_external_query(
                user_query, mode, context_messages, personality, user_settings
            )
            return {
                "success": True,
                "response": response_text,
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
        print(f"[LATENCY] Start processing: {0:.4f}s")
        
        # Combined Analysis (Intent + Tools)
        # Pass context_messages for sticky routing (history awareness)
        # Pass personality for Content Mode overrides (Pure LLM / Web Search)
        analysis = await analyze_and_route_query(user_query, mode, context_messages, personality=personality)
        intent = analysis.get("intent", "EXTERNAL")
        selected_tools = analysis.get("tools", [])
        
        print(f"[LATENCY] Analysis Complete ({intent}): {time.time() - start_time:.4f}s")
        
        if intent == "INTERNAL" or not selected_tools:
            # Stream internal response
            if personality:
                 # 🎭 Advanced Routing: Dynamic Persona Switching (ONLY for Generic/Normal + Relyce AI)
                 # If sub_intent is detected (debugging, sql, etc.), use that specific prompt overlay
                 sub_intent = analysis.get("sub_intent", "general")
                 from app.llm.router import INTERNAL_MODE_PROMPTS
                 
                 base_prompt = get_internal_system_prompt_for_personality(personality, user_settings, user_id)
                 
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
            
            # 🚀 IMMEDIATELY yield a signal token to hide the loader on frontend
            yield " " 

            # Construct messages with Context (Option 2 Support)
            messages = [{"role": "system", "content": system_prompt}]
            
            # Inject context messages (including potential Summary system message)
            if context_messages:
                # If we have a summary system message injected by websocket.py, it will be first
                # We simply append all context messages. 
                # Note: Logic to prune/summarize is handled in websocket.py/main.py before calling this.
                messages.extend(context_messages)
            
            messages.append({"role": "user", "content": user_query})

            stream = await get_openai_client().chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
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
            if personality:
                # 🎭 Personalities are now honored in ALL modes
                from app.llm.router import get_system_prompt_for_personality
                system_prompt = get_system_prompt_for_personality(personality, user_settings, user_id)
                
                # If in Business or Deep Search, append mode-specific constraints to the personality
                if mode != "normal":
                    from app.llm.router import get_system_prompt_for_mode
                    mode_prompt = get_system_prompt_for_mode(mode, user_settings, user_id)
                    system_prompt = f"{system_prompt}\n\n**MODE CONSTRAINT ({mode.upper()}):**\n{mode_prompt}"
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
            
            # 🚀 Signal frontend to show message bubble
            yield " "

            stream = await get_openai_client().chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

# Global processor instance
llm_processor = LLMProcessor()
