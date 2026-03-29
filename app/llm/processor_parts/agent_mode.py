"""Extracted stream entry for LLMProcessor."""
from __future__ import annotations

from app.llm.processor import *  # type: ignore
from app.chat.mode_mapper import normalize_chat_mode


async def process_agent_mode_impl(
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
    mode: str = "smart",
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
    mode = normalize_chat_mode(mode)
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
    # - Smart: limit to essential tools only
    if mode == "smart" and agent_result.tool_allowed:
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

    if mode == "research_pro" and agent_result.tool_allowed:
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
    elif agent_result.action_decision and agent_result.action_decision.action_type == "ACTION":
        _confidence = 0.6
    elif agent_result.action_decision and agent_result.action_decision.action_type == "QUESTION":
        _confidence = 0.9
    
    intel_payload = {
        "mode": ui_mode,
        "action_type": action_type,
        "decision": agent_result.decision,
        "tool_allowed": agent_result.tool_allowed,
        "risk_tier": "low",
        "reversible": True,
        "time_sensitive": False,
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
    active_executions[execution_id] = {
        "ctx": exec_ctx,
        "created_at": _time.time(),
        "user_id": str(user_id or ""),
    }
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

    # --- Fast Path Bypass (Phase 3) ---
    if agent_result.tool_allowed:
        import re as _re
        from app.agent.tool_executor import execute_tool, ToolCall
        
        fast_path_match = None
        q_strip = user_query.strip().lower()
        
        # 1. Time lookup
        time_pattern = r"\bwhat time\b|\bcurrent time\b|\bwhat's the time\b"
        if _re.search(time_pattern, q_strip):
            fast_path_match = "get_current_time"
            
        # 2. Math/Calculation (pure digits and operators)
        elif _re.search(r"^[\d\.\s\+\-\*\/\(\)]+$", q_strip) and any(c.isdigit() for c in q_strip) and any(op in q_strip for op in ['+', '-', '*', '/']):
            fast_path_match = "calculate"
            
        if fast_path_match and fast_path_match in agent_result.allowed_tools:
            yield f"[INFO]{_json.dumps({'agent_state': 'fast_path', 'tool': fast_path_match})}"
            call = ToolCall(name=fast_path_match, args=user_query)
            res = await execute_tool(call, exec_ctx)
            if res.success:
                yield f"{res.data}"
                yield f"[INFO]{_json.dumps({'agent_state': 'completed', 'completed': True})}"
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
            plan_graph = await compile_plan_graph(session_id, task_id, user_query)
        
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
                    "content": PROCESSOR_SYNTHESIS_SYSTEM_PROMPT
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
            "content": AGENT_CRITICAL_OVERRIDE_PROMPT
        })
        
    step_count = 0
    forced_tool_attempted = False
    fact_lookup_query = bool(re.search(
        r"\b(who is|founder|ceo|owner|director|price|market cap|valuation|address|incorporated|registered|linkedin|crunchbase|cbse|matric|matriculation|affiliation|board|school|college)\b",
        (user_query or "").lower()
    ))
    
    # --- Apply Phase 3 Budget Limits ---
    agent_budget = getattr(exec_ctx, "budget", None)
    if not agent_budget:
        from app.agent.agent_state import AgentBudget
        agent_budget = AgentBudget()
        
    step_budget_limit = int(getattr(agent_budget, "max_steps", 8))
    tool_budget_limit = int(getattr(agent_budget, "max_total_tool_calls", getattr(agent_budget, "max_tool_calls", 12)))
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
                    "content": f"Fatal: Task has exceeded the absolute maximum allowed tool calls ({tool_budget_limit}) across all resumes. Finalize immediately."
                })
                print(f"[Agent] Task {task_id} exhausted its absolute tool call budget.")
                break

        if step_count > step_budget_limit:
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

        print(f"[Agent] Cycle {step_count}/{step_budget_limit} | Streaming via {model_to_use} (Time: {_time.time() - start_time:.4f}s)")
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
            if mode == "smart" and len(valid_calls) > 1:
                valid_calls = [valid_calls[0]]

            # Record the pre-tool text into full_response
            first_call_idx = step_output.find("TOOL_CALL:")
            before_tool = step_output[:first_call_idx].strip()
            full_response += before_tool
            truncated_step_output = step_output.strip()

            # Budget check
            exec_ctx.tool_calls_made += len(valid_calls)
            if exec_ctx.tool_calls_made > tool_budget_limit:
                exec_ctx.forced_finalize = True
                exec_ctx.degraded = True
                exec_ctx.degradation_reasons.append("Max total tool calls reached")
                print(f"[Agent] Max tool calls ({tool_budget_limit}) reached, forcing finalize")
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
                    messages.append({"role": "user", "content": PROCESSOR_TOOL_LOOP_BREAKER_USER_PROMPT_TEMPLATE.format(tool_name=tc.name)})
                    _loop_broken = True
                    break
            if _loop_broken:
                continue

            print(f"[Agent] Executing {len(valid_calls)} tools in PARALLEL: {[tc.name for tc in valid_calls]} "
                  f"[total calls: {exec_ctx.tool_calls_made}/{tool_budget_limit}]")

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
            time_aware = False
            if "date" in user_query.lower() or "today" in user_query.lower() or "now" in user_query.lower():
                time_aware = True
            requires_live = False
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
        if step_count >= step_budget_limit or exec_ctx.total_operations >= step_budget_limit:
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

    # --- FIX 3/7 + Phase 4: Final Merge & Self Critic ---
    if not exec_ctx.forced_finalize and not exec_ctx.final_delivery:
        exec_ctx.final_delivery = True
        
        # Phase 4 pre-finalization review
        if agent_result.action_decision and agent_result.action_decision.action_type in ["TASK", "RESEARCH", "CODING"]:
            from app.agent.self_critic import run_self_critic
            critic_res = await run_self_critic(user_query, messages, client, model_to_use)
            if not critic_res.get("passed", True) and agent_budget.max_steps > 1:
                print(f"[Agent] Self-Critic FAILED. Feedback: {critic_res.get('feedback', '')}")
                messages.append({"role": "system", "content": f"Self-Review Feedback: Your drafted response is missing crucial elements. FIX THIS:\n{critic_res.get('feedback', '')}\n\nYou must provide the missing data before finalization."})
                
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
