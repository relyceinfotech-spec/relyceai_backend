"""Extracted stream entry for LLMProcessor."""
from __future__ import annotations

from app.llm.processor import *  # type: ignore
from app.chat.mode_mapper import normalize_chat_mode


async def process_message_stream_impl(
    self, 
    user_query: str, 
    mode: str = "smart",
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
    mode = normalize_chat_mode(mode)
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
    strict_structured_mode = False
    
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
                if mode == "smart" and sub_intent == "general":
                    # Main mode: single canonical structured explainer format.
                    system_prompt = get_system_prompt_for_mode("smart", user_settings, user_id, user_query, session_id=session_id)
                else:
                    from app.llm.router import _build_user_context_string
                    system_prompt = INTERNAL_SYSTEM_PROMPT + _build_user_context_string(user_settings)
                    if mode == "smart":
                        from app.llm.router import NORMAL_MARKDOWN_POLISH
                        system_prompt = f"{system_prompt}\n{NORMAL_MARKDOWN_POLISH}"
        else:
            if personality and mode == "smart":
                from app.llm.router import get_system_prompt_for_personality
                system_prompt = get_system_prompt_for_personality(personality, user_settings, user_id, user_query, session_id=session_id)
        else:
            system_prompt = get_system_prompt_for_mode(mode, user_settings, user_id, user_query, session_id=session_id)

    from app.llm.router import INTERNAL_MODE_PROMPTS
    if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
        specialized_prompt = INTERNAL_MODE_PROMPTS[sub_intent]
        system_prompt = f"{system_prompt}\n\n" + MODE_SWITCH_PROMPT_TEMPLATE.format(sub_intent=sub_intent.upper(), specialized_prompt=specialized_prompt)
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
    if not strict_structured_mode:
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
        
        if not strict_structured_mode:
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
    if not strict_structured_mode and user_id and user_profile:
        personalization = user_profiler.get_personalization_instruction(user_profile)
        if personalization:
            system_prompt = f"{system_prompt}\n\n{personalization}"

    # Inject strategy instruction
    if not strict_structured_mode and strategy_instruction:
        system_prompt = f"{system_prompt}\n\n{strategy_instruction}"

    # Inject prompt variant
    if not strict_structured_mode and prompt_variant_instruction:
        system_prompt = f"{system_prompt}\n\n{prompt_variant_instruction}"

    # === SAFE CHAT AGENT: Constraint Analysis (Steps 1-11) ===
    if not strict_structured_mode and constraint_result and constraint_result.constraint_prompt:
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
                {"role": "system", "content": PROCESSOR_CODING_THINKING_SYSTEM_PROMPT},
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
                    f"{system_prompt}\n\n" + PROCESSOR_CODING_IMPLEMENT_SUFFIX_PROMPT
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
            if strict_structured_mode:
                # Buffer full answer and enforce rigid structure at end.
                tokens_yielded += 1
                continue
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
    # No backend structural rewriting.
    if strict_structured_mode:
        yield streamed_output

    followup_payload = self._build_followup_payload(streamed_output, mode, session_id)
    yield f"[INFO]{json.dumps(followup_payload)}"

    # End of Unified Normal Mode
    return


