"""
Relyce AI - WebSocket Manager
Multi-device safe WebSocket implementation
"""
import json
import asyncio
import time

from typing import Dict, List, Optional, Tuple
from fastapi import WebSocket, WebSocketDisconnect

# Rate limiter for background memory extraction (max 3 concurrent workers)
_EXTRACTION_SEMAPHORE = asyncio.Semaphore(3)
from app.auth import verify_token, get_firestore_db
from app.platform import get_ai_platform, CapabilityRequest
from app.chat.context import get_context_for_llm, update_context_with_exchange
from app.chat.user_profile import get_user_settings, merge_settings
from app.chat.context import (
    SUMMARY_TRIGGER_MESSAGES, 
    KEEP_LAST_MESSAGES,
    get_session_summary, 
    update_session_summary, 
    prune_context_messages, 
    get_raw_message_count
)
from app.chat.history import save_message_to_firebase
from app.chat.runtime_helpers import resolve_personality_context
from app.chat.session_rules import increment_turn as _increment_session_turn
from app.governance.chat_rate_limit import check_rate_limit
from app.config import MAX_CHAT_MESSAGE_CHARS, RATE_LIMIT_PER_MINUTE
from app.state.phase_guard import get_phase_guard
from app.safety.stream_moderator import StreamingOutputModerator
from app.context.response_contract import normalize_chat_response, build_final_answer_payload
from app.streaming.stream_controller import StreamController
from app.realtime.connection_manager import ConnectionManager
from app.chat.mode_mapper import normalize_chat_mode, is_agent_premium_mode

# SSE deterministic sequence counter (per chat session)
_event_seq_counters: Dict[Tuple, int] = {}
_agent_rate_limits: Dict[str, list] = {}  # user_id -> [timestamp, ...] for agent mode rate limiting
_agent_active_requests: set = set()  # user_ids with agent requests currently in-flight
_AGENT_PREMIUM_PLANS = {"plus", "pro", "business"}
_AGENT_FREE_MAX_MESSAGES_PER_CHAT = 2

# Cancel flags for interrupt safety
_cancel_flags: Dict[Tuple, bool] = {}

manager = ConnectionManager()
ai_platform = get_ai_platform()



def _mask_identifier(value: str) -> str:
    text = str(value or "")
    if len(text) <= 6:
        return "***"
    return f"{text[:3]}***{text[-3:]}"


def _normalize_membership_plan(raw_plan: str, raw_plan_name: str = "") -> str:
    value = str(raw_plan or raw_plan_name or "").strip().lower()
    value = value.replace(" ", "")
    if value.endswith("plan"):
        value = value[:-4]
    if value == "professional":
        value = "pro"
    if value == "businessplan":
        value = "business"
    if value == "proplan":
        value = "pro"
    if value == "plusplan":
        value = "plus"
    return value or "free"


async def handle_websocket_message(
    websocket: WebSocket,
    connection_id: str,
    data: dict,
    manager: ConnectionManager
) -> None:
    """
    Handle incoming WebSocket message.
    Process the message and stream response to all connected devices.
    """
    msg_type = data.get("type", "message")
    
    info = manager.connection_info.get(connection_id, {})
    user_id = info.get("user_id", "anonymous")
    chat_id = info.get("chat_id", "default")
    chat_key = info.get("chat_key") or (user_id, chat_id)
    
    if msg_type == "stop":
        # Stop generation
        manager.set_stop_flag(chat_key)
        await manager.broadcast_to_chat(
            chat_key,
            json.dumps({"type": "info", "content": "Generation stopped"})
        )
        return
    
    if msg_type == "ping":
        await manager.send_personal_message(
            json.dumps({"type": "pong"}), 
            websocket
        )
        return
    
    if msg_type != "message":
        return
        
    print(f"[WS DEBUG] Received payload: chat_mode={data.get('chat_mode')}, type={msg_type}, text_len={len(data.get('content', ''))}")
    
    # Process chat message
    content = data.get("content", "")
    chat_mode = normalize_chat_mode(data.get("chat_mode", "smart"))
    personality_id = data.get("personality_id")
    user_settings = data.get("user_settings")
    effective_settings = merge_settings(get_user_settings(user_id), user_settings)
    
    # Resolve personality context (provided -> session -> default).
    personality = None
    conn_info = manager.connection_info.get(connection_id, {})
    user_id = conn_info.get("user_id")
    if user_id:
        personality, personality_id = resolve_personality_context(
            user_id=user_id,
            session_id=chat_id,
            chat_mode=chat_mode,
            personality=personality,
            personality_id=personality_id,
        )
        if personality_id:
            conn_info["personality_id"] = personality_id
    if not content.strip():
        return

    if not check_rate_limit(user_id):
        await manager.send_personal_message(
            json.dumps({"type": "error", "content": f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min). Try again later."}),
            websocket
        )
        return

    # === AGENT MODE: Tiered Rate Limit + Concurrency Guard ===
    if is_agent_premium_mode(chat_mode):
        import time as _rl_time
        _now = _rl_time.time()
        _uid = user_id or "anonymous"

        # --- Concurrency Guard: No concurrent agent requests ---
        if _uid in _agent_active_requests:
            await manager.send_personal_message(
                json.dumps({"type": "error", "content": "An agent request is already in progress. Please wait for it to finish."}),
                websocket
            )
            return

        # --- Resolve membership plan (cached per connection) ---
        _plan = conn_info.get("_cached_plan")
        if not _plan:
            try:
                db = get_firestore_db()
                if db and _uid != "anonymous":
                    _user_doc = db.collection("users").document(_uid).get()
                    if _user_doc.exists:
                        _membership = (_user_doc.to_dict() or {}).get("membership", {})
                        _plan = _normalize_membership_plan(
                            _membership.get("plan"),
                            _membership.get("planName"),
                        )
                    else:
                        _plan = "free"
                else:
                    _plan = "free"
            except Exception as _e:
                print(f"[AgentRateLimit] Plan lookup failed: {_e}")
                _plan = "free"
            conn_info["_cached_plan"] = _plan
        # Free users can access Agent tab, but only 2 messages per chat and no continuation payloads
        if _plan not in _AGENT_PREMIUM_PLANS:
            if any(marker in content for marker in ("CONTINUE_AVAILABLE", "Continue generating UI code", "<PREVIOUS_OUTPUT>")):
                await manager.send_personal_message(
                    json.dumps({"type": "error", "content": "Agent continuation is available only on Plus, Pro, or Business plans."}),
                    websocket
                )
                return

            _raw_count = get_raw_message_count(user_id, chat_id)
            _used_agent_messages = _raw_count // 2
            if _used_agent_messages >= _AGENT_FREE_MAX_MESSAGES_PER_CHAT:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "content": "Free Agent limit reached for this chat (2 messages). Upgrade to Plus, Pro, or Business for unlimited Agent usage."}),
                    websocket
                )
                return

        # --- Tiered limits ---
        _AGENT_LIMITS = {"free": 4, "plus": 6, "pro": 8, "business": 10}
        _limit = _AGENT_LIMITS.get(_plan, 4)

        # Sliding window check
        if _uid not in _agent_rate_limits:
            _agent_rate_limits[_uid] = []
        _agent_rate_limits[_uid] = [t for t in _agent_rate_limits[_uid] if _now - t < 60]
        if len(_agent_rate_limits[_uid]) >= _limit:
            await manager.send_personal_message(
                json.dumps({"type": "error", "content": f"Agent rate limit exceeded ({_limit} req/min for {_plan.capitalize()} plan). Upgrade your plan for higher limits."}),
                websocket
            )
            return
        _agent_rate_limits[_uid].append(_now)

        # Mark this user as having an active agent request
        _agent_active_requests.add(_uid)

    # Clear any previous stop flag
    manager.clear_stop_flag(chat_key)
    
    # Reset phase guard for this session (new execution = fresh state machine)
    get_phase_guard().reset(session_id=f"{user_id}:{chat_id}")
    
    # Reset SSE sequence counter
    _event_seq_counters[chat_key] = 0
    
    # Clear cancel flag
    _cancel_flags[chat_key] = False
    
    # Notify all devices - processing started (IMMEDIATE FEEDBACK)
    await manager.broadcast_to_chat(
        chat_key,
        json.dumps({"type": "info", "content": "processing"})
    )
    # Send immediate empty token to transition UI from loading to streaming state
    await manager.stream_to_chat(chat_key, "\u200B")
    
    # Extract and store user facts (Run in background to not block)
    # Use unique_user_id (resolved during auth) for memory storage
    memory_user_id = conn_info.get("unique_user_id") or user_id
    if memory_user_id and memory_user_id != "anonymous":
        try:
            from app.chat.smart_memory import process_message as smart_memory_process
            # Fire and forget smart memory extraction
            asyncio.create_task(asyncio.to_thread(smart_memory_process, memory_user_id, content))
        except Exception as e:
            print(f"[WS] Smart memory extraction error (non-blocking): {e}")

    # Safety Check: Max Message Length
    if len(content) > MAX_CHAT_MESSAGE_CHARS:
        # Release lock for oversized agent requests that returned early.
        if is_agent_premium_mode(chat_mode):
            _agent_active_requests.discard(user_id or "anonymous")
        await manager.send_personal_message(
            json.dumps({"type": "error", "content": f"Message too long (max {MAX_CHAT_MESSAGE_CHARS} chars)"}), 
            websocket
        )
        return
    
    resolved_personality_id = personality_id
    if personality and not resolved_personality_id:
        resolved_personality_id = personality.get("id")

    # Get context for LLM (personality-aware)
    try:
        context_messages = await asyncio.to_thread(get_context_for_llm, user_id, chat_id, resolved_personality_id, chat_mode)
    except Exception as e:
        print(f"[WS] Failed to load context: {e}")
        context_messages = []
    
    # Stream response to all connected devices
    stream_controller = StreamController(min_info_interval_s=0.2)
    full_response = ""
    stream_generator = None
    stream_moderator = StreamingOutputModerator(query=content)
    # BACKPRESSURE: Bounded queue between LLM producer and WebSocket consumer
    # If client is slow, queue fills up and naturally throttles the model stream
    BACKPRESSURE_QUEUE_SIZE = 128
    token_queue = asyncio.Queue(maxsize=BACKPRESSURE_QUEUE_SIZE)
    _SENTINEL = object()  # Marks end of stream
    
    retrieved_context_meta = {}
    
    async def _producer():
        """Reads from LLM generator and enqueues tokens."""
        nonlocal stream_generator
        try:
            req = CapabilityRequest(
                user_query=content,
                chat_mode=chat_mode,
                context_messages=context_messages,
                personality=personality,
                user_settings=effective_settings,
                user_id=memory_user_id,
                session_id=chat_id,
            )
            stream_generator = ai_platform.run_stream(req)
            async for token in stream_generator:
                await token_queue.put(token)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await token_queue.put(e)  # Signal error to consumer
        finally:
            await token_queue.put(_SENTINEL)
    
    try:
        producer_task = asyncio.create_task(_producer())
        
        start_time = time.time()
        while True:
            item = await token_queue.get()
            
            # End of stream
            if item is _SENTINEL:
                break
            
            # Error propagation from producer
            if isinstance(item, Exception):
                raise item
            
            token = item
            
            # Check stop flag
            if manager.should_stop(chat_key):
                producer_task.cancel()
                await manager.broadcast_to_chat(
                    chat_key,
                    json.dumps({"type": "info", "content": "stopped"})
                )
                break
            
            # Intercept [INFO] messages (like search status)
            if token.strip().startswith("[INFO]"):
                clean_info = token.replace("[INFO]", "").strip()
                
                if clean_info.startswith("RETRIEVED_CONTEXT:"):
                    try:
                        retrieved_context_meta = json.loads(clean_info.replace("RETRIEVED_CONTEXT:", "", 1))
                    except Exception:
                        pass
                    continue
                
                # Phase Guard: validate agent_state transitions
                try:
                    parsed = json.loads(clean_info) if clean_info.startswith("{") else {}
                    if "agent_state" in parsed:
                        phase_ok = get_phase_guard().transition(
                            f"{user_id}:{chat_id}", parsed["agent_state"]
                        )
                        if not phase_ok:
                            continue  # Skip regressed phase emission
                except (json.JSONDecodeError, Exception):
                    pass
                
                filtered_info = stream_controller.filter_info(clean_info)
                if not filtered_info:
                    continue

                # SSE sequence counter
                _event_seq_counters[chat_key] = _event_seq_counters.get(chat_key, 0) + 1
                seq = _event_seq_counters[chat_key]
                
                await manager.broadcast_to_chat(
                    chat_key,
                    json.dumps({"type": "info", "content": filtered_info, "_event_seq": seq})
                )
                continue # Do not append to full_response or stream as token
            
            # Check cancel flag (interrupt safety)
            if _cancel_flags.get(chat_key, False):
                producer_task.cancel()
                await manager.broadcast_to_chat(
                    chat_key,
                    json.dumps({"type": "info", "content": '{"agent_state": "cancelled"}'})
                )
                break

            safe_chunk = stream_moderator.ingest(token)
            if safe_chunk:
                full_response += safe_chunk
                import time as _t
                _t_recv = _t.time()
                await manager.stream_to_chat(chat_key, safe_chunk)
                _t_sent = _t.time()
            # Removed per-token print to prevent event loop blocking on Windows terminal
        
        tail_chunk = stream_moderator.finalize()
        if tail_chunk:
            full_response += tail_chunk
            await manager.stream_to_chat(chat_key, tail_chunk)

        final_payload = build_final_answer_payload(
            {"success": True, "response": full_response},
            user_query=content,
            chat_mode=chat_mode,
        )

        await manager.broadcast_to_chat(
            chat_key,
            json.dumps({"type": "final_answer", "content": final_payload})
        )

        # Send completion signal
        await manager.broadcast_to_chat(
            chat_key,
            json.dumps({"type": "done", "content": ""})
        )
        
        # Update context with this exchange
        update_context_with_exchange(user_id, chat_id, content, full_response, resolved_personality_id, chat_mode)
        
        # === SAFE CHAT AGENT: Track session turn ===
        try:
            _increment_session_turn(user_id, chat_id, content, full_response)
        except Exception as e:
            print(f"[WS] Session turn tracking failed (non-blocking): {e}")
        
        # Save to Firebase (background, non-blocking)
        async def save_history_background():
            try:
                await asyncio.to_thread(save_message_to_firebase, user_id, chat_id, "user", content, resolved_personality_id)
                await asyncio.to_thread(save_message_to_firebase, user_id, chat_id, "assistant", full_response, resolved_personality_id)
                
                # Smart Memory Compression
                from app.llm.router import get_openrouter_client
                from app.memory.summary_manager import summarize_if_needed
                fresh_context_msgs = get_context_for_llm(user_id, chat_id, resolved_personality_id, chat_mode)
                await summarize_if_needed(user_id, chat_id, fresh_context_msgs, get_openrouter_client())
            except Exception as e:
                print(f"[WS] Background history save failed: {e}")

        asyncio.create_task(save_history_background())

        # === Memory Extraction (background, rate-limited, 3s timeout) ===
        # Skip if response is too short (~40 tokens ~= 200 chars); trivial replies produce junk
        if user_id and user_id != "anonymous" and full_response and len(full_response) > 200:
            async def _extract_vector_memory():
                async with _EXTRACTION_SEMAPHORE:
                    try:
                        from app.memory.vector_memory import extract_and_store_memories
                        stored = await asyncio.wait_for(
                            extract_and_store_memories(user_id, content, full_response),
                            timeout=3.0
                        )
                        if stored:
                            print(f"[WS] Vector memory: {stored} facts extracted for user={_mask_identifier(user_id)}")
                    except asyncio.TimeoutError:
                        print("[WS] Vector memory extraction timed out (3s)")
                    except Exception as e:
                        print(f"[WS] Vector memory extraction failed (non-blocking): {e}")
            
            asyncio.create_task(_extract_vector_memory())

            async def _extract_knowledge_graph():
                async with _EXTRACTION_SEMAPHORE:
                    try:
                        from app.memory.knowledge_graph import extract_and_store_graph
                        stored = await asyncio.wait_for(
                            extract_and_store_graph(user_id, content, full_response),
                            timeout=3.0
                        )
                        if stored:
                            print(f"[WS] Knowledge graph: {stored} triples extracted for user={_mask_identifier(user_id)}")
                    except asyncio.TimeoutError:
                        print("[WS] Knowledge graph extraction timed out (3s)")
                    except Exception as e:
                        print(f"[WS] Knowledge graph extraction failed (non-blocking): {e}")
            
            asyncio.create_task(_extract_knowledge_graph())

        # === Retrieval Quality Scoring (background) ===
        if retrieved_context_meta and full_response and user_id != "anonymous":
            async def _score_retrieval_quality():
                try:
                    from app.context.citation_engine import score_retrievals
                    await score_retrievals(user_id, retrieved_context_meta, full_response)
                except Exception as e:
                    print(f"[WS] Retrieval scoring failed (non-blocking): {e}")
            
            asyncio.create_task(_score_retrieval_quality())

        # === User Profiler: Learn from this streaming interaction ===
        if user_id and user_id != "anonymous":
            try:
                from app.llm.user_profiler import user_profiler
                profile = await user_profiler.load_profile(user_id)
                user_profiler.update_from_interaction(
                    profile,
                    sub_intent=chat_mode,
                    query=content,
                    response_length=len(full_response),
                    emotions=[],  # Emotions already processed in processor
                    model_used=chat_mode
                )
            except Exception as e:
                print(f"[WS] Profiler update failed (non-blocking): {e}")
        
    except asyncio.CancelledError:
        print("[WS] Stream cancelled by client disconnect")
        if stream_generator:
            try:
                await stream_generator.aclose()
            except Exception:
                pass
        raise  # Re-raise to let the router handle the disconnect
    except Exception as e:
        error_msg = "Error processing message"
        print(f"[WS] {error_msg} type={e.__class__.__name__}")
        await manager.broadcast_to_chat(
            chat_key,
            json.dumps({"type": "error", "content": error_msg})
        )
        if stream_generator:
            try:
                await stream_generator.aclose()
            except Exception:
                pass
    finally:
        # Release agent concurrency lock
        _uid_cleanup = (user_id or "anonymous")
        _agent_active_requests.discard(_uid_cleanup)




























