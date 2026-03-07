"""
Relyce AI - WebSocket Manager
Multi-device safe WebSocket implementation
"""
import json
import asyncio
import uuid
import time

try:
    import redis.asyncio as redis
except Exception:  # optional dependency
    redis = None
from typing import Dict, List, Set, Optional, Tuple
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Rate limiter for background memory extraction (max 3 concurrent workers)
_EXTRACTION_SEMAPHORE = asyncio.Semaphore(3)
from datetime import datetime
from app.auth import verify_token, get_firestore_db
from app.llm.processor import llm_processor
from app.chat.context import get_context_for_llm, update_context_with_exchange
from app.chat.user_profile import get_user_settings, get_session_personality_id, merge_settings
from app.chat.context import (
    SUMMARY_TRIGGER_MESSAGES, 
    KEEP_LAST_MESSAGES,
    get_session_summary, 
    update_session_summary, 
    prune_context_messages, 
    get_raw_message_count
)
from app.chat.history import save_message_to_firebase
from app.chat.session_rules import increment_turn as _increment_session_turn
from app.rate_limit import check_rate_limit
from app.config import MAX_CHAT_MESSAGE_CHARS, RATE_LIMIT_PER_MINUTE, MAX_WS_CONNECTIONS_PER_CHAT, MAX_WS_CONNECTIONS_TOTAL, REDIS_URL, REDIS_WS_CHANNEL_PREFIX
from app.state.phase_guard import get_phase_guard

# SSE deterministic sequence counter (per chat session)
_event_seq_counters: Dict[Tuple, int] = {}
_agent_rate_limits: Dict[str, list] = {}  # user_id -> [timestamp, ...] for agent mode rate limiting
_agent_active_requests: set = set()  # user_ids with agent requests currently in-flight
_AGENT_PREMIUM_PLANS = {"plus", "pro", "business"}
_AGENT_FREE_MAX_MESSAGES_PER_CHAT = 2

# Cancel flags for interrupt safety
_cancel_flags: Dict[Tuple, bool] = {}

class ConnectionManager:
    """
    Multi-device safe connection manager.

    Key Design (from architecture):
    - One user â†’ Multiple devices â†’ Multiple WebSocket connections â†’ SAME chat_id
    - Connections are organized by (user_id, chat_id) to prevent cross-user leakage
    - This allows multiple devices for the same user to connect to the same chat and receive synced responses
    """

    def __init__(self):
        # Structure: {(user_id, chat_id): {connection_id: WebSocket}}
        self.active_connections: Dict[Tuple[str, str], Dict[str, WebSocket]] = {}
        # Track user info per connection: {connection_id: {"user_id": str, "chat_id": str}}
        self.connection_info: Dict[str, Dict] = {}
        # Stop flags for generation: {(user_id, chat_id): bool}
        self.stop_flags: Dict[Tuple[str, str], bool] = {}
        # Connection counter for unique IDs
        self._counter = 0
        # Concurrency guard
        self._lock = asyncio.Lock()
        # Optional Redis pubsub for horizontal fanout
        self._node_id = f"node_{uuid.uuid4().hex[:8]}"
        self._redis = None
        self._pubsub = None
        self._pubsub_task = None

    async def _ensure_redis(self) -> None:
        if not REDIS_URL or redis is None:
            return
        if self._redis:
            return
        try:
            self._redis = redis.from_url(REDIS_URL, decode_responses=True)
            self._pubsub = self._redis.pubsub()
            await self._pubsub.psubscribe(f"{REDIS_WS_CHANNEL_PREFIX}*")
            self._pubsub_task = asyncio.create_task(self._pubsub_loop())
        except Exception as e:
            print(f"[WS] Redis init failed: {e}")
            self._redis = None
            self._pubsub = None
            self._pubsub_task = None

    async def _pubsub_loop(self) -> None:
        if not self._pubsub:
            return
        try:
            async for message in self._pubsub.listen():
                if message.get("type") not in ("pmessage", "message"):
                    continue
                try:
                    data = json.loads(message.get("data") or "{}")
                except Exception:
                    continue
                if data.get("source") == self._node_id:
                    continue
                chat_key = tuple(data.get("chat_key") or [])
                payload = data.get("payload")
                if chat_key and payload is not None:
                    await self._broadcast_local(chat_key, payload)
        except Exception as e:
            print(f"[WS] Redis pubsub loop stopped: {e}")

    def _generate_connection_id(self) -> str:
        """Generate unique connection ID"""
        self._counter += 1
        return f"conn_{self._counter}_{datetime.now().timestamp()}"

    def _chat_key(self, user_id: str, chat_id: str) -> Tuple[str, str]:
        """Scope connections by user to prevent cross-user leakage."""
        return (user_id, chat_id)

    def _is_connected(self, websocket: WebSocket) -> bool:
        """Check if a websocket is still open for sending."""
        try:
            return (
                websocket.client_state == WebSocketState.CONNECTED
                and websocket.application_state == WebSocketState.CONNECTED
            )
        except Exception:
            return False

    def _total_connections(self) -> int:
        return sum(len(v) for v in self.active_connections.values())

    async def connect(
        self,
        websocket: WebSocket,
        chat_id: str,
        user_id: str
    ) -> str:
        """
        Register WebSocket connection.
        Multiple connections allowed per chat_id (multi-device support).

        Returns: connection_id
        """
        await self._ensure_redis()
        connection_id = self._generate_connection_id()
        chat_key = self._chat_key(user_id, chat_id)

        async with self._lock:
            if MAX_WS_CONNECTIONS_TOTAL and self._total_connections() >= MAX_WS_CONNECTIONS_TOTAL:
                raise RuntimeError("Server is busy. Try again later.")
            existing = self.active_connections.get(chat_key, {})
            if MAX_WS_CONNECTIONS_PER_CHAT and len(existing) >= MAX_WS_CONNECTIONS_PER_CHAT:
                raise RuntimeError("Too many connections for this chat.")

            # Initialize chat_id bucket if not exists
            if chat_key not in self.active_connections:
                self.active_connections[chat_key] = {}

            # Add connection to chat_id bucket
            self.active_connections[chat_key][connection_id] = websocket

            # Track connection info
            self.connection_info[connection_id] = {
                "user_id": user_id,
                "chat_id": chat_id,
                "chat_key": chat_key,
                "connected_at": datetime.now().isoformat()
            }

        print(f"[WS] âœ… Connected: {connection_id} to chat {chat_id} (user={user_id})")
        print(f"[WS] Active connections for chat {chat_id} (user={user_id}): {len(self.active_connections[chat_key])}")

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Remove a connection gracefully.
        Other connections to same chat_id remain active.
        """
        async with self._lock:
            if connection_id not in self.connection_info:
                return

            info = self.connection_info[connection_id]
            chat_id = info["chat_id"]
            chat_key = info.get("chat_key") or self._chat_key(info["user_id"], chat_id)

            # Remove from active connections
            if chat_key in self.active_connections:
                if connection_id in self.active_connections[chat_key]:
                    del self.active_connections[chat_key][connection_id]

                # Clean up empty chat_id bucket
                if not self.active_connections[chat_key]:
                    del self.active_connections[chat_key]
                    # Also clean up stop flag
                    if chat_key in self.stop_flags:
                        del self.stop_flags[chat_key]

            # Remove connection info
            del self.connection_info[connection_id]

        print(f"[WS] âŒ Disconnected: {connection_id} from chat {chat_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """Send message to a specific connection"""
        try:
            if not self._is_connected(websocket):
                return
            await websocket.send_text(message)
        except Exception as e:
            print(f"[WS] Error sending message: {e}")

    async def _broadcast_local(
        self,
        chat_key: Tuple[str, str],
        message: str,
        exclude_connection: Optional[str] = None
    ) -> None:
        # Lock-free snapshot: dict.get() returns the live dict reference;
        # list() copies the items tuple-list atomically enough for our needs.
        # This avoids acquiring the asyncio.Lock on every single token,
        # which was serializing all WebSocket sends and adding latency.
        bucket = self.active_connections.get(chat_key)
        if not bucket:
            return

        connections = list(bucket.items())

        disconnected = []
        for conn_id, websocket in connections:
            if conn_id == exclude_connection:
                continue
            if not self._is_connected(websocket):
                disconnected.append(conn_id)
                continue
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(conn_id)

        for conn_id in disconnected:
            await self.disconnect(conn_id)

    async def _publish_broadcast(self, chat_key: Tuple[str, str], message: str) -> None:
        if not self._redis:
            return
        try:
            channel = f"{REDIS_WS_CHANNEL_PREFIX}{chat_key[0]}:{chat_key[1]}"
            payload = {
                "source": self._node_id,
                "chat_key": list(chat_key),
                "payload": message
            }
            await self._redis.publish(channel, json.dumps(payload))
        except Exception as e:
            print(f"[WS] Redis publish failed: {e}")

    async def broadcast_to_chat(
        self,
        chat_key: Tuple[str, str],
        message: str,
        exclude_connection: Optional[str] = None
    ) -> None:
        """
        Broadcast message to ALL connections in a chat_id.
        This enables multi-device sync - all devices see the same response.
        """
        await self._broadcast_local(chat_key, message, exclude_connection=exclude_connection)
        await self._publish_broadcast(chat_key, message)

    async def stream_to_chat(
        self,
        chat_key: Tuple[str, str],
        token: str
    ) -> None:
        """
        Stream a token to all connections in a chat.
        Used for real-time streaming responses.
        
        NOTE: Bypasses Redis _publish_broadcast intentionally.
        Tokens are ephemeral â€” cross-node fan-out adds per-token overhead
        that directly increases visible streaming latency.
        Only the 'done' and 'info' signals use full broadcast_to_chat.
        """
        message = json.dumps({"type": "token", "content": token})
        await self._broadcast_local(chat_key, message)

    def set_stop_flag(self, chat_key: Tuple[str, str], value: bool = True) -> None:
        """Set stop generation flag for a chat"""
        self.stop_flags[chat_key] = value

    def should_stop(self, chat_key: Tuple[str, str]) -> bool:
        """Check if generation should stop for a chat"""
        return self.stop_flags.get(chat_key, False)

    def clear_stop_flag(self, chat_key: Tuple[str, str]) -> None:
        """Clear stop flag for a chat"""
        if chat_key in self.stop_flags:
            del self.stop_flags[chat_key]

    def get_connection_count(self, chat_id: str, user_id: Optional[str] = None) -> int:
        """Get number of active connections for a chat"""
        if isinstance(chat_id, tuple):
            chat_key = chat_id
        elif user_id:
            chat_key = self._chat_key(user_id, chat_id)
        else:
            return 0
        return len(self.active_connections.get(chat_key, {}))


# Global connection manager instance
manager = ConnectionManager()


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
    chat_mode = data.get("chat_mode", "normal")
    personality_id = data.get("personality_id")
    user_settings = data.get("user_settings")
    effective_settings = merge_settings(get_user_settings(user_id), user_settings)
    
    # Resolve personality if provided, else from session, else default to Relyce AI for normal mode
    personality = None
    conn_info = manager.connection_info.get(connection_id, {})
    user_id = conn_info.get("user_id")
    if user_id:
        from app.chat.personalities import get_personality_by_id
        if not personality_id:
            cached_id = conn_info.get("personality_id")
            if cached_id:
                personality_id = cached_id
            else:
                saved_id = get_session_personality_id(user_id, chat_id)
                if saved_id:
                    personality_id = saved_id
                    conn_info["personality_id"] = saved_id
        if personality_id:
            p_data = get_personality_by_id(user_id, personality_id)
            if p_data:
                personality = p_data
        if not personality and chat_mode == "normal":
            p_data = get_personality_by_id(user_id, "default_relyce")
            if p_data:
                personality = p_data

    if not content.strip():
        return

    if not check_rate_limit(user_id):
        await manager.send_personal_message(
            json.dumps({"type": "error", "content": f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min). Try again later."}),
            websocket
        )
        return

    # === AGENT MODE: Tiered Rate Limit + Concurrency Guard ===
    if chat_mode == "agent":
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
                        _plan = (_membership.get("plan") or "free").lower()
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
    
    # ðŸš€ Send immediate empty token to instantly transition UI from 'loading' to 'streaming' state
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
        context_messages = await asyncio.to_thread(get_context_for_llm, user_id, chat_id, resolved_personality_id)
    except Exception as e:
        print(f"[WS] Failed to load context: {e}")
        context_messages = []
    
    # Stream response to all connected devices
    full_response = ""
    stream_generator = None
    
    # ðŸ”§ BACKPRESSURE: Bounded queue between LLM producer and WebSocket consumer
    # If client is slow, queue fills up and naturally throttles the model stream
    BACKPRESSURE_QUEUE_SIZE = 128
    token_queue = asyncio.Queue(maxsize=BACKPRESSURE_QUEUE_SIZE)
    _SENTINEL = object()  # Marks end of stream
    
    retrieved_context_meta = {}
    
    async def _producer():
        """Reads from LLM generator and enqueues tokens."""
        nonlocal stream_generator
        try:
            stream_generator = llm_processor.process_message_stream(
                content, 
                mode=chat_mode,
                context_messages=context_messages,
                personality=personality,
                user_settings=effective_settings,
                user_id=memory_user_id,
                session_id=chat_id
            )
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
                
                # SSE sequence counter
                _event_seq_counters[chat_key] = _event_seq_counters.get(chat_key, 0) + 1
                seq = _event_seq_counters[chat_key]
                
                await manager.broadcast_to_chat(
                    chat_key,
                    json.dumps({"type": "info", "content": clean_info, "_event_seq": seq})
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
            
            full_response += token
            import time as _t
            _t_recv = _t.time()
            await manager.stream_to_chat(chat_key, token)
            _t_sent = _t.time()
            # Removed per-token print to prevent event loop blocking on Windows terminal
        
        # Send completion signal
        await manager.broadcast_to_chat(
            chat_key,
            json.dumps({"type": "done", "content": ""})
        )
        
        # Update context with this exchange
        update_context_with_exchange(user_id, chat_id, content, full_response, resolved_personality_id)
        
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
                fresh_context_msgs = get_context_for_llm(user_id, chat_id, resolved_personality_id)
                await summarize_if_needed(user_id, chat_id, fresh_context_msgs, get_openrouter_client())
            except Exception as e:
                print(f"[WS] Background history save failed: {e}")

        asyncio.create_task(save_history_background())

        # === Memory Extraction (background, rate-limited, 3s timeout) ===
        # Skip if response too short (~40 tokens â‰ˆ 200 chars) â€” trivial replies produce junk
        if user_id and user_id != "anonymous" and full_response and len(full_response) > 120:
            async def _extract_vector_memory():
                async with _EXTRACTION_SEMAPHORE:
                    try:
                        from app.memory.vector_memory import extract_and_store_memories
                        stored = await asyncio.wait_for(
                            extract_and_store_memories(user_id, content, full_response),
                            timeout=3.0
                        )
                        if stored:
                            print(f"[WS] Vector memory: {stored} facts extracted for {user_id}")
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
                            print(f"[WS] Knowledge graph: {stored} triples extracted for {user_id}")
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
        print(f"[WS] Stream cancelled by client disconnect for {chat_key}")
        if stream_generator:
            try:
                await stream_generator.aclose()
            except Exception:
                pass
        raise  # Re-raise to let the router handle the disconnect
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        print(f"[WS] {error_msg}")
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
