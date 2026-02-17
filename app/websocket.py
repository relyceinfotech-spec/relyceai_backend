"""
Relyce AI - WebSocket Manager
Multi-device safe WebSocket implementation
"""
import json
import asyncio
import uuid

try:
    import redis.asyncio as redis
except Exception:  # optional dependency
    redis = None
from typing import Dict, List, Set, Optional, Tuple
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from datetime import datetime
from app.auth import verify_token
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
from app.rate_limit import check_rate_limit
from app.config import MAX_CHAT_MESSAGE_CHARS, RATE_LIMIT_PER_MINUTE, MAX_WS_CONNECTIONS_PER_CHAT, MAX_WS_CONNECTIONS_TOTAL, REDIS_URL, REDIS_WS_CHANNEL_PREFIX

class ConnectionManager:
    """
    Multi-device safe connection manager.

    Key Design (from architecture):
    - One user → Multiple devices → Multiple WebSocket connections → SAME chat_id
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

        print(f"[WS] ✅ Connected: {connection_id} to chat {chat_id} (user={user_id})")
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

        print(f"[WS] ❌ Disconnected: {connection_id} from chat {chat_id}")

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
        if chat_key not in self.active_connections:
            return

        async with self._lock:
            connections = list(self.active_connections.get(chat_key, {}).items())

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
        """
        message = json.dumps({"type": "token", "content": token})
        await self.broadcast_to_chat(chat_key, message)

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
    
    # Clear any previous stop flag
    manager.clear_stop_flag(chat_key)
    
    # Notify all devices - processing started (IMMEDIATE FEEDBACK)
    await manager.broadcast_to_chat(
        chat_key,
        json.dumps({"type": "info", "content": "processing"})
    )
    
    # Extract and store user facts (Run in background to not block)
    if user_id and user_id != "anonymous":
        try:
            from app.chat.memory import process_and_store_facts
            # Fire and forget fact extraction
            asyncio.create_task(asyncio.to_thread(process_and_store_facts, user_id, content))
        except Exception as e:
            print(f"[WS] Fact extraction error (non-blocking): {e}")

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
    
    try:
        async for token in llm_processor.process_message_stream(
            content, 
            mode=chat_mode,
            context_messages=context_messages,
            personality=personality,
            user_settings=effective_settings,

            user_id=user_id,
            session_id=chat_id
        ):
            # Check stop flag
            if manager.should_stop(chat_key):
                await manager.broadcast_to_chat(
                    chat_key,
                    json.dumps({"type": "info", "content": "stopped"})
                )
                break
            
            # Intercept [INFO] messages (like search status)
            if token.strip().startswith("[INFO]"):
                clean_info = token.replace("[INFO]", "").strip()
                await manager.broadcast_to_chat(
                    chat_key,
                    json.dumps({"type": "info", "content": clean_info})
                )
                continue # Do not append to full_response or stream as token
            
            full_response += token
            await manager.stream_to_chat(chat_key, token)
        
        # Send completion signal
        await manager.broadcast_to_chat(
            chat_key,
            json.dumps({"type": "done", "content": ""})
        )
        
        # Update context with this exchange
        update_context_with_exchange(user_id, chat_id, content, full_response, resolved_personality_id)
        
        # Save to Firebase (background, non-blocking)
        async def save_history_background():
            try:
                await asyncio.to_thread(save_message_to_firebase, user_id, chat_id, "user", content, resolved_personality_id)
                await asyncio.to_thread(save_message_to_firebase, user_id, chat_id, "assistant", full_response, resolved_personality_id)
            except Exception as e:
                print(f"[WS] Background history save failed: {e}")

        asyncio.create_task(save_history_background())

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
        
        # =========================================================================
        # Post-Processing: Context Summarization (Deferred to improve latency)
        # =========================================================================
        msg_count = get_raw_message_count(user_id, chat_id)
        if chat_mode == "normal" and msg_count >= SUMMARY_TRIGGER_MESSAGES:
            try:
                print(f"[WS] Triggering background summarization for {chat_id}")
                # 1. Get messages to summarize (everything except last N)
                # We access via get_context_for_llm which returns [System, ...Raw]
                current_context = get_context_for_llm(user_id, chat_id, resolved_personality_id)
                raw_msgs = [m for m in current_context if m['role'] != 'system']
                
                # 2. Split
                to_summarize = raw_msgs[:-KEEP_LAST_MESSAGES] # Older messages to compact
                
                # 3. Generate summary (cumulative)
                if to_summarize:
                    existing_summary = get_session_summary(user_id, chat_id) or ""
                    
                    # This returns the NEW FULL summary (merged)
                    new_cumulative_summary = await llm_processor.summarize_context(to_summarize, existing_summary)
                    
                    # 4. Overwrite Store (The new summary contains everything)
                    if new_cumulative_summary:
                       update_session_summary(user_id, chat_id, new_cumulative_summary)
                       prune_context_messages(user_id, chat_id, KEEP_LAST_MESSAGES)
                       print(f"[WS] Summarization complete for {chat_id}")
                
            except Exception as e:
                print(f"[Context Error] Failed to summarize: {e}")
        
    except Exception as e:
        error_msg = f"Error processing message: {str(e)}"
        print(f"[WS] {error_msg}")
        await manager.broadcast_to_chat(
            chat_key,
            json.dumps({"type": "error", "content": error_msg})
        )
