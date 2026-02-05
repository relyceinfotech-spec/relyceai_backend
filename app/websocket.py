"""
Relyce AI - WebSocket Manager
Multi-device safe WebSocket implementation
"""
import json
import asyncio
from typing import Dict, List, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
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

class ConnectionManager:
    """
    Multi-device safe connection manager.
    
    Key Design (from architecture):
    - One user → Multiple devices → Multiple WebSocket connections → SAME chat_id
    - Connections are organized by chat_id, NOT by user_id
    - This allows multiple devices to connect to the same chat and receive synced responses
    """
    
    def __init__(self):
        # Structure: {chat_id: {connection_id: WebSocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # Track user info per connection: {connection_id: {"user_id": str, "chat_id": str}}
        self.connection_info: Dict[str, Dict] = {}
        # Stop flags for generation: {chat_id: bool}
        self.stop_flags: Dict[str, bool] = {}
        # Connection counter for unique IDs
        self._counter = 0
    
    def _generate_connection_id(self) -> str:
        """Generate unique connection ID"""
        self._counter += 1
        return f"conn_{self._counter}_{datetime.now().timestamp()}"
    
    async def connect(
        self, 
        websocket: WebSocket, 
        chat_id: str, 
        user_id: str
    ) -> str:
        """
        Accept WebSocket connection.
        Multiple connections allowed per chat_id (multi-device support).
        
        Returns: connection_id
        """
        await websocket.accept()
        
        connection_id = self._generate_connection_id()
        
        # Initialize chat_id bucket if not exists
        if chat_id not in self.active_connections:
            self.active_connections[chat_id] = {}
        
        # Add connection to chat_id bucket
        self.active_connections[chat_id][connection_id] = websocket
        
        # Track connection info
        self.connection_info[connection_id] = {
            "user_id": user_id,
            "chat_id": chat_id,
            "connected_at": datetime.now().isoformat()
        }
        
        print(f"[WS] ✅ Connected: {connection_id} to chat {chat_id}")
        print(f"[WS] Active connections for chat {chat_id}: {len(self.active_connections[chat_id])}")
        
        return connection_id
    
    def disconnect(self, connection_id: str) -> None:
        """
        Remove a connection gracefully.
        Other connections to same chat_id remain active.
        """
        if connection_id not in self.connection_info:
            return
        
        info = self.connection_info[connection_id]
        chat_id = info["chat_id"]
        
        # Remove from active connections
        if chat_id in self.active_connections:
            if connection_id in self.active_connections[chat_id]:
                del self.active_connections[chat_id][connection_id]
            
            # Clean up empty chat_id bucket
            if not self.active_connections[chat_id]:
                del self.active_connections[chat_id]
                # Also clean up stop flag
                if chat_id in self.stop_flags:
                    del self.stop_flags[chat_id]
        
        # Remove connection info
        del self.connection_info[connection_id]
        
        print(f"[WS] ❌ Disconnected: {connection_id} from chat {chat_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """Send message to a specific connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"[WS] Error sending message: {e}")
    
    async def broadcast_to_chat(
        self, 
        chat_id: str, 
        message: str,
        exclude_connection: Optional[str] = None
    ) -> None:
        """
        Broadcast message to ALL connections in a chat_id.
        This enables multi-device sync - all devices see the same response.
        """
        if chat_id not in self.active_connections:
            return
        
        disconnected = []
        
        for conn_id, websocket in self.active_connections[chat_id].items():
            if conn_id == exclude_connection:
                continue
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(conn_id)
        
        # Clean up disconnected
        for conn_id in disconnected:
            self.disconnect(conn_id)
    
    async def stream_to_chat(
        self, 
        chat_id: str, 
        token: str
    ) -> None:
        """
        Stream a token to all connections in a chat.
        Used for real-time streaming responses.
        """
        message = json.dumps({"type": "token", "content": token})
        await self.broadcast_to_chat(chat_id, message)
    
    def set_stop_flag(self, chat_id: str, value: bool = True) -> None:
        """Set stop generation flag for a chat"""
        self.stop_flags[chat_id] = value
    
    def should_stop(self, chat_id: str) -> bool:
        """Check if generation should stop for a chat"""
        return self.stop_flags.get(chat_id, False)
    
    def clear_stop_flag(self, chat_id: str) -> None:
        """Clear stop flag for a chat"""
        if chat_id in self.stop_flags:
            del self.stop_flags[chat_id]
    
    def get_connection_count(self, chat_id: str) -> int:
        """Get number of active connections for a chat"""
        return len(self.active_connections.get(chat_id, {}))


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
    
    if msg_type == "stop":
        # Stop generation
        manager.set_stop_flag(chat_id)
        await manager.broadcast_to_chat(
            chat_id, 
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
    
    # Clear any previous stop flag
    manager.clear_stop_flag(chat_id)
    
    # Notify all devices - processing started (IMMEDIATE FEEDBACK)
    await manager.broadcast_to_chat(
        chat_id,
        json.dumps({"type": "info", "content": "processing"})
    )
    
    # Extract and store user facts (Run in background to not block)
    if user_id and user_id != "anonymous":
        try:
            from app.chat.memory import process_and_store_facts
            from starlette.concurrency import run_in_threadpool
            # Fire and forget fact extraction
            asyncio.create_task(run_in_threadpool(process_and_store_facts, user_id, content))
        except Exception as e:
            print(f"[WS] Fact extraction error (non-blocking): {e}")

    # Safety Check: Max Message Length
    if len(content) > 6000:
        await manager.send_personal_message(
            json.dumps({"type": "error", "content": "Message too long (max 6000 chars)"}), 
            websocket
        )
        return
    
    # Get context for LLM
    context_messages = get_context_for_llm(user_id, chat_id)
    
    # Stream response to all connected devices
    full_response = ""
    
    try:
        async for token in llm_processor.process_message_stream(
            content, 
            mode=chat_mode,
            context_messages=context_messages,
            personality=personality,
            user_settings=effective_settings,
            user_id=user_id
        ):
            # Check stop flag
            if manager.should_stop(chat_id):
                await manager.broadcast_to_chat(
                    chat_id,
                    json.dumps({"type": "info", "content": "stopped"})
                )
                break
            
            # Intercept [INFO] messages (like search status)
            if token.strip().startswith("[INFO]"):
                clean_info = token.replace("[INFO]", "").strip()
                await manager.broadcast_to_chat(
                    chat_id,
                    json.dumps({"type": "info", "content": clean_info})
                )
                continue # Do not append to full_response or stream as token
            
            full_response += token
            await manager.stream_to_chat(chat_id, token)
        
        # Send completion signal
        await manager.broadcast_to_chat(
            chat_id,
            json.dumps({"type": "done", "content": ""})
        )
        
        # Update context with this exchange
        update_context_with_exchange(user_id, chat_id, content, full_response)
        
        # Save to Firebase (async, don't block)
        save_message_to_firebase(user_id, chat_id, "user", content)
        save_message_to_firebase(user_id, chat_id, "assistant", full_response)
        
        # =========================================================================
        # Post-Processing: Context Summarization (Deferred to improve latency)
        # =========================================================================
        msg_count = get_raw_message_count(user_id, chat_id)
        if chat_mode == "normal" and msg_count >= SUMMARY_TRIGGER_MESSAGES:
            try:
                print(f"[WS] Triggering background summarization for {chat_id}")
                # 1. Get messages to summarize (everything except last N)
                # We access via get_context_for_llm which returns [System, ...Raw]
                current_context = get_context_for_llm(user_id, chat_id)
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
            chat_id,
            json.dumps({"type": "error", "content": error_msg})
        )
