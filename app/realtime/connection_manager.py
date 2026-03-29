"""
WebSocket connection manager.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple

from fastapi import WebSocket
from starlette.websockets import WebSocketState

try:
    import redis.asyncio as redis
except Exception:
    redis = None

from app.config import (
    MAX_WS_CONNECTIONS_PER_CHAT,
    MAX_WS_CONNECTIONS_TOTAL,
    REDIS_URL,
    REDIS_WS_CHANNEL_PREFIX,
)


def _mask_identifier(value: str) -> str:
    text = str(value or "")
    if len(text) <= 6:
        return "***"
    return f"{text[:3]}***{text[-3:]}"


class ConnectionManager:
    """
    Multi-device safe connection manager.

    Key Design (from architecture):
    - One user -> Multiple devices -> Multiple WebSocket connections -> SAME chat_id
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

        print(f"[WS] Connected: conn={_mask_identifier(connection_id)} chat={_mask_identifier(chat_id)} user={_mask_identifier(user_id)}")
        print(f"[WS] Active connections chat={_mask_identifier(chat_id)} count={len(self.active_connections[chat_key])}")

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

        print(f"[WS] Disconnected: conn={_mask_identifier(connection_id)} chat={_mask_identifier(chat_id)}")

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
        Tokens are ephemeral - cross-node fan-out adds per-token overhead
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






