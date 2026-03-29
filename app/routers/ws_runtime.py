"""Runtime WebSocket router."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from app.auth import verify_token
from app.chat.runtime_helpers import resolve_memory_user_id
from app.llm.emotion_engine import emotion_engine
from app.realtime.websocket_runtime import handle_websocket_message, manager

router = APIRouter()


def _safe_err_name(err: Exception) -> str:
    return err.__class__.__name__


@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket, chat_id: Optional[str] = Query(None)):
    """WebSocket chat endpoint with multi-device support."""
    await websocket.accept()

    async def safe_send(payload: dict) -> bool:
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            await websocket.send_text(json.dumps(payload))
            return True
        except Exception:
            return False

    token = None
    resolved_chat_id = chat_id

    try:
        auth_raw = await asyncio.wait_for(websocket.receive_text(), timeout=6)
        auth_msg = json.loads(auth_raw)
        if auth_msg.get("type") == "auth":
            token = auth_msg.get("token")
            if not resolved_chat_id:
                resolved_chat_id = auth_msg.get("chat_id")
        else:
            await safe_send({"type": "error", "content": "Unauthorized: Missing auth"})
            await websocket.close(code=1008)
            return
    except asyncio.TimeoutError:
        await safe_send({"type": "error", "content": "Unauthorized: Auth timeout"})
        await websocket.close(code=1008)
        return
    except Exception:
        await safe_send({"type": "error", "content": "Unauthorized: Invalid auth"})
        await websocket.close(code=1008)
        return

    if token:
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token or len(token) < 10:
            await safe_send({"type": "error", "content": "Invalid token format"})
            await websocket.close(code=1008)
            return

    if not token:
        await safe_send({"type": "error", "content": "Unauthorized: Missing token"})
        await websocket.close(code=1008)
        return

    try:
        is_valid, user_info = verify_token(token)
    except Exception as e:
        print(f"[WS] Auth error type={_safe_err_name(e)}")
        is_valid, user_info = False, None

    if not is_valid or not user_info:
        await safe_send({"type": "error", "content": "Unauthorized: Invalid token"})
        await websocket.close(code=1008)
        return

    user_id = user_info.get("uid")
    if not user_id:
        await safe_send({"type": "error", "content": "Unauthorized: Invalid user"})
        await websocket.close(code=1008)
        return

    if not resolved_chat_id:
        resolved_chat_id = f"chat_{datetime.now().timestamp()}"

    try:
        connection_id = await manager.connect(websocket, resolved_chat_id, user_id)
    except Exception as e:
        print(f"[WS] Connect failure type={_safe_err_name(e)}")
        await safe_send({"type": "error", "content": "Connection rejected"})
        await websocket.close(code=1013)
        return

    unique_user_id = resolve_memory_user_id(user_id)
    if connection_id in manager.connection_info:
        manager.connection_info[connection_id]["unique_user_id"] = unique_user_id

    await safe_send({"type": "auth_ok"})

    try:
        await emotion_engine.prewarm(resolved_chat_id)
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, connection_id, message, manager)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "content": "Invalid JSON"}),
                    websocket,
                )
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        print(f"[WS] Runtime error type={_safe_err_name(e)}")
        await manager.disconnect(connection_id)




