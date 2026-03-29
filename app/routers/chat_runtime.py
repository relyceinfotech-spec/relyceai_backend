"""Runtime chat endpoints router."""
from __future__ import annotations

import asyncio
import traceback
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from app.auth import get_current_user
from app.config import MAX_CHAT_MESSAGE_CHARS, IS_PRODUCTION
from app.context.response_contract import normalize_chat_response, build_final_answer_payload
from app.governance.abuse_detector import get_abuse_detector
from app.governance.usage_store import get_usage_store
from app.llm.processor import active_executions
from app.platform import get_ai_platform, get_task_queue, CapabilityRequest
from app.models import ChatRequest, ChatResponse, SearchRequest
from app.chat.context import get_context_for_llm
from app.chat.user_profile import get_user_settings, merge_settings
from app.chat.history import load_chat_history
from app.chat.runtime_helpers import (
    persist_exchange_and_schedule_summary,
    resolve_memory_user_id,
    resolve_personality_context,
)
from app.chat.mode_mapper import normalize_chat_mode
from app.safety.stream_moderator import StreamingOutputModerator
from app.streaming.stream_controller import StreamController
from app.routers.system import governance_gate, circuit_breaker_gate, parse_gate_json_response

router = APIRouter()
ai_platform = get_ai_platform()
task_queue = get_task_queue()


class ConfirmRequest(BaseModel):
    confirm: bool

@router.post("/agent/cancel/{execution_id}")
async def cancel_agent_execution(execution_id: str, user_info: dict = Depends(get_current_user)):
    """Cancel a running agent execution"""
    if execution_id in active_executions:
        exec_data = active_executions[execution_id]
        user_id = user_info["uid"]
        memory_user_id = resolve_memory_user_id(user_id)
        owner_id = str(exec_data.get("user_id") or "")
        if owner_id not in {user_id, memory_user_id}:
            raise HTTPException(status_code=403, detail="Forbidden")
        ctx = exec_data["ctx"]
        ctx.terminate = True
        if hasattr(ctx, "pause_event") and ctx.pause_event:
            ctx.pause_event.set()
        return {"success": True, "message": "Execution cancelled"}
    raise HTTPException(status_code=404, detail="Execution not found or already completed")

@router.post("/agent/confirm/{execution_id}")
async def confirm_agent_tool(execution_id: str, request: ConfirmRequest, user_info: dict = Depends(get_current_user)):
    """Confirm or deny an awaiting tool action"""
    if execution_id in active_executions:
        exec_data = active_executions[execution_id]
        user_id = user_info["uid"]
        memory_user_id = resolve_memory_user_id(user_id)
        owner_id = str(exec_data.get("user_id") or "")
        if owner_id not in {user_id, memory_user_id}:
            raise HTTPException(status_code=403, detail="Forbidden")
        ctx = exec_data["ctx"]
        ctx.confirmation = request.confirm
        if hasattr(ctx, "pause_event") and ctx.pause_event:
            ctx.pause_event.set()
        return {"success": True, "message": f"Action {'confirmed' if request.confirm else 'denied'}"}
    raise HTTPException(status_code=404, detail="Execution not found or already completed")

# ============================================
# PHASE 4C: WORKFLOW STATE ENDPOINTS
# ============================================

class ResumeRequest(BaseModel):
    session_id: str
    message: str = "Continue task"


def _user_owns_task(task, user_id: str, memory_user_id: str) -> bool:
    owner_id = getattr(task.request, "user_id", "")
    return owner_id in {user_id, memory_user_id}

@router.post("/agent/resume/{task_id}")
async def resume_task(task_id: str, request: ResumeRequest, req: Request, user_info: dict = Depends(get_current_user)):
    """
    Resume a PAUSED or RUNNING multi-step agent task.
    """
    from app.state.task_state_engine import get_task_state, set_task_status, summarize_task_history
    from app.agent.hybrid_controller import generate_strategy_advice

    user_id = user_info["uid"]
    session_id = request.session_id
    
    # Enforce all platform safety and governance gates (Abuse, Quota, Breaker)
    await check_platform_gates(req, user_id)

    # 1. Fetch State
    state = get_task_state(session_id, task_id)
    if not state:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    
    if state.status == "FAILED":
        raise HTTPException(status_code=400, detail="Cannot resume a FAILED task without explicit retry reset")
    
    # 2. Re-Validate Strategy (Constraint 2)
    # We do NOT trust state.planning_mode blindly. We recalculate intent.
    # We pass the user's latest "continue" prompt or explicit instruction.
    strategy = generate_strategy_advice(request.message, {}, session_id)
    
    # 3. Summarize History (Constraint 1)
    # Prevent context window explosion by digesting step_results
    history_summary = summarize_task_history(session_id, task_id)
    
    # 4. Inject resume context securely into user prompt
    augmented_message = request.message
    
    resume_graph = None
    if getattr(state, "plan_graph_snapshot", None):
        from app.state.plan_graph import PlanGraph
        resume_graph = PlanGraph.deserialize(state.plan_graph_snapshot)
        augmented_message += "\n\n(Resuming background graph execution...)"
    else:
        augmented_message = (
            f"{request.message}\n\n"
            f"--- RESUMING TASK {task_id} ---\n"
            f"You are resuming a multi-step execution. Do NOT restart from the beginning.\n"
            f"Pick up where you left off based on your history:\n\n{history_summary}"
        )

    set_task_status(session_id, task_id, "RUNNING")

    # 5. Execute Streaming processor (identical to chat_stream but with task_id passed through args via processor kwargs)
    async def generate():
        store = get_usage_store()
        store.increment_concurrent(user_id)
        stream_controller = StreamController(min_info_interval_s=0.2)
        full_response = ""
        sent_final = False
        message_id = task_id
        try:
            yield ": " + (" " * 1024) + "\n\n"
            yield f"data: {json.dumps({'type': 'message_start', 'message_id': message_id, 'mode': 'agent_resume', 'stream_schema_version': 2, 'timestamp': time.time()})}\n\n"
            
            # Fetch smart context
            context_messages = get_context_for_llm(user_id, session_id, None, "smart")
            effective_settings = get_user_settings(user_id)
            
            # Fire LLM (passing resume_graph into kwargs)
            stream_moderator = StreamingOutputModerator(query=request.message)
            req_obj = CapabilityRequest(
                user_query=augmented_message,
                chat_mode="smart",
                context_messages=context_messages,
                personality=None,
                user_settings=effective_settings,
                user_id=user_id,
                session_id=session_id,
                task_id=task_id,
                resume_graph=resume_graph,
            )
            async for token in ai_platform.run_stream(req_obj):
                if token.strip().startswith("[INFO]"):
                    clean_info = token.replace("[INFO]", "").strip()
                    filtered = stream_controller.filter_info(clean_info)
                    if filtered:
                        # Legacy info compatibility shim -> canonical progress
                        try:
                            payload = json.loads(filtered) if filtered.startswith("{") else {"label": str(filtered)}
                        except Exception:
                            payload = {"label": str(filtered)}
                        event_name = str(payload.get("event") or "progress")
                        if event_name == "final_answer":
                            answer_text = str(payload.get("answer") or "").strip()
                            if answer_text:
                                full_response = answer_text
                            continue
                        yield f"data: {json.dumps({'type': 'progress', 'message_id': message_id, 'state': event_name, 'label': str(payload.get('label') or payload.get('message') or event_name), 'percent': payload.get('percent'), 'timestamp': time.time()})}\n\n"
                    continue
                safe_chunk = stream_moderator.ingest(token)
                if safe_chunk:
                    full_response += safe_chunk
                    yield f"data: {json.dumps({'type': 'token', 'message_id': message_id, 'text': safe_chunk, 'timestamp': time.time()})}\n\n"
                
            tail_chunk = stream_moderator.finalize()
            if tail_chunk:
                full_response += tail_chunk
                yield f"data: {json.dumps({'type': 'token', 'message_id': message_id, 'text': tail_chunk, 'timestamp': time.time()})}\n\n"

            final_payload = build_final_answer_payload(
                    {"success": True, "response": full_response},
                    user_query=request.message,
                    chat_mode="smart",
                )
            yield f"data: {json.dumps({'type': 'final', 'message_id': message_id, 'answer': str(final_payload.get('answer') or ''), 'summary': final_payload.get('summary'), 'metadata': final_payload.get('metadata', {}), 'timestamp': time.time()})}\n\n"
            sent_final = True
            yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'timestamp': time.time()})}\n\n"

        except Exception as e:
            set_task_status(session_id, task_id, "PAUSED")
            print(f"[Chat/Resume] Error type={e.__class__.__name__}")
            if not sent_final:
                yield f"data: {json.dumps({'type': 'error', 'message_id': message_id, 'code': 'resume_error', 'message': 'Internal server error', 'timestamp': time.time()})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message_id': message_id, 'timestamp': time.time()})}\n\n"
        finally:
            store.decrement_concurrent(user_id)
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )



async def check_platform_gates(req: Request, user_id: str, tier: str = "free"):
    """
    Enforces the platform's execution chain:
    1. Abuse Detector
    2. Governance (Rate -> Quota -> Spend)
    3. Circuit Breaker
    """
    # 1. Abuse Detector (Logs and flags only for now)
    ip = req.client.host if req and req.client else "unknown"
    get_abuse_detector().evaluate_request(ip, user_id=user_id)

    # 2. Governance Gate
    gov_response = await governance_gate(user_id, tier)
    if gov_response:
        payload = parse_gate_json_response(gov_response)
        raise HTTPException(status_code=gov_response.status_code, detail=payload)

    # 3. Circuit Breaker Gate
    cb_response = await circuit_breaker_gate()
    if cb_response:
        payload = parse_gate_json_response(cb_response)
        raise HTTPException(status_code=cb_response.status_code, detail=payload)


def _task_public_state(task):
    return {
        "task_id": task.task_id,
        "lane": getattr(task, "lane", "fast"),
        "status": task.status,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "error": task.error,
        "result": task.result,
    }


@router.post("/chat/submit")
async def chat_submit(request: ChatRequest, req: Request, user_info: dict = Depends(get_current_user)):
    """Submit chat work to background queue and return task id immediately."""
    if len(request.message or "") > MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="Message too long")

    user_id = user_info["uid"]
    # Guard against long-running sync dependencies (Firestore/network stalls)
    # so submit never hangs indefinitely on the client.
    try:
        memory_user_id = await asyncio.wait_for(
            asyncio.to_thread(resolve_memory_user_id, user_id),
            timeout=4.0,
        )
    except Exception as e:
        print(f"[chat_submit] memory_user_id fallback for {user_id}: {e}")
        memory_user_id = user_id

    try:
        await asyncio.wait_for(check_platform_gates(req, user_id), timeout=6.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Request gating timed out. Please retry.")

    personality, personality_id = resolve_personality_context(
        user_id=user_id,
        session_id=request.session_id,
        chat_mode=request.chat_mode,
        personality=request.personality,
        personality_id=request.personality_id,
    )

    context_messages = []
    if request.session_id:
        try:
            context_messages = await asyncio.wait_for(
                asyncio.to_thread(
                    get_context_for_llm,
                    user_id,
                    request.session_id,
                    personality_id,
                    request.chat_mode,
                ),
                timeout=6.0,
            )
        except Exception as e:
            print(f"[chat_submit] context load fallback for {request.session_id}: {e}")
            context_messages = []

    effective_settings = merge_settings(get_user_settings(user_id), request.user_settings)
    req_obj = CapabilityRequest(
        user_query=request.message,
        chat_mode=request.chat_mode,
        context_messages=context_messages,
        personality=personality,
        user_settings=effective_settings,
        user_id=memory_user_id,
        session_id=request.session_id,
        file_ids=list(request.file_ids or []),
        metadata={
            "file_ids": list(request.file_ids or []),
            "personality_id": personality_id,
            "stream_schema_version": 2,
            "auth_uid": user_id,
        },
    )
    try:
        task_id = await task_queue.submit(req_obj)
    except RuntimeError as e:
        if "HEAVY_LANE_LIMIT_REACHED" in str(e):
            raise HTTPException(status_code=429, detail="Too many heavy tasks in progress for this user")
        if "HEAVY_LANE_SESSION_LIMIT_REACHED" in str(e):
            raise HTTPException(status_code=429, detail="Too many heavy tasks in progress for this session")
        raise

    task = await task_queue.get_task(task_id)
    lane = getattr(task, "lane", "fast") if task else "fast"
    depth = await task_queue.queue_depth()
    lane_depth = await task_queue.queue_depth(lane)
    return {
        "success": True,
        "task_id": task_id,
        "status": "queued",
        "lane": lane,
        "queue_depth": depth,
        "lane_queue_depth": lane_depth,
        "stream_schema_version": 2,
    }


@router.get("/chat/tasks/{task_id}")
async def chat_task_status(task_id: str, user_info: dict = Depends(get_current_user)):
    user_id = user_info["uid"]
    memory_user_id = resolve_memory_user_id(user_id)

    task = await task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not _user_owns_task(task, user_id, memory_user_id):
        raise HTTPException(status_code=403, detail="Forbidden")

    return {"success": True, **_task_public_state(task)}


@router.get("/chat/tasks/{task_id}/events")
async def chat_task_events(task_id: str, after_seq: int = 0, user_info: dict = Depends(get_current_user)):
    user_id = user_info["uid"]
    memory_user_id = resolve_memory_user_id(user_id)

    task = await task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if not _user_owns_task(task, user_id, memory_user_id):
        raise HTTPException(status_code=403, detail="Forbidden")

    events = await task_queue.get_events(task_id, after_seq=after_seq)
    latest_seq = events[-1]["seq"] if events else after_seq
    done = task.status in {"completed", "failed", "cancelled"}
    return {
        "success": True,
        "task_id": task_id,
        "status": task.status,
        "events": events,
        "latest_seq": latest_seq,
        "done": done,
        "result": task.result if done else None,
        "error": task.error if done else None,
    }


@router.get("/chat/tasks/{task_id}/stream")
async def chat_task_stream(task_id: str, user_info: dict = Depends(get_current_user)):
    """SSE stream for queued task updates."""
    user_id = user_info["uid"]
    memory_user_id = resolve_memory_user_id(user_id)

    async def event_gen():
        last_seq = 0
        sent_done = False
        sent_final = False
        last_token_seq = 0

        while True:
            task = await task_queue.get_task(task_id)
            if not task:
                now = time.time()
                yield f"data: {json.dumps({'type': 'error', 'code': 'not_found', 'message': 'Task not found', 'timestamp': now})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message_id': task_id, 'timestamp': now})}\n\n"
                break

            if not _user_owns_task(task, user_id, memory_user_id):
                now = time.time()
                yield f"data: {json.dumps({'type': 'error', 'code': 'forbidden', 'message': 'Forbidden', 'timestamp': now})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message_id': task_id, 'timestamp': now})}\n\n"
                break

            events = await task_queue.get_events(task_id, after_seq=last_seq)
            for event in events:
                last_seq = max(last_seq, int(event.get("seq", last_seq)))
                ev_type = str(event.get("event") or "progress")
                payload = {k: v for k, v in event.items() if k != "event"}
                if sent_final and ev_type != "done":
                    continue
                if ev_type == "token":
                    payload["seq"] = int(payload.pop("stream_seq", payload.pop("token_seq", 0)) or 0)
                    current_token_seq = int(payload.get("seq") or 0)
                    if current_token_seq <= last_token_seq:
                        continue
                    last_token_seq = current_token_seq
                payload["type"] = ev_type
                if "message_id" not in payload:
                    payload["message_id"] = task_id
                yield f"data: {json.dumps(payload)}\n\n"
                if ev_type == "final":
                    sent_final = True
                if ev_type == "done":
                    sent_done = True

            if sent_done:
                break

            if task.status in {"failed", "cancelled", "completed"} and not events:
                # Safety net for legacy/incomplete traces: ensure deterministic stream termination.
                now = time.time()
                if task.status == "completed":
                    if not sent_final:
                        payload = task.result or {"answer": "", "metadata": {"mode": normalize_chat_mode(str(task.request.chat_mode or "smart"))}}
                        yield f"data: {json.dumps({'type': 'final', 'message_id': task_id, 'answer': payload.get('answer', ''), 'summary': payload.get('summary'), 'metadata': payload.get('metadata', {}), 'timestamp': now})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message_id': task_id, 'code': task.status, 'message': task.error or 'Task failed', 'timestamp': now})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message_id': task_id, 'timestamp': now})}\n\n"
                break

            await asyncio.sleep(0.25)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
        },
    )


@router.post("/chat/tasks/{task_id}/cancel")
async def cancel_chat_task(task_id: str, user_info: dict = Depends(get_current_user)):
    user_id = user_info["uid"]
    memory_user_id = resolve_memory_user_id(user_id)
    task = await task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not _user_owns_task(task, user_id, memory_user_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    updated = await task_queue.cancel_task(task_id)
    return {
        "success": True,
        "task_id": task_id,
        "status": getattr(updated, "status", "cancelled"),
    }


@router.post("/chat/tasks/{task_id}/confirm")
async def confirm_chat_task(task_id: str, request: ConfirmRequest, user_info: dict = Depends(get_current_user)):
    user_id = user_info["uid"]
    memory_user_id = resolve_memory_user_id(user_id)
    task = await task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not _user_owns_task(task, user_id, memory_user_id):
        raise HTTPException(status_code=403, detail="Forbidden")
    updated = await task_queue.confirm_task(task_id, request.confirm)
    return {
        "success": True,
        "task_id": task_id,
        "status": getattr(updated, "status", "queued"),
        "confirmed": bool(request.confirm),
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request, user_info: dict = Depends(get_current_user)):
    """Non-streaming chat endpoint."""
    try:
        if len(request.message or "") > MAX_CHAT_MESSAGE_CHARS:
            raise HTTPException(status_code=413, detail="Message too long")

        user_id = user_info["uid"]
        memory_user_id = resolve_memory_user_id(user_id)

        await check_platform_gates(req, user_id)
        request.user_id = user_id

        personality, personality_id = resolve_personality_context(
            user_id=user_id,
            session_id=request.session_id,
            chat_mode=request.chat_mode,
            personality=request.personality,
            personality_id=request.personality_id,
        )

        context_messages = []
        if request.session_id:
            context_messages = get_context_for_llm(user_id, request.session_id, personality_id, request.chat_mode)

        store = get_usage_store()
        store.increment_concurrent(user_id)
        try:
            effective_settings = merge_settings(get_user_settings(user_id), request.user_settings)
            req_obj = CapabilityRequest(
                user_query=request.message,
                chat_mode=request.chat_mode,
                context_messages=context_messages,
                personality=personality,
                user_settings=effective_settings,
                user_id=memory_user_id,
                session_id=request.session_id,
                file_ids=list(request.file_ids or []),
                metadata={"file_ids": list(request.file_ids or [])},
            )
            result = (await ai_platform.run(req_obj)).payload
        finally:
            store.decrement_concurrent(user_id)

        if request.session_id:
            msg_id = persist_exchange_and_schedule_summary(
                user_id=user_id,
                session_id=request.session_id,
                user_text=request.message,
                assistant_text=result["response"],
                personality_id=personality_id,
                chat_mode=request.chat_mode,
                include_message_count=True,
            )
            result["message_id"] = msg_id

        result = normalize_chat_response(result, user_query=request.message)
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Chat] Error type={e.__class__.__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request, user_info: dict = Depends(get_current_user)):
    """Compatibility SSE endpoint.

    Internally delegates to queued task execution so all modes share one streaming pipeline.
    """
    if len(request.message or "") > MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="Message too long")

    user_id = user_info["uid"]
    memory_user_id = resolve_memory_user_id(user_id)
    await check_platform_gates(req, user_id)

    personality, personality_id = resolve_personality_context(
        user_id=user_id,
        session_id=request.session_id,
        chat_mode=request.chat_mode,
        personality=request.personality,
        personality_id=request.personality_id,
    )

    context_messages = []
    if request.session_id:
        context_messages = get_context_for_llm(user_id, request.session_id, personality_id, request.chat_mode)

    effective_settings = merge_settings(get_user_settings(user_id), request.user_settings)
    req_obj = CapabilityRequest(
        user_query=request.message,
        chat_mode=request.chat_mode,
        context_messages=context_messages,
        personality=personality,
        user_settings=effective_settings,
        user_id=memory_user_id,
        session_id=request.session_id,
        file_ids=list(request.file_ids or []),
        metadata={
            "file_ids": list(request.file_ids or []),
            "personality_id": personality_id,
            "stream_schema_version": 2,
            "auth_uid": user_id,
        },
    )
    try:
        task_id = await task_queue.submit(req_obj)
    except RuntimeError as e:
        if "HEAVY_LANE_LIMIT_REACHED" in str(e):
            raise HTTPException(status_code=429, detail="Too many heavy tasks in progress for this user")
        if "HEAVY_LANE_SESSION_LIMIT_REACHED" in str(e):
            raise HTTPException(status_code=429, detail="Too many heavy tasks in progress for this session")
        raise

    async def shim_stream():
        yield ": " + (" " * 1024) + "\n\n"
        last_seq = 0
        sent_done = False
        sent_final = False
        last_token_seq = 0

        while True:
            task = await task_queue.get_task(task_id)
            if not task:
                now = time.time()
                yield f"data: {json.dumps({'type': 'error', 'code': 'not_found', 'message': 'Task not found', 'timestamp': now})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message_id': task_id, 'timestamp': now})}\n\n"
                break

            events = await task_queue.get_events(task_id, after_seq=last_seq)
            for event in events:
                last_seq = max(last_seq, int(event.get("seq", last_seq)))
                ev_type = str(event.get("event") or "progress")
                payload = {k: v for k, v in event.items() if k != "event"}
                if sent_final and ev_type != "done":
                    continue
                if ev_type == "token":
                    payload["seq"] = int(payload.pop("stream_seq", payload.pop("token_seq", 0)) or 0)
                    current_token_seq = int(payload.get("seq") or 0)
                    if current_token_seq <= last_token_seq:
                        continue
                    last_token_seq = current_token_seq
                payload["type"] = ev_type
                if "message_id" not in payload:
                    payload["message_id"] = task_id
                yield f"data: {json.dumps(payload)}\n\n"
                if ev_type == "final":
                    sent_final = True
                if ev_type == "done":
                    sent_done = True
            if sent_done:
                break

            if task.status in {"failed", "cancelled", "completed"} and not events:
                now = time.time()
                if task.status == "completed":
                    if not sent_final:
                        payload = task.result or {"answer": "", "metadata": {"mode": normalize_chat_mode(str(task.request.chat_mode or "smart"))}}
                        yield f"data: {json.dumps({'type': 'final', 'message_id': task_id, 'answer': payload.get('answer', ''), 'summary': payload.get('summary'), 'metadata': payload.get('metadata', {}), 'timestamp': now})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'error', 'message_id': task_id, 'code': task.status, 'message': task.error or 'Task failed', 'timestamp': now})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'message_id': task_id, 'timestamp': now})}\n\n"
                break

            await asyncio.sleep(0.25)

    return StreamingResponse(
        shim_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream",
        },
    )
@router.post("/search")
async def web_search(request: SearchRequest):
    """
    Web search endpoint.
    Returns search results from Serper API.
    """
    from app.llm.router import execute_serper_batch, SERPER_TOOLS
    
    try:
        tools = request.tools or ["Search"]
        results = {}
        
        for tool in tools:
            if tool in SERPER_TOOLS:
                endpoint = SERPER_TOOLS[tool]
                param_key = "url" if tool == "Webpage" else "q"
                result = await execute_serper_batch(endpoint, [request.query], param_key=param_key)
                results[tool] = result
        
        return {"success": True, "results": results}
        
    except Exception as e:
        print(f"[Chat/Search] Error type={e.__class__.__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50, user_info: dict = Depends(get_current_user)):
    """Get chat history for a session"""
    try:
        uid = user_info["uid"]
        messages = load_chat_history(uid, session_id, limit)
        return {"success": True, "messages": messages}
    except Exception as e:
        print(f"[Chat/History] Error type={e.__class__.__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")
