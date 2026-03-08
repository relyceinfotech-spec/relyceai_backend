"""
Relyce AI - FastAPI Main Application
Production-grade ChatGPT-style API with REST and WebSocket support
"""
import json
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.websockets import WebSocketState

from app.config import (
    HOST,
    PORT,
    CORS_ORIGINS,
    CORS_ORIGIN_REGEX,
    CORS_ALLOWED_METHODS,
    CORS_ALLOWED_HEADERS,
    FORCE_HTTPS,
    MAX_CHAT_MESSAGE_CHARS,
    RATE_LIMIT_PER_MINUTE,
    SECURITY_HEADERS_ENABLED,
    SECURITY_CSP,
)
from app.models import (
    ChatRequest, ChatResponse, SearchRequest,
    HealthResponse, WebSocketMessage, Personality
)
from app.auth import verify_token, initialize_firebase, get_current_user
from app.llm.processor import llm_processor
from app.chat.context import get_context_for_llm, update_context_with_exchange
from app.chat.user_profile import get_user_settings, get_session_personality_id, merge_settings
from app.chat.history import save_message_to_firebase, load_chat_history, increment_message_count
from app.rate_limit import check_rate_limit as check_chat_rate_limit
from app.websocket import manager, handle_websocket_message
from app.llm.emotion_engine import emotion_engine
from app.llm.feedback_engine import feedback_engine
from app.llm.prompt_optimizer import prompt_optimizer
from app.llm.user_profiler import user_profiler
from app.safety.stream_moderator import StreamingOutputModerator

from app.llm.processor import active_executions
from pydantic import BaseModel

class ConfirmRequest(BaseModel):
    confirm: bool

from app.payment import router as payment_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("=" * 60)
    print("   [RELYCE-AI] BACKEND - Starting Up")
    print("=" * 60)
    
    try:
        initialize_firebase()
        print("[Startup] - Firebase initialized")
    except Exception as e:
        print(f"[Startup] ! Firebase init failed: {e}")
    
    # Load intent embeddings into memory (for fast routing) in the background
    try:
        import asyncio
        from app.llm.embeddings import load_intent_embeddings
        from app.llm.skill_router_runtime import preload_skill_vectors

        async def _load_embeddings_background():
            try:
                loaded = await load_intent_embeddings()
                if loaded:
                    print("[Startup] - Intent embeddings loaded into RAM")
                else:
                    print("[Startup] - Intent embeddings not found in Firestore")
            except Exception as e:
                print(f"[Startup] ! Embedding load failed: {e}")

        asyncio.create_task(_load_embeddings_background())

        try:
            preload_skill_vectors()
            print("[Startup] - Skill vectors cache loaded")
        except Exception as e:
            print(f"[Startup] ! Skill vector preload failed: {e}")

    except Exception as e:
        print(f"[Startup] ! Embedding task init failed: {e}")

    # Load self-learning scores from Firestore (strategy scores + prompt variants)
    try:
        async def _load_learning_scores():
            try:
                await feedback_engine.load_scores()
                print("[Startup] - Feedback engine scores loaded")
            except Exception as e:
                print(f"[Startup] ! Feedback score load failed: {e}")
            try:
                await prompt_optimizer.load_stats()
                print("[Startup] - Prompt optimizer stats loaded")
            except Exception as e:
                print(f"[Startup] ! Prompt optimizer load failed: {e}")

        asyncio.create_task(_load_learning_scores())
    except Exception as e:
        print(f"[Startup] ! Learning score task init failed: {e}")

    try:
        from app.llm.processor import _execution_registry_cleanup
        asyncio.create_task(_execution_registry_cleanup())
        print("[Startup] - Execution registry cleanup task started")
    except Exception as e:
        print(f"[Startup] ! Execution cleanup init failed: {e}")

    # Ã°Å¸â€Â§ API WARMUP: Pre-warm OpenRouter TLS/DNS pools to eliminate cold start latency
    try:
        async def _warmup_api():
            try:
                from app.llm.router import get_openrouter_client
                client = get_openrouter_client()
                # Lightweight ping to establish TLS handshake and DNS resolution
                await client.models.list()
                print("[Startup] - OpenRouter API pool warmed (TLS/DNS ready)")
            except Exception as e:
                print(f"[Startup] ! API warmup failed (non-blocking): {e}")

        asyncio.create_task(_warmup_api())
    except Exception as e:
        print(f"[Startup] ! API warmup task init failed: {e}")

    # Ã°Å¸â€Â§ WEAVIATE WARMUP: Pre-connect to Weaviate Cloud for vector memory
    try:
        async def _warmup_weaviate():
            try:
                from app.memory.weaviate_client import get_weaviate_client
                from app.memory.vector_memory import set_vector_enabled
                client = await get_weaviate_client()
                if client:
                    set_vector_enabled(True)
                    print("[Startup] - Weaviate vector memory connected")
                else:
                    set_vector_enabled(False)
                    print("[Startup] - Weaviate not configured (Firestore fallback active)")
            except Exception as e:
                from app.memory.vector_memory import set_vector_enabled
                set_vector_enabled(False)
                print(f"[Startup] ! Weaviate warmup failed (Firestore fallback active): {e}")

        asyncio.create_task(_warmup_weaviate())
    except Exception as e:
        print(f"[Startup] ! Weaviate warmup task init failed: {e}")

    print(f"[Startup] - Server ready on {HOST}:{PORT}")
    print("=" * 60)
    
    yield
    
    # Shutdown
    await emotion_engine.shutdown()
    await feedback_engine.shutdown()
    await prompt_optimizer.save_stats()
    await user_profiler.shutdown()
    try:
        from app.memory.weaviate_client import close_weaviate_client
        await close_weaviate_client()
    except Exception:
        pass
    print("[Shutdown] - Relyce AI Backend shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Relyce AI API",
    description="Production-grade ChatGPT-style backend with multi-device support",
    version="1.0.0",
    lifespan=lifespan
)

# HTTPS redirect & HSTS (enable in production)
if FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

    @app.middleware("http")
    async def add_hsts_header(request, call_next):
        response = await call_next(request)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        return response

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print(f"[Validation Error] 422 Unprocessable Entity - Body: {body}")
    print(f"[Validation Error] Details: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode("utf-8")},
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)

# Security headers (skip CSP on docs/redoc to avoid breaking Swagger UI)
if SECURITY_HEADERS_ENABLED:
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "interest-cohort=()")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        path = request.url.path
        if not (path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi.json")):
            response.headers.setdefault("Content-Security-Policy", SECURITY_CSP)
            response.headers.setdefault("X-Frame-Options", "DENY")
        return response


# ============================================
# OBSERVABILITY API (Phase 7)
# ============================================

from app.observability.event_logger import get_event_logger
from app.observability.metrics_collector import get_metrics_collector
from app.observability.anomaly_detector import AnomalyDetector

@app.get("/agent/metrics")
async def agent_metrics():
    """Returns aggregated execution kernel metrics."""
    metrics = get_metrics_collector()
    return {"metrics": metrics.get_metrics()}

@app.post("/agent/cancel")
async def agent_cancel(request: Request):
    """
    Cancel a running agent execution for a given chat session.
    Enables interrupt safety Ã¢â‚¬â€ frontend calls this before sending a new prompt.
    """
    body = await request.json()
    user_id = body.get("user_id", "")
    chat_id = body.get("chat_id", "")
    if not user_id or not chat_id:
        return {"status": "error", "message": "user_id and chat_id required"}
    
    from app.websocket import _cancel_flags
    cancel_key = (user_id, chat_id)
    _cancel_flags[cancel_key] = True
    return {"status": "cancelled", "user_id": user_id, "chat_id": chat_id}

# ============================================
# SMART MEMORY ENDPOINTS
# ============================================

@app.get("/api/memories/{user_id}")
async def get_memories(user_id: str):
    """Get all auto-detected memories for a user."""
    from app.chat.smart_memory import get_all_memories_for_api
    memories = get_all_memories_for_api(user_id)
    return {"memories": memories, "count": len(memories)}

@app.delete("/api/memories/{user_id}/{memory_id}")
async def delete_memory_endpoint(user_id: str, memory_id: str):
    """Delete a specific memory."""
    from app.chat.smart_memory import delete_memory
    success = delete_memory(user_id, memory_id)
    if success:
        return {"status": "deleted", "memory_id": memory_id}
    return {"status": "error", "message": "Failed to delete memory"}

@app.post("/api/memories/import")
async def import_memories_endpoint(request: Request):
    """
    Import memories from other AI platforms.
    Uses LLM to parse raw pasted text into structured memory entries.
    """
    body = await request.json()
    user_id = body.get("user_id", "")
    raw_text = body.get("text", "")
    
    if not user_id or not raw_text:
        return {"status": "error", "message": "user_id and text required"}
    
    if len(raw_text) > 10000:
        return {"status": "error", "message": "Text too long (max 10000 chars)"}
    
    try:
        from app.llm.router import get_openai_client
        from app.chat.smart_memory import MemoryEntry, store_memory
        import json as json_lib
        
        # Use LLM to parse the raw text into structured memories
        parse_prompt = """You are a memory parser. Given raw memory exports from AI platforms (ChatGPT, Gemini, Claude etc.), extract individual memory entries.

For each memory, output:
- content: A clean, concise description (max 100 chars)
- category: One of: identity, profession, preference, project, context
- importance: 0.0 to 1.0 (higher = more important)

Rules:
- identity: name, age, location, education, language, family
- profession: job, skills, tech stack, career history
- preference: likes, dislikes, communication style, instructions
- project: things they built, are building, or plan to build
- context: hobbies, relationships, general life details

Output ONLY valid JSON array. No explanation.
Example: [{"content": "Name: Tamizh", "category": "identity", "importance": 0.85}, ...]"""

        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": parse_prompt},
                {"role": "user", "content": f"Parse these memories:\n\n{raw_text[:8000]}"}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2000,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        parsed = json_lib.loads(result_text)
        
        # Handle both {"memories": [...]} and direct [...]
        entries_data = parsed if isinstance(parsed, list) else parsed.get("memories", parsed.get("entries", []))
        
        stored_count = 0
        stored_entries = []
        for item in entries_data:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            entry = MemoryEntry(
                content=item["content"][:200],
                category=item.get("category", "context"),
                importance=min(1.0, max(0.1, float(item.get("importance", 0.6)))),
                source="imported"
            )
            if store_memory(user_id, entry):
                stored_count += 1
                stored_entries.append({"content": entry.content, "category": entry.category})
        
        return {
            "status": "success",
            "imported": stored_count,
            "entries": stored_entries
        }
        
    except Exception as e:
        print(f"[MemoryImport] Error: {e}")
        return {"status": "error", "message": f"Import failed: {str(e)}"}

@app.get("/api/memories/export/{user_id}")
async def export_memories_endpoint(user_id: str):
    """Export all memories in a structured format for copying."""
    from app.chat.smart_memory import load_all_memories
    memories = load_all_memories(user_id, force_refresh=True)
    
    if not memories:
        return {"status": "empty", "export": "No memories stored yet."}
    
    # Group by category
    groups = {}
    for mem in memories:
        cat = mem.category.title()
        if cat not in groups:
            groups[cat] = []
        date = mem.created_at[:10] if mem.created_at else "unknown"
        groups[cat].append(f"[{date}] - {mem.content}")
    
    export_lines = []
    category_order = ["Identity", "Profession", "Project", "Preference", "Context"]
    for cat in category_order:
        if cat in groups:
            export_lines.append(f"## {cat}")
            for line in groups[cat]:
                export_lines.append(line)
            export_lines.append("")
    
    return {"status": "success", "export": "\n".join(export_lines), "count": len(memories)}

@app.get("/api/memories/summary/{user_id}")
async def get_memory_summary(user_id: str):
    """Generate an LLM-powered prose summary of what Relyce knows about the user."""
    from app.chat.smart_memory import load_all_memories
    
    memories = load_all_memories(user_id, force_refresh=True)
    if not memories:
        return {"status": "empty", "summary": "No memories stored yet. Start chatting and Relyce will learn about you!"}
    
    try:
        from app.llm.router import get_openai_client
        
        # Build memory context for LLM
        mem_lines = []
        for mem in memories:
            mem_lines.append(f"[{mem.category}] {mem.content}")
        mem_text = "\n".join(mem_lines)
        
        client = get_openai_client()
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are generating a memory summary for an AI assistant. Given the user's stored memories, write a clear prose summary organized into sections:

**Work context** - Education, career, skills, tech stack
**Personal context** - Name, preferences, communication style, personality 
**Top of mind** - Current projects, active goals, recent interests

Keep it concise but informative. Write in third person (e.g., "User is..."). If a section has no relevant memories, skip it."""},
                {"role": "user", "content": f"Summarize these memories:\n\n{mem_text}"}
            ],
            max_completion_tokens=500,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        return {"status": "success", "summary": summary, "memory_count": len(memories)}
        
    except Exception as e:
        print(f"[MemorySummary] Error: {e}")
        # Fallback: just list them
        fallback = "\n".join([f"Ã¢â‚¬Â¢ {m.content}" for m in memories[:20]])
        return {"status": "success", "summary": fallback, "memory_count": len(memories)}

@app.patch("/api/memories/{user_id}/{memory_id}")
async def update_memory_endpoint(user_id: str, memory_id: str, request: Request):
    """Update a memory's content."""
    body = await request.json()
    new_content = body.get("content", "")
    if not new_content:
        return {"status": "error", "message": "content required"}
    
    from app.chat.smart_memory import _get_entries_ref, _invalidate_cache
    try:
        ref = _get_entries_ref(user_id)
        if ref:
            from datetime import datetime
            ref.document(memory_id).update({
                "content": new_content[:200],
                "last_used": datetime.utcnow().isoformat()
            })
            _invalidate_cache(user_id)
            return {"status": "updated", "memory_id": memory_id}
    except Exception as e:
        print(f"[MemoryUpdate] Error: {e}")
    return {"status": "error", "message": "Failed to update memory"}

@app.delete("/api/memories/clear/{user_id}")
async def clear_all_memories(user_id: str):
    """Delete all memories for a user."""
    from app.chat.smart_memory import load_all_memories, _get_entries_ref, _invalidate_cache
    try:
        memories = load_all_memories(user_id, force_refresh=True)
        ref = _get_entries_ref(user_id)
        if ref:
            for mem in memories:
                if mem.doc_id:
                    ref.document(mem.doc_id).delete()
            _invalidate_cache(user_id)
        return {"status": "cleared", "count": len(memories)}
    except Exception as e:
        print(f"[MemoryClear] Error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/agent/events")
async def agent_events(limit: int = Query(default=100, ge=1, le=500)):
    """Returns recent telemetry events."""
    logger = get_event_logger()
    return {"recent_events": logger.get_recent(limit=limit)}

@app.get("/agent/anomalies")
async def agent_anomalies():
    """Evaluates current metrics and returns any triggered anomalies."""
    detector = AnomalyDetector(get_metrics_collector(), get_event_logger())
    alerts = detector.evaluate()
    return {"anomalies": alerts, "count": len(alerts)}


# ============================================
# HEALTH GOVERNANCE API (Phase 8)
# ============================================

from app.health.health_monitor import get_health_monitor
from app.health.circuit_breaker import get_circuit_breaker
from app.health.mitigation_policies import get_mitigation_policies
from app.health.worker_lifecycle import get_worker_lifecycle
from fastapi.responses import JSONResponse

@app.get("/agent/health")
async def agent_health():
    """Returns full system health status."""
    monitor = get_health_monitor()
    breaker = get_circuit_breaker()
    lifecycle = get_worker_lifecycle()
    mitigation = get_mitigation_policies()

    signal = monitor.evaluate()
    breaker.evaluate(signal)

    return {
        "status": signal,
        "circuit_breaker": breaker.get_state(),
        "worker": lifecycle.get_state(),
        "active_mitigations": mitigation.get_active(),
        "metrics_snapshot": get_metrics_collector().get_metrics(),
    }

async def circuit_breaker_gate():
    """
    FastAPI dependency that evaluates health and blocks if breaker is OPEN.
    Inject into any route: Depends(circuit_breaker_gate)
    """
    monitor = get_health_monitor()
    breaker = get_circuit_breaker()
    mitigation = get_mitigation_policies()

    signal = monitor.evaluate()
    breaker.evaluate(signal)
    mitigation.apply(signal)

    if not breaker.allow_execution():
        logger = get_event_logger()
        from app.observability.event_types import EventType
        logger.emit(EventType.ANOMALY_ALERT, {
            "reason": "Circuit open Ã¢â‚¬â€ execution blocked",
            "signal": signal,
        })
        return JSONResponse(
            content={"error": "SYSTEM_UNAVAILABLE", "signal": signal},
            status_code=503,
        )
    return None

# ============================================
# COST & RATE GOVERNANCE API (Phase 10)
# ============================================

from app.governance.rate_limiter import get_rate_limiter
from app.governance.quota_manager import get_quota_manager
from app.governance.spend_guard import get_spend_guard
from app.governance.token_tracker import get_token_tracker
from app.governance.usage_store import get_usage_store

@app.get("/agent/governance")
async def agent_governance(user_id: str = Query(default="anonymous")):
    """Returns governance status for a user."""
    quota = get_quota_manager()
    spend = get_spend_guard()
    usage = get_usage_store().get_or_create(user_id)
    return {
        "user_id": user_id,
        "usage": {
            "daily_requests": usage.daily_requests,
            "daily_tokens": usage.daily_tokens,
            "concurrent_tasks": usage.concurrent_tasks,
        },
        "quota_remaining": quota.get_remaining(user_id),
        "system_spend": spend.get_status(),
    }

@app.get("/agent/spend")
async def agent_spend():
    """Returns global system spend status."""
    spend = get_spend_guard()
    tracker = get_token_tracker()
    return {
        "spend": spend.get_status(),
        "token_totals": tracker.get_totals(),
    }

async def governance_gate(user_id: str, tier: str = "free"):
    """
    Reusable governance gate. Call before any LLM execution.
    Returns None if allowed, or a JSONResponse with error if blocked.

    Chain: rate_limiter Ã¢â€ â€™ quota_manager Ã¢â€ â€™ spend_guard
    """
    # 1. Burst rate limit
    limiter = get_rate_limiter()
    allowed, reason = limiter.check_and_record(user_id)
    if not allowed:
        return JSONResponse(
            content={"error": reason, "retry_after_seconds": 60},
            status_code=429,
        )

    # 2. Daily quota check
    quota = get_quota_manager()
    allowed, reason = quota.check(user_id, tier=tier)
    if not allowed:
        return JSONResponse(
            content={"error": reason, "quota": quota.get_remaining(user_id, tier=tier)},
            status_code=429,
        )

    # 3. Global spend guard
    guard = get_spend_guard()
    if not guard.allow():
        return JSONResponse(
            content={"error": "SYSTEM_SPEND_LIMIT_REACHED", "spend": guard.get_status()},
            status_code=503,
        )

    return None


# ============================================
# ABUSE DETECTION API (Advanced Hardening)
# ============================================

from app.governance.abuse_detector import get_abuse_detector
from app.governance.ip_tracker import get_ip_tracker

@app.get("/agent/abuse-status")
async def agent_abuse_status(ip: str = Query(default="unknown")):
    """Returns abuse detection status for an IP."""
    tracker = get_ip_tracker()
    return {
        "ip": ip,
        "flagged": tracker.is_flagged(ip),
        "request_count_1m": tracker.get_request_count(ip, window_seconds=60),
        "failure_count_5m": tracker.get_failure_count(ip, window_seconds=300),
        "associated_users": tracker.get_associated_users(ip),
    }

# ============================================
# UX DASHBOARD PULL API (Part 3)
# ============================================

from app.dashboard.dashboard_data import get_dashboard_aggregator, DEFAULT_MODEL

@app.get("/dashboard/overview")
async def dashboard_overview(model: str = Query(default=DEFAULT_MODEL)):
    """Full dashboard payload including task, cost, health, and errors."""
    aggregator = get_dashboard_aggregator()
    return aggregator.get_full_overview(model_name=model)

@app.get("/dashboard/tokens")
async def dashboard_tokens(model: str = Query(default=DEFAULT_MODEL)):
    """Token usage and cost estimates."""
    aggregator = get_dashboard_aggregator()
    return aggregator.get_token_dashboard(model_name=model)

@app.get("/dashboard/errors")
async def dashboard_errors():
    """Error rates and top failing tools."""
    aggregator = get_dashboard_aggregator()
    return aggregator.get_error_breakdown()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/agent/cancel/{execution_id}")
async def cancel_agent_execution(execution_id: str):
    """Cancel a running agent execution"""
    if execution_id in active_executions:
        exec_data = active_executions[execution_id]
        ctx = exec_data["ctx"]
        ctx.terminate = True
        if hasattr(ctx, "pause_event") and ctx.pause_event:
            ctx.pause_event.set()
        return {"success": True, "message": "Execution cancelled"}
    raise HTTPException(status_code=404, detail="Execution not found or already completed")

@app.post("/agent/confirm/{execution_id}")
async def confirm_agent_tool(execution_id: str, request: ConfirmRequest):
    """Confirm or deny an awaiting tool action"""
    if execution_id in active_executions:
        exec_data = active_executions[execution_id]
        ctx = exec_data["ctx"]
        ctx.confirmation = request.confirm
        if hasattr(ctx, "pause_event") and ctx.pause_event:
            ctx.pause_event.set()
        return {"success": True, "message": f"Action {'confirmed' if request.confirm else 'denied'}"}
    raise HTTPException(status_code=404, detail="Execution not found or already completed")

# ============================================
# PHASE 4C: WORKFLOW STATE ENDPOINTS
# ============================================

from pydantic import BaseModel
class ResumeRequest(BaseModel):
    session_id: str
    message: str = "Continue task"

@app.post("/agent/resume/{task_id}")
async def resume_task(task_id: str, request: ResumeRequest, req: Request, user_info: dict = Depends(get_current_user)):
    """
    Resume a PAUSED or RUNNING multi-step agent task.
    """
    from app.state.task_state_engine import get_task_state, set_task_status, summarize_task_history
    from app.agent.hybrid_controller import generate_strategy_advice
    from fastapi.responses import StreamingResponse

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
        try:
            yield ": " + (" " * 1024) + "\n\n"
            
            # Fetch normal context
            context_messages = get_context_for_llm(user_id, session_id, None)
            effective_settings = get_user_settings(user_id)
            
            # Fire LLM (passing resume_graph into kwargs)
            stream_moderator = StreamingOutputModerator(query=request.message)
            async for token in llm_processor.process_message_stream(
                user_query=augmented_message,
                mode="normal",
                context_messages=context_messages,
                personality=None,
                user_settings=effective_settings,
                user_id=user_id,
                session_id=session_id,
                task_id=task_id, # Phase 4C context injection
                resume_graph=resume_graph # Phase 5 Graph Resumption
            ):
                if token.strip().startswith("[INFO]"):
                    clean_info = token.replace("[INFO]", "").strip()
                    yield f"data: {json.dumps({'type': 'info', 'content': clean_info})}\n\n"
                    continue
                safe_chunk = stream_moderator.ingest(token)
                if safe_chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': safe_chunk})}\n\n"
                
            tail_chunk = stream_moderator.finalize()
            if tail_chunk:
                yield f"data: {json.dumps({'type': 'token', 'content': tail_chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            
        except Exception as e:
            set_task_status(session_id, task_id, "PAUSED")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
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



# ============================================
# AUTH RATE LIMITING ENDPOINTS
# ============================================
from fastapi import Request
from pydantic import BaseModel, EmailStr
from app.rate_limiter import check_rate_limit as check_login_rate_limit, record_failed_attempt, clear_attempts

class RateLimitRequest(BaseModel):
    email: EmailStr

@app.post("/auth/check-limit")
async def check_login_limit(request: RateLimitRequest, req: Request):
    """Check if login attempt is allowed for this email+IP"""
    ip = req.client.host if req.client else "unknown"
    result = check_login_rate_limit(request.email, ip)
    return result

@app.post("/auth/record-failure")
async def record_login_failure(request: RateLimitRequest, req: Request):
    """Record a failed login attempt"""
    ip = req.client.host if req.client else "unknown"
    result = record_failed_attempt(request.email, ip)
    return result

@app.post("/auth/clear-attempts")
async def clear_login_attempts(request: RateLimitRequest, req: Request):
    """Clear login attempts on successful login"""
    ip = req.client.host if req.client else "unknown"
    success = clear_attempts(request.email, ip)
    return {"success": success}

from app.chat.personalities import (
    get_all_personalities, 
    create_custom_personality, 
    get_personality_by_id
)

@app.get("/personalities/{user_id}")
async def get_personalities(user_id: str, user_info: dict = Depends(get_current_user)):
    """Get all available personalities for a user"""
    uid = user_info["uid"]
    return {"success": True, "personalities": get_all_personalities(uid)}

@app.post("/personalities")
async def create_personality(request: Personality, user_info: dict = Depends(get_current_user)):
    """Create a new custom personality"""
    user_id = user_info["uid"]
    result = create_custom_personality(
        user_id, 
        request.name, 
        request.description, 
        request.prompt,
        request.content_mode,  # Pass content_mode
        request.specialty  # Pass specialty
    )
    if result:
        return {"success": True, "personality": result}
    raise HTTPException(status_code=500, detail="Failed to create personality")


@app.put("/personalities/{personality_id}")
async def update_personality(personality_id: str, request: Personality, user_info: dict = Depends(get_current_user)):
    """Update a custom personality"""
    from app.chat.personalities import update_custom_personality
    
    user_id = user_info["uid"]
    success = update_custom_personality(
        user_id,
        personality_id,
        request.name,
        request.description,
        request.prompt,
        request.content_mode,  # Pass content_mode
        request.specialty  # Pass specialty
    )
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to update")


@app.delete("/personalities/{personality_id}")
async def delete_personality(personality_id: str, user_info: dict = Depends(get_current_user)):
    """Delete a custom personality"""
    from app.chat.personalities import delete_custom_personality
    
    user_id = user_info["uid"]
    success = delete_custom_personality(user_id, personality_id)
    
    if success:
        return {"success": True}
    raise HTTPException(status_code=404, detail="Personality not found or failed to delete")


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
        import json
        payload = json.loads(gov_response.body)
        raise HTTPException(status_code=gov_response.status_code, detail=payload)

    # 3. Circuit Breaker Gate
    cb_response = await circuit_breaker_gate()
    if cb_response:
        import json
        payload = json.loads(cb_response.body)
        raise HTTPException(status_code=cb_response.status_code, detail=payload)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request, user_info: dict = Depends(get_current_user)):
    """
    Non-streaming chat endpoint.
    Processes message and returns complete response.
    """
    try:
        if len(request.message or "") > MAX_CHAT_MESSAGE_CHARS:
            raise HTTPException(status_code=413, detail="Message too long")
        user_id = user_info["uid"]
        
        # Resolve uniqueUserId from Firestore (memories are stored under this ID)
        memory_user_id = user_id  # fallback to uid
        try:
            from app.auth import get_firestore_db
            _db = get_firestore_db()
            if _db:
                user_doc = _db.collection("users").document(user_id).get()
                if user_doc.exists:
                    memory_user_id = user_doc.to_dict().get("uniqueUserId", user_id)
        except Exception as e:
            print(f"[HTTP] uniqueUserId resolution error (non-blocking): {e}")

        # Enforce all platform safety and governance gates
        await check_platform_gates(req, user_id)
        
        request.user_id = user_id

        # Resolve personality
        personality = request.personality
        personality_id = request.personality_id
        if personality and not personality_id:
            if hasattr(personality, "id"):
                personality_id = personality.id
        if not personality and not personality_id and request.session_id:
            saved_id = get_session_personality_id(user_id, request.session_id)
            if saved_id:
                personality_id = saved_id
        if not personality and personality_id:
            p_data = get_personality_by_id(user_id, personality_id)
            if p_data:
                personality = p_data
        # Default to Relyce AI for normal mode if none provided
        if not personality and request.chat_mode == "normal":
            p_data = get_personality_by_id(user_id, "default_relyce")
            if p_data:
                personality = p_data
                personality_id = personality_id or p_data.get("id")
        
        # Convert Personality object to dict if it is an object
        if hasattr(personality, "dict"):
            personality = personality.dict()
        elif hasattr(personality, "model_dump"):
            personality = personality.model_dump()
        if personality and not personality_id:
            personality_id = personality.get("id")

        # Get context if session exists (personality-aware)
        context_messages = []
        if request.session_id:
            context_messages = get_context_for_llm(user_id, request.session_id, personality_id)

        # Process message with concurrency tracking
        store = get_usage_store()
        store.increment_concurrent(user_id)
        try:
            effective_settings = merge_settings(get_user_settings(user_id), request.user_settings)
            result = await llm_processor.process_message(
                user_query=request.message,
                mode=request.chat_mode,
                context_messages=context_messages,
                personality=personality,
                user_settings=effective_settings,
                user_id=memory_user_id, # Pass unique user ID for facts
                session_id=request.session_id
            )
        finally:
            store.decrement_concurrent(user_id)
        
        # Update context
        if request.session_id:
            update_context_with_exchange(
                user_id, 
                request.session_id,
                request.message,
                result["response"],
                personality_id
            )
            
            # Save to Firebase
            from app.chat.history import save_message_to_firebase
            save_message_to_firebase(
                user_id, request.session_id, "user", request.message, personality_id
            )
            msg_id = save_message_to_firebase(
                user_id, request.session_id, "assistant", result["response"], personality_id
            )
            result["message_id"] = msg_id
            
            # Increment Usage
            increment_message_count(user_id)
            
            # Smart Memory Compression (Background)
            try:
                from app.llm.router import get_openrouter_client
                from app.memory.summary_manager import summarize_if_needed
                # Pass the fresh context for summarization check
                fresh_context_msgs = get_context_for_llm(user_id, request.session_id, personality_id)
                asyncio.create_task(summarize_if_needed(user_id, request.session_id, fresh_context_msgs, get_openrouter_client()))
            except Exception as sum_e:
                print(f"[Chat] Background summary error: {sum_e}")
        
        return ChatResponse(**result)
        
    except Exception as e:
        print(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request, user_info: dict = Depends(get_current_user)):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Returns tokens as they're generated.
    """
    if len(request.message or "") > MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="Message too long")
    user_id = user_info["uid"]
    
    # Resolve uniqueUserId from Firestore (memories are stored under this ID)
    memory_user_id = user_id  # fallback to uid
    try:
        from app.auth import get_firestore_db
        _db = get_firestore_db()
        if _db:
            user_doc = _db.collection("users").document(user_id).get()
            if user_doc.exists:
                memory_user_id = user_doc.to_dict().get("uniqueUserId", user_id)
    except Exception as e:
        print(f"[HTTP] uniqueUserId resolution error (non-blocking): {e}")

    # Enforce all platform safety and governance gates
    await check_platform_gates(req, user_id)

    async def generate():
        store = None
        try:
            # Ã°Å¸Å¡â‚¬ Force flush buffer immediately with padding (1KB)
            # This helps in environments like Vercel/Nginx/Render that buffer responses
            yield ": " + (" " * 1024) + "\n\n"

            request.user_id = user_id

            # Resolve personality
            personality = request.personality
            personality_id = request.personality_id
            if personality and not personality_id:
                if hasattr(personality, "id"):
                    personality_id = personality.id
            if not personality and not personality_id and request.session_id:
                saved_id = get_session_personality_id(user_id, request.session_id)
                if saved_id:
                    personality_id = saved_id
            if not personality and personality_id:
                p_data = get_personality_by_id(user_id, personality_id)
                if p_data:
                    personality = p_data
            # Default to Relyce AI for normal mode if none provided
            if not personality and request.chat_mode == "normal":
                p_data = get_personality_by_id(user_id, "default_relyce")
                if p_data:
                    personality = p_data
                    personality_id = personality_id or p_data.get("id")
            
             # Convert Personality object to dict if it is an object
            if hasattr(personality, "model_dump"):
                personality = personality.model_dump()
            elif hasattr(personality, "dict"):
                personality = personality.dict()
            if personality and not personality_id:
                personality_id = personality.get("id")

            # Get context if session exists (personality-aware)
            context_messages = []
            if request.session_id:
                context_messages = get_context_for_llm(user_id, request.session_id, personality_id)

            full_response = ""
            
            effective_settings = merge_settings(get_user_settings(user_id), request.user_settings)
            
            store = get_usage_store()
            store.increment_concurrent(user_id)
            stream_generator = None
            stream_moderator = StreamingOutputModerator(query=request.message)
            try:
                stream_generator = llm_processor.process_message_stream(
                    request.message,
                    mode=request.chat_mode,
                    context_messages=context_messages,
                    personality=personality,
                    user_settings=effective_settings,
                    user_id=memory_user_id, # Pass unique user ID for facts
                    session_id=request.session_id
                )
                async for token in stream_generator:
                    if token.strip().startswith("[INFO]"):
                        clean_info = token.replace("[INFO]", "").strip()
                        yield f"data: {json.dumps({'type': 'info', 'content': clean_info})}\n\n"
                        continue

                    safe_chunk = stream_moderator.ingest(token)
                    if safe_chunk:
                        full_response += safe_chunk
                        yield f"data: {json.dumps({'type': 'token', 'content': safe_chunk})}\n\n"

                # Send done signal
                tail_chunk = stream_moderator.finalize()
                if tail_chunk:
                    full_response += tail_chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': tail_chunk})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"

                # Update context and save to Firebase
                if request.session_id:
                    update_context_with_exchange(
                        user_id,
                        request.session_id,
                        request.message,
                        full_response,
                        personality_id
                    )
                    
                    # Save to Firebase
                    from app.chat.history import save_message_to_firebase
                    save_message_to_firebase(
                        user_id, request.session_id, "user", request.message, personality_id
                    )
                    save_message_to_firebase(
                        user_id, request.session_id, "assistant", full_response, personality_id
                    )
                    
                    # Increment Usage
                    increment_message_count(user_id)
                    
                    # Smart Memory Compression (Background)
                    try:
                        from app.llm.router import get_openrouter_client
                        from app.memory.summary_manager import summarize_if_needed
                        # Pass the fresh context for summarization check
                        fresh_context_msgs = get_context_for_llm(user_id, request.session_id, personality_id)
                        asyncio.create_task(summarize_if_needed(user_id, request.session_id, fresh_context_msgs, get_openrouter_client()))
                    except Exception as sum_e:
                        print(f"[Chat/Stream] Background summary error: {sum_e}")
                        
            except asyncio.CancelledError:
                print(f"[Chat/Stream] Stream cancelled by client disconnect")
                if stream_generator:
                    try:
                        await stream_generator.aclose()
                    except Exception:
                        pass
                raise
                
        except Exception as e:
            if not isinstance(e, asyncio.CancelledError):
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            if store:
                store.decrement_concurrent(user_id)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # Critical for Nginx/Vercel
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/search")
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50, user_info: dict = Depends(get_current_user)):
    """Get chat history for a session"""
    try:
        uid = user_info["uid"]
        messages = load_chat_history(uid, session_id, limit)
        return {"success": True, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Payment Router
app.include_router(payment_router, prefix="/payment", tags=["Payment"])

# Admin & User Management
from app.routers import admin, users
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(users.router, prefix="/users", tags=["Users"])



# ============================================
# WEBSOCKET ENDPOINT
# ============================================

@app.websocket("/ws/chat")
async def websocket_endpoint(
    websocket: WebSocket,
    chat_id: Optional[str] = Query(None)
):
    """
    WebSocket chat endpoint with multi-device support.
    
    Connection URL: ws://localhost:8080/ws/chat?chat_id=SESSION_ID
    
    Message format (JSON):
    - Authenticate: {"type": "auth", "token": "FIREBASE_TOKEN", "chat_id": "SESSION_ID"}
    - Send message: {"type": "message", "content": "Hello", "chat_mode": "normal"}
    - Stop generation: {"type": "stop"}
    - Ping: {"type": "ping"}
    
    Response format (JSON):
    - Token: {"type": "token", "content": "..."}
    - Done: {"type": "done", "content": ""}
    - Error: {"type": "error", "content": "..."}
    - Info: {"type": "info", "content": "processing"|"stopped"}
    """

    # Accept connection immediately to handle errors gracefully
    await websocket.accept()

    async def safe_send(payload: dict) -> bool:
        """Send only if the socket is still open."""
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                return False
            await websocket.send_text(json.dumps(payload))
            return True
        except Exception:
            return False

    # Authenticate via initial auth message (preferred) or query param (legacy)
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

    # Verify token safely
    try:
        is_valid, user_info = verify_token(token)
    except Exception as e:
        print(f"[WS] Auth Error: {e}")
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
    
    # Use provided chat_id or generate one
    if not resolved_chat_id:
        resolved_chat_id = f"chat_{datetime.now().timestamp()}"
    
    # Connect
    try:
        connection_id = await manager.connect(websocket, resolved_chat_id, user_id)
    except Exception as e:
        await safe_send({"type": "error", "content": str(e)})
        await websocket.close(code=1013)
        return

    # Resolve uniqueUserId from Firestore (memories are stored under this ID)
    unique_user_id = user_id  # fallback to uid
    try:
        from app.auth import get_firestore_db
        _db = get_firestore_db()
        if _db:
            user_doc = _db.collection("users").document(user_id).get()
            if user_doc.exists:
                unique_user_id = user_doc.to_dict().get("uniqueUserId", user_id)
        # Store in connection info so handle_websocket_message can use it
        if connection_id in manager.connection_info:
            manager.connection_info[connection_id]["unique_user_id"] = unique_user_id
    except Exception as e:
        print(f"[WS] uniqueUserId resolution error (non-blocking): {e}")

    # Notify client auth success
    await safe_send({"type": "auth_ok"})

    # Pre-warm emotion cache for this session
    try:
        await emotion_engine.prewarm(resolved_chat_id)
    except Exception:
        pass  # Non-critical, don't block connection
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, connection_id, message, manager)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "content": "Invalid JSON"}),
                    websocket
                )
                
    except WebSocketDisconnect:
        await manager.disconnect(connection_id)
    except Exception as e:
        print(f"[WS] Error: {e}")
        await manager.disconnect(connection_id)


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )








