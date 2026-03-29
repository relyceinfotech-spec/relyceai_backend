"""
Relyce AI - FastAPI Main Application
Production-grade ChatGPT-style API with REST and WebSocket support
"""
import asyncio
import hashlib
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

from app.config import (
    HOST,
    PORT,
    EMBEDDING_MODEL,
    CORS_ORIGINS,
    CORS_ORIGIN_REGEX,
    CORS_ALLOWED_METHODS,
    CORS_ALLOWED_HEADERS,
    FORCE_HTTPS,
    SECURITY_HEADERS_ENABLED,
    SECURITY_CSP,
)
from app.auth import initialize_firebase
from app.llm.emotion_engine import emotion_engine
from app.llm.feedback_engine import feedback_engine
from app.llm.prompt_optimizer import prompt_optimizer
from app.llm.user_profiler import user_profiler
from app.routers.memory import router as memory_router
from app.routers.account import router as account_router
from app.routers.system import router as system_router

from app.billing.payment_router import router as payment_router
from app.routers.ws_runtime import router as ws_runtime_router
from app.routers.chat_runtime import router as chat_runtime_router
from app.routers.files import router as files_router
from app.platform import get_task_queue
from app.safety.safety_filter import detect_injection

MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", "1048576"))  # 1MB default
SECURITY_INPUT_FILTER_ENABLED = os.getenv("SECURITY_INPUT_FILTER_ENABLED", "1").strip() in {"1", "true", "True"}


def _redact_sensitive_text(text: str, max_len: int = 1000) -> str:
    if not text:
        return ""
    clipped = text[:max_len]
    patterns = [
        r"(?i)(authorization\s*[:=]\s*)(bearer\s+[A-Za-z0-9\-._~+/]+=*)",
        r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9\-._~+/]{10,})",
        r"(?i)(token\s*[:=]\s*)([A-Za-z0-9\-._~+/]{10,})",
        r"(?i)(password\s*[:=]\s*)([^\s,\"']{4,})",
    ]
    out = clipped
    for pat in patterns:
        out = re.sub(pat, r"\1[REDACTED]", out)
    return out


_HIGH_RISK_PROMPT_EXFIL_PATTERNS = [
    re.compile(r"(?i)\b(reveal|show|print|dump|output)\b.{0,40}\b(system prompt|developer instructions?|hidden instructions?)\b"),
    re.compile(r"(?i)\b(ignore|bypass|disregard)\b.{0,50}\b(previous|all)\b.{0,20}\binstructions?\b"),
    re.compile(r"(?i)\b(expose|leak)\b.{0,30}\b(prompt|policy|internal)\b"),
]


def _is_blocked_prompt_exfil_attempt(path: str, content_type: str, body: bytes) -> bool:
    if not SECURITY_INPUT_FILTER_ENABLED:
        return False
    if not path.startswith("/chat"):
        return False
    low_ct = (content_type or "").lower()
    if "application/json" not in low_ct and "text/plain" not in low_ct:
        return False
    text = body.decode("utf-8", errors="replace")[:8000]
    if detect_injection(text):
        return any(p.search(text) for p in _HIGH_RISK_PROMPT_EXFIL_PATTERNS)
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("=" * 60)
    print("   [RELYCE-AI] BACKEND - Starting Up")
    print("=" * 60)
    print(f"[Startup] - Embedding provider: OpenRouter ({EMBEDDING_MODEL})")

    try:
        initialize_firebase()
        print("[Startup] - Firebase initialized")
    except Exception as e:
        print(f"[Startup] ! Firebase init failed: {e}")

    # Start background agent task workers (prevents long synchronous request blocking)
    try:
        fast_workers = max(1, int(os.getenv("FAST_LANE_WORKERS", "6")))
        heavy_workers = max(1, int(os.getenv("HEAVY_LANE_WORKERS", "2")))
        max_heavy_per_user = max(1, int(os.getenv("MAX_HEAVY_TASKS_PER_USER", "2")))
        max_heavy_per_session = max(0, int(os.getenv("MAX_HEAVY_TASKS_PER_SESSION", "0")))
        await get_task_queue().start(
            fast_workers=fast_workers,
            heavy_workers=heavy_workers,
            max_heavy_tasks_per_user=max_heavy_per_user,
            max_heavy_tasks_per_session=max_heavy_per_session,
        )
        print(
            f"[Startup] - Agent task queue started "
            f"(fast={fast_workers}, heavy={heavy_workers}, "
            f"heavy/user={max_heavy_per_user}, heavy/session={max_heavy_per_session or 'off'})"
        )
    except Exception as e:
        print(f"[Startup] ! Agent task queue start failed: {e}")

    # Load intent embeddings into memory (for fast routing) in the background
    try:
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

    # API WARMUP: Pre-warm OpenRouter TLS/DNS pools to reduce cold start latency
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

    # WEAVIATE WARMUP: Pre-connect to Weaviate Cloud for vector memory
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
    try:
        await get_task_queue().shutdown()
        print("[Shutdown] - Agent task queue stopped")
    except Exception as e:
        print(f"[Shutdown] ! Agent task queue shutdown failed: {e}")
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


@app.middleware("http")
async def limit_request_body_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body too large (max {MAX_REQUEST_BODY_BYTES} bytes)"},
                )
        except Exception:
            pass
    return await call_next(request)




@app.middleware("http")
async def input_security_filter(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            body = await request.body()
            if _is_blocked_prompt_exfil_attempt(
                path=request.url.path,
                content_type=request.headers.get("content-type", ""),
                body=body,
            ):
                print(f"[Security] Blocked suspicious input on {request.url.path}")
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Request blocked by security policy"},
                )
        except Exception as e:
            print(f"[Security] Input filter error type={e.__class__.__name__}")
    return await call_next(request)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    body_hash = hashlib.sha256(body).hexdigest()[:16] if body else "empty"
    print(f"[Validation Error] 422 Unprocessable Entity - body_sha256={body_hash} size={len(body)}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request payload"},
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[Unhandled Error] path={request.url.path} type={exc.__class__.__name__}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.add_exception_handler(Exception, unhandled_exception_handler)


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


# Payment Router
app.include_router(memory_router)
app.include_router(system_router)
app.include_router(account_router)
app.include_router(payment_router, prefix="/payment", tags=["Payment"])
app.include_router(chat_runtime_router)
app.include_router(files_router, prefix="/files", tags=["Files"])

# Admin & User Management
from app.routers import admin, users
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(ws_runtime_router)


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
