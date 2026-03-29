"""
Relyce AI - Configuration Module
Centralized configuration using environment variables
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "x-ai/grok-4-fast")

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
FAST_MODEL = "x-ai/grok-4-fast"
CODING_MODEL = os.getenv("CODING_MODEL", "x-ai/grok-4-fast")
UI_MODEL = os.getenv("UI_MODEL", "x-ai/grok-4-fast")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")
ERNIE_THINKING_MODEL = "x-ai/grok-4-fast"

# Embedding Configuration (via OpenRouter)
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TIEBREAKER_MODEL = "openai/gpt-4o-mini"
EMBEDDING_THRESHOLD_HIGH = 0.75
EMBEDDING_THRESHOLD_MEDIUM = 0.60
EMBEDDING_TIMEOUT_MS = 300

# Weaviate Vector Memory Configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
MAX_MEMORIES_PER_USER = int(os.getenv("MAX_MEMORIES_PER_USER", "200"))
MEMORY_DEDUP_THRESHOLD = 0.85  # Cosine similarity threshold for deduplication
MEMORY_EXTRACTION_MODEL = os.getenv("MEMORY_EXTRACTION_MODEL", "x-ai/grok-4-fast")

# Firebase Configuration
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_PRIVATE_KEY_ID = os.getenv("FIREBASE_PRIVATE_KEY_ID")
FIREBASE_PRIVATE_KEY = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")
FIREBASE_CLIENT_EMAIL = os.getenv("FIREBASE_CLIENT_EMAIL")
FIREBASE_CLIENT_ID = os.getenv("FIREBASE_CLIENT_ID")
FIREBASE_CLIENT_CERT_URL = os.getenv("FIREBASE_CLIENT_CERT_URL")
FIREBASE_DATABASE_ID = os.getenv("FIREBASE_DATABASE_ID", "relyceinfotech")

# Razorpay Configuration
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET")

# Server Configuration
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV in {"prod", "production"}
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))

# CORS Origins (add your frontend URLs)
def _csv_env_list(name: str, default: list) -> list:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]

_DEFAULT_CORS_ORIGINS = [
    "https://relyceai.com",
    "https://www.relyceai.com",
    "https://relyceai-frontend.vercel.app",
]
_DEV_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
_DEFAULT_CORS_ORIGINS += ([] if IS_PRODUCTION else _DEV_CORS_ORIGINS)

CORS_ORIGINS = _csv_env_list("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)

# CORS Methods/Headers
_DEFAULT_CORS_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_DEFAULT_CORS_HEADERS = [
    "Authorization",
    "Content-Type",
    "Accept",
    "Origin",
    "X-Requested-With",
]

CORS_ALLOWED_METHODS = _csv_env_list("CORS_ALLOWED_METHODS", _DEFAULT_CORS_METHODS)
CORS_ALLOWED_HEADERS = _csv_env_list("CORS_ALLOWED_HEADERS", _DEFAULT_CORS_HEADERS)

# Regex pattern for preview deployments and subdomains
CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https://(.*\.)?relyceai\.com$"
)

# HTTPS enforcement (recommended to enable in production)
FORCE_HTTPS = os.getenv("FORCE_HTTPS", "true" if IS_PRODUCTION else "false").lower() == "true"

# Chat input limits
MAX_CHAT_MESSAGE_CHARS = int(os.getenv("MAX_CHAT_MESSAGE_CHARS", 6000))
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "30"))

# Upload constraints
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", 25))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
DEFAULT_STORAGE_QUOTA_MB = float(os.getenv("DEFAULT_STORAGE_QUOTA_MB", 500))
UPLOAD_ALLOWED_MIME_TYPES = _csv_env_list(
    "UPLOAD_ALLOWED_MIME_TYPES",
    [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    ],
)
UPLOAD_ALLOWED_EXTENSIONS = _csv_env_list(
    "UPLOAD_ALLOWED_EXTENSIONS",
    [".pdf", ".txt", ".md", ".csv", ".json", ".docx"],
)
UPLOAD_ALLOWED_EXTENSIONS = [ext.lower() for ext in UPLOAD_ALLOWED_EXTENSIONS]

# Rate limit config (Firestore-backed for chat)
# FAIL_OPEN=False means requests are DENIED if rate limiting fails (more secure)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 30))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))
RATE_LIMIT_FAIL_OPEN = os.getenv("RATE_LIMIT_FAIL_OPEN", "false").lower() == "true"

# Agent mode rate limit (stricter — heavier compute per request)
AGENT_RATE_LIMIT_PER_MINUTE = int(os.getenv("AGENT_RATE_LIMIT_PER_MINUTE", 4))

# Serper Tool Endpoints
from urllib.parse import urlparse

_DEFAULT_SERPER_TOOLS = {
    "Search": "https://google.serper.dev/search",
    "Images": "https://google.serper.dev/images",
    "Videos": "https://google.serper.dev/videos",
    "Places": "https://google.serper.dev/places",
    "Maps": "https://google.serper.dev/maps",
    "Reviews": "https://google.serper.dev/reviews",
    "News": "https://google.serper.dev/news",
    "Shopping": "https://google.serper.dev/shopping",
    "Lens": "https://google.serper.dev/lens",
    "Scholar": "https://google.serper.dev/scholar",
    "Patents": "https://google.serper.dev/patents",
    "Webpage": "https://scrape.serper.dev"
}

SERPER_ALLOWED_HOSTS = _csv_env_list(
    "SERPER_ALLOWED_HOSTS",
    ["google.serper.dev", "scrape.serper.dev"],
)

def _validate_serper_tools(tools: dict) -> dict:
    validated = {}
    for name, url in tools.items():
        try:
            parsed = urlparse(url)
            if parsed.scheme != "https":
                print(f"[Config] Serper URL blocked (non-https): {name} -> {url}")
                continue
            if parsed.netloc not in SERPER_ALLOWED_HOSTS:
                print(f"[Config] Serper URL blocked (host not allowed): {name} -> {url}")
                continue
            validated[name] = url
        except Exception as e:
            print(f"[Config] Serper URL invalid: {name} -> {url} ({e})")
    return validated

_SERPER_OVERRIDE = os.getenv("SERPER_TOOLS_OVERRIDE", "").strip()
if _SERPER_OVERRIDE:
    try:
        import json as _json
        override_map = _json.loads(_SERPER_OVERRIDE)
        if isinstance(override_map, dict):
            _validated_override = _validate_serper_tools(override_map)
            SERPER_TOOLS = _validated_override or _validate_serper_tools(_DEFAULT_SERPER_TOOLS)
        else:
            SERPER_TOOLS = _validate_serper_tools(_DEFAULT_SERPER_TOOLS)
    except Exception as e:
        print(f"[Config] Failed to parse SERPER_TOOLS_OVERRIDE: {e}")
        SERPER_TOOLS = _validate_serper_tools(_DEFAULT_SERPER_TOOLS)
else:
    SERPER_TOOLS = _validate_serper_tools(_DEFAULT_SERPER_TOOLS)

# Serper request safety
SERPER_CONNECT_TIMEOUT = float(os.getenv("SERPER_CONNECT_TIMEOUT", 3.05))
SERPER_READ_TIMEOUT = float(os.getenv("SERPER_READ_TIMEOUT", 10))
SERPER_MAX_RETRIES = int(os.getenv("SERPER_MAX_RETRIES", 2))
SERPER_RETRY_BACKOFF = float(os.getenv("SERPER_RETRY_BACKOFF", 0.5))

# WebSocket scaling & limits
MAX_WS_CONNECTIONS_PER_CHAT = int(os.getenv("MAX_WS_CONNECTIONS_PER_CHAT", 8))
MAX_WS_CONNECTIONS_TOTAL = int(os.getenv("MAX_WS_CONNECTIONS_TOTAL", 2000))
REDIS_URL = os.getenv("REDIS_URL")
REDIS_WS_CHANNEL_PREFIX = os.getenv("REDIS_WS_CHANNEL_PREFIX", "ws-chat:")

# Security headers
SECURITY_HEADERS_ENABLED = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"
SECURITY_CSP = os.getenv(
    "SECURITY_CSP",
    "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)

# Chat sanitization
SANITIZE_CHAT_HTML = os.getenv("SANITIZE_CHAT_HTML", "true").lower() == "true"

# Personality prompt validation
MAX_PERSONALITY_PROMPT_CHARS = int(os.getenv("MAX_PERSONALITY_PROMPT_CHARS", 2000))
MAX_PERSONALITY_NAME_CHARS = int(os.getenv("MAX_PERSONALITY_NAME_CHARS", 60))
MAX_PERSONALITY_DESC_CHARS = int(os.getenv("MAX_PERSONALITY_DESC_CHARS", 200))

# Sandbox Execution Limits (Phase 6)
SANDBOX_ENABLED = os.getenv("SANDBOX_ENABLED", "true").lower() == "true"
SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", 10))  # seconds
SANDBOX_MEMORY_LIMIT_MB = int(os.getenv("SANDBOX_MEMORY_LIMIT_MB", 256))
SANDBOX_CPU_SECONDS = int(os.getenv("SANDBOX_CPU_SECONDS", 5))

# Reliability hardening flags (v1 + v2 unified package)
FF_CONFIDENCE_CRITIC_LOOPS = os.getenv("FF_CONFIDENCE_CRITIC_LOOPS", "true").lower() == "true"
FF_TOOL_FIRST_EVIDENCE = os.getenv("FF_TOOL_FIRST_EVIDENCE", "true").lower() == "true"
FF_TOOL_MEMORY = os.getenv("FF_TOOL_MEMORY", "true").lower() == "true"
FF_LLM_INTENT_CLASSIFIER = os.getenv("FF_LLM_INTENT_CLASSIFIER", "true").lower() == "true"
FF_TOOL_CONFIDENCE_SCORING = os.getenv("FF_TOOL_CONFIDENCE_SCORING", "true").lower() == "true"

# Reliability thresholds and loop budgets
CRITIC_CONFIDENCE_MIN = float(os.getenv("CRITIC_CONFIDENCE_MIN", "0.70"))
CRITIC_MAX_REPAIRS = int(os.getenv("CRITIC_MAX_REPAIRS", "2"))
FAST_LANE_RELIABILITY_BUDGET_MS = int(os.getenv("FAST_LANE_RELIABILITY_BUDGET_MS", "15000"))
HEAVY_LANE_RELIABILITY_BUDGET_MS = int(os.getenv("HEAVY_LANE_RELIABILITY_BUDGET_MS", "45000"))
CURRENT_INFO_RECENCY_DAYS = int(os.getenv("CURRENT_INFO_RECENCY_DAYS", "180"))
RELIABILITY_TOOL_RETRY_BACKOFF_MS = os.getenv("RELIABILITY_TOOL_RETRY_BACKOFF_MS", "200,600")

# Tool-memory v2.1 settings
TOOL_SCHEMA_VERSION = os.getenv("TOOL_SCHEMA_VERSION", "v2.1")
TOOL_MEMORY_TTL_SECONDS = int(os.getenv("TOOL_MEMORY_TTL_SECONDS", "86400"))
TOOL_MEMORY_FRESHNESS_MIN = float(os.getenv("TOOL_MEMORY_FRESHNESS_MIN", "0.25"))
TOOL_MEMORY_MAX_ITEMS_PER_USER = int(os.getenv("TOOL_MEMORY_MAX_ITEMS_PER_USER", "200"))
TOOL_MEMORY_MAX_ITEMS_PER_SESSION = int(os.getenv("TOOL_MEMORY_MAX_ITEMS_PER_SESSION", "50"))

# Tool confidence/adaptive planning
TOOL_CONFIDENCE_MIN_SAMPLES = int(os.getenv("TOOL_CONFIDENCE_MIN_SAMPLES", "4"))
