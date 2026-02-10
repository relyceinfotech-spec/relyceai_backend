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
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_MODEL = "google/gemini-2.5-flash-lite:nitro"
CODING_MODEL = os.getenv("CODING_MODEL", "z-ai/glm-4.7-flash")
ERNIE_THINKING_MODEL = "baidu/ernie-4.5-21b-a3b-thinking"

# Embedding Configuration (via OpenRouter)
EMBEDDING_MODEL = "openai/text-embedding-3-small"
TIEBREAKER_MODEL = "openai/gpt-4o-mini"
EMBEDDING_THRESHOLD_HIGH = 0.75
EMBEDDING_THRESHOLD_MEDIUM = 0.60
EMBEDDING_TIMEOUT_MS = 300

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
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))

# CORS Origins (add your frontend URLs)
CORS_ORIGINS = [
    "https://relyceai.com",
    "https://www.relyceai.com",
    "http://localhost:5173",      # Vite dev server
    "http://localhost:3000",      # Alternative dev port
    "http://127.0.0.1:5173",      # Vite dev server (IP)
    "http://127.0.0.1:3000",      # Alternative dev port (IP)
]

# Regex pattern for Vercel preview deployments
CORS_ORIGIN_REGEX = r"https://.*\.vercel\.app"

# Serper Tool Endpoints
SERPER_TOOLS = {
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
