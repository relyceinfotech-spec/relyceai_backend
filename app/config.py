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
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Firebase Configuration
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_PRIVATE_KEY_ID = os.getenv("FIREBASE_PRIVATE_KEY_ID")
FIREBASE_PRIVATE_KEY = os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n")
FIREBASE_CLIENT_EMAIL = os.getenv("FIREBASE_CLIENT_EMAIL")
FIREBASE_CLIENT_ID = os.getenv("FIREBASE_CLIENT_ID")
FIREBASE_CLIENT_CERT_URL = os.getenv("FIREBASE_CLIENT_CERT_URL")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))

# CORS Origins (add your frontend URLs)
CORS_ORIGINS = [
    "https://relyceai.com",
    "https://www.relyceai.com",
    "https://relyceai-frontend.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
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
