# Relyce AI Backend

Production-grade ChatGPT-style backend with multi-device support.

## Features

- ✅ FastAPI REST API
- ✅ WebSocket real-time chat
- ✅ Multi-device support (same chat on multiple devices)
- ✅ Firebase Auth integration
- ✅ Streaming responses
- ✅ Three chat modes: Normal, Business, DeepSearch
- ✅ Serper API integration for web search

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment

Make sure your `.env` file has:

```env
OPENAI_API_KEY=your_key
SERPER_API_KEY=your_key
LLM_MODEL=gpt-4o

# Firebase (already configured)
FIREBASE_PROJECT_ID=...
FIREBASE_PRIVATE_KEY=...
FIREBASE_CLIENT_EMAIL=...
```

### 3. Run the Server

```bash
# From backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or directly
python -m app.main
```

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Send message (non-streaming) |
| `/chat/stream` | POST | Send message (SSE streaming) |
| `/search` | POST | Web search |
| `/history/{user_id}/{session_id}` | GET | Get chat history |

### WebSocket

Connect to: `ws://localhost:8000/ws/chat?token=FIREBASE_TOKEN&chat_id=SESSION_ID`

**Message Types:**

```json
// Send message
{"type": "message", "content": "Hello", "chat_mode": "normal"}

// Stop generation
{"type": "stop"}

// Ping
{"type": "ping"}
```

**Response Types:**

```json
// Streaming token
{"type": "token", "content": "..."}

// Generation complete
{"type": "done", "content": ""}

// Error
{"type": "error", "content": "..."}
```

## Architecture

```
backend/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── config.py        # Configuration
│   ├── models.py        # Pydantic models
│   ├── auth.py          # Firebase auth
│   ├── websocket.py     # WebSocket manager
│   ├── llm/
│   │   ├── router.py    # Intent routing
│   │   └── processor.py # LLM processing
│   └── chat/
│       ├── context.py   # Context management
│       └── history.py   # Firebase history
├── .env
└── requirements.txt
```

## Multi-Device Support

The architecture supports multiple devices connecting to the same chat:

```
User
├── Mobile (WebSocket 1)  ─┐
├── Laptop (WebSocket 2)  ─┼──► Same chat_id ──► Synced responses
└── Tablet (WebSocket 3)  ─┘
```

All devices receive the same streaming response in real-time!
