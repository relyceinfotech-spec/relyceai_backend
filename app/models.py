"""
Relyce AI - Pydantic Models
Request/Response models for API endpoints
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict
from datetime import datetime
import uuid

# Chat Modes
ChatMode = Literal["normal", "business", "deepsearch"]

# Content Mode for Personas - Controls how queries are processed
ContentMode = Literal["hybrid", "web_search", "llm_only"]

class Personality(BaseModel):
    """Chat Personality configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    prompt: str
    is_default: bool = False
    is_system: bool = False  # System personas (like Relyce AI) cannot be edited
    content_mode: ContentMode = "hybrid"  # Controls query processing behavior

class ChatMessage(BaseModel):
    """Single chat message"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    """REST API chat request"""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    chat_mode: ChatMode = "normal"
    file_ids: List[str] = Field(default_factory=list)
    personality_id: Optional[str] = None
    personality: Optional[Personality] = None
    user_settings: Optional[Dict] = None

class ChatResponse(BaseModel):
    """REST API chat response"""
    success: bool
    response: Optional[str] = None
    message_id: Optional[str] = None
    error: Optional[str] = None
    mode_used: Optional[str] = None
    tools_activated: Optional[List[str]] = None

class StreamChunk(BaseModel):
    """Streaming response chunk"""
    type: Literal["token", "done", "error", "info"]
    content: str
    metadata: Optional[dict] = None

class WebSocketMessage(BaseModel):
    """WebSocket message frame"""
    type: Literal["message", "stop", "ping", "pong", "auth"]
    content: Optional[str] = None
    chat_id: Optional[str] = None
    chat_mode: ChatMode = "normal"
    token: Optional[str] = None

class SearchRequest(BaseModel):
    """Web search request"""
    query: str
    tools: Optional[List[str]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
