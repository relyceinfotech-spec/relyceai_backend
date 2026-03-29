"""
Relyce AI - Pydantic Models
Request/Response models for API endpoints
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime
import uuid
from app.config import MAX_CHAT_MESSAGE_CHARS

# Chat Modes
# Canonical: smart, agent, research_pro
# Legacy values are accepted for backward compatibility and normalized at runtime.
ChatMode = Literal["auto", "smart", "agent", "research_pro", "normal", "business", "deepsearch", "hybrid_main", "research"]

# Content Mode for Personas - Controls how queries are processed
ContentMode = Literal["hybrid", "web_search", "llm_only"]


class Personality(BaseModel):
    """Chat Personality configuration"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    # Optional for backward compatibility with legacy frontend payloads.
    # Runtime will hydrate full personality by id when prompt is missing.
    prompt: Optional[str] = None
    is_default: bool = False
    is_system: bool = False  # System personas (like Relyce AI) cannot be edited
    content_mode: ContentMode = "hybrid"  # Controls query processing behavior
    specialty: str = "general"  # Expertise area: coding, ecommerce, music, legal, etc.


class ChatMessage(BaseModel):
    """Single chat message"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """REST API chat request"""
    message: str = Field(..., min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    chat_mode: ChatMode = "smart"
    file_ids: Optional[List[str]] = None

    personality_id: Optional[str] = None
    personality: Optional[Personality] = None
    user_settings: Optional[Dict] = None


class SourceRef(BaseModel):
    name: str = ""
    url: str
    trust: Optional[float] = None


class StructuredResponse(BaseModel):
    schema_version: str = "1.0"
    summary: str = ""
    answer: str = ""
    key_points: List[str] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_level: Optional[str] = None
    answer_type: str = "summary"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """REST API chat response"""
    success: bool
    response: Optional[str] = None
    message_id: Optional[str] = None
    error: Optional[str] = None
    mode_used: Optional[str] = None
    tools_activated: Optional[List[str]] = None

    # Stable response contract (v1)
    schema_version: str = "1.0"
    summary: str = ""
    answer: str = ""
    key_points: List[str] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_level: Optional[str] = None
    answer_type: str = "summary"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[Dict[str, Any]] = Field(default_factory=list)
    structured_response: Optional[StructuredResponse] = None


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
    chat_mode: ChatMode = "smart"
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
