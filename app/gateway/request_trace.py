"""
Request Trace — Per-request structured logging for debugging.
Captures all subsystem decisions, latencies, and outcomes.
"""
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class RequestTrace:
    """Structured trace for a single request pipeline."""
    trace_id: str = ""
    user_id: str = ""
    query_preview: str = ""

    # Routing decisions
    sub_intent: str = ""
    model_selected: str = ""
    model_tier: str = ""

    # Context layers
    active_layers: str = ""
    vector_memories: int = 0
    graph_triples: int = 0
    has_web_content: bool = False
    web_urls_fetched: int = 0
    chunks_analyzed: int = 0

    # Subsystems used
    planner_used: bool = False
    critic_used: bool = False
    critic_result: str = ""  # "approved", "corrected", "skipped", "timeout"

    # Budget
    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0

    # Timing
    start_time: float = field(default_factory=time.time)
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0

    def finalize(self):
        """Call when request completes to record final latency."""
        self.total_latency_ms = (time.time() - self.start_time) * 1000

    def log(self):
        """Print structured trace to stdout."""
        self.finalize()
        print(
            f"[TRACE:{self.trace_id[:8]}] "
            f"intent={self.sub_intent} "
            f"model={self.model_tier} "
            f"mem={self.vector_memories} "
            f"graph={self.graph_triples} "
            f"web={self.has_web_content} "
            f"critic={self.critic_result or 'none'} "
            f"llm_calls={self.llm_calls} "
            f"tokens={self.total_tokens} "
            f"latency={self.total_latency_ms:.0f}ms"
        )

    def to_dict(self) -> Dict:
        """Serialize for structured logging/monitoring."""
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "sub_intent": self.sub_intent,
            "model": self.model_selected,
            "model_tier": self.model_tier,
            "vector_memories": self.vector_memories,
            "graph_triples": self.graph_triples,
            "web_content": self.has_web_content,
            "web_urls": self.web_urls_fetched,
            "chunks": self.chunks_analyzed,
            "planner": self.planner_used,
            "critic": self.critic_result or "none",
            "llm_calls": self.llm_calls,
            "tokens": self.total_tokens,
            "ttft_ms": round(self.ttft_ms, 1),
            "latency_ms": round(self.total_latency_ms, 1),
        }
