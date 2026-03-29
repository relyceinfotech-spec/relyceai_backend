"""
Agent State
Centralised state container for a single agent execution turn.

Replaces the scattered local variables in processor.py with a single
object that travels through the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import time
from enum import Enum


# ============================================
# OBSERVABILITY — Structured Tracing
# ============================================

@dataclass
class TraceEvent:
    """A single measurable event in the agent's execution."""
    trace_id: str
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0
    node_id: Optional[str] = None
    action: str = ""  # e.g. "PLANNING", "TOOL_CALL", "SYNTHESIS"
    tool: Optional[str] = None
    latency_ms: float = 0.0
    result_status: str = "success"  # "success", "failure", "skipped"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================
# WORKSPACE — Intermediate Cognition
# ============================================

@dataclass
class Workspace:
    """
    The agent's scratchpad for a single goal.
    Helps maintain focus and avoid repetitive discovery.
    """
    goal: str = ""
    knowledge: List[str] = field(default_factory=list)      # Raw findings discovered
    facts: List[Dict[str, Any]] = field(default_factory=list)  # {fact, source, confidence}
    sources: List[Dict[str, Any]] = field(default_factory=list) # {url, domain, trust_score, snippet}
    claims: List[Dict[str, Any]] = field(default_factory=list)  # {claim, confidence}
    claim_sources: Dict[str, List[str]] = field(default_factory=dict)  # claim -> [source_url]
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    contradictions: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    progress_summary: str = "Starting..."


# ============================================
# BUDGET — Hard Execution Limits
# ============================================

@dataclass
class AgentBudget:
    """
    Hard per-turn limits that prevent runaway agent loops.

    When any limit is exceeded the pipeline must finalize immediately
    using whatever data it has accumulated so far.
    """
    max_steps: int = 8
    max_tool_calls: int = 12
    max_llm_calls: int = 30
    max_iterations: int = 10

    # Runtime counters (mutated during execution)
    steps_used: int = 0
    tool_calls_used: int = 0
    llm_calls_used: int = 0
    iterations_used: int = 0

    def is_exceeded(self) -> bool:
        """Return True if ANY limit has been hit."""
        return (
            self.steps_used >= self.max_steps
            or self.tool_calls_used >= self.max_tool_calls
            or self.llm_calls_used >= self.max_llm_calls
            or self.iterations_used >= self.max_iterations
        )

    def increment_step(self) -> None:
        self.steps_used += 1

    def increment_tool(self) -> None:
        self.tool_calls_used += 1

    def increment_llm(self) -> None:
        self.llm_calls_used += 1

    def increment_iteration(self) -> None:
        self.iterations_used += 1

    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps_used)


    # Backward-compat alias used by legacy processor paths.
    @property
    def max_total_tool_calls(self) -> int:
        return self.max_tool_calls

    @max_total_tool_calls.setter
    def max_total_tool_calls(self, value: int) -> None:
        self.max_tool_calls = int(value)
    def summary(self) -> Dict:
        return {
            "steps": f"{self.steps_used}/{self.max_steps}",
            "tool_calls": f"{self.tool_calls_used}/{self.max_tool_calls}",
            "llm_calls": f"{self.llm_calls_used}/{self.max_llm_calls}",
            "iterations": f"{self.iterations_used}/{self.max_iterations}",
        }


# ============================================
# AGENT STATE — Single Turn Container
# ============================================

@dataclass
class AgentState:
    """
    All state for one agent execution turn.

    Passed through the pipeline instead of scattered local variables.
    Mutable — each stage writes to it.
    """
    # Core inputs
    query: str = ""
    goal: str = ""
    intent: str = ""
    sub_intent: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: str = field(default_factory=lambda: "tr_" + str(uuid.uuid4())[:8])
    mode: str = "agent"

    # Plan
    plan: Optional[Any] = None          # PlanGraph  (typed as Any to avoid circular imports)
    plan_confidence: float = 1.0
    plan_history: List[Any] = field(default_factory=list)

    # Tool tracking
    tool_results: List[Dict] = field(default_factory=list)
    tool_last_called: Dict[str, float] = field(default_factory=dict)  # tool_name -> unix ts

    # Execution
    step_count: int = 0
    iteration: int = 0
    completed_steps: List[Dict] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    forced_finalize: bool = False
    errors: List[str] = field(default_factory=list)
    terminate: bool = False

    # Memory / Context
    memory: Dict = field(default_factory=dict)
    context_messages: List[Dict] = field(default_factory=list)

    # Execution budget
    budget: AgentBudget = field(default_factory=AgentBudget)

    # Production Scaling (Phase 6)
    workspace: Workspace = field(default_factory=Workspace)
    traces: List[TraceEvent] = field(default_factory=list)

    # Execution context (from tool_executor)
    execution_context: Optional[Any] = None

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    def record_tool_call(self, tool_name: str) -> None:
        """Mark that a tool was just invoked."""
        import time
        self.tool_last_called[tool_name] = time.time()
        self.budget.increment_tool()

    def record_step(self) -> None:
        self.budget.increment_step()
        self.step_count += 1

    def record_llm_call(self) -> None:
        self.budget.increment_llm()

    def record_iteration(self) -> None:
        self.budget.increment_iteration()
        self.iteration += 1

    def add_trace(self, action: str, node_id: Optional[str] = None, tool: Optional[str] = None, latency_ms: float = 0.0, status: str = "success", metadata: Optional[Dict] = None) -> None:
        """Helper to append a structured trace event."""
        event = TraceEvent(
            trace_id=self.trace_id,
            iteration=self.iteration,
            node_id=node_id,
            action=action,
            tool=tool,
            latency_ms=latency_ms,
            result_status=status,
            metadata=metadata or {}
        )
        self.traces.append(event)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def should_terminate(self) -> bool:
        return self.terminate or self.budget.is_exceeded() or self.forced_finalize

    def to_debug_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query[:80],
            "goal": self.goal[:80],
            "iteration": self.iteration,
            "plan_confidence": self.plan_confidence,
            "tool_results": len(self.tool_results),
            "errors": self.errors,
            "budget": self.budget.summary(),
            "terminated": self.terminate,
        }

