"""
Request Guard — Budget guardrails and global timeout for every request.
Prevents runaway loops, tool explosions, and latency cascades.

Limits:
  MAX_LLM_CALLS = 6
  MAX_TOOL_CALLS = 4
  MAX_REQUEST_TIME = 15s
"""
import time
from dataclasses import dataclass, field


MAX_LLM_CALLS = 6
MAX_TOOL_CALLS = 4
MAX_REQUEST_TIME = 15.0  # seconds


@dataclass
class RequestBudget:
    """Track resource usage for a single request."""
    start_time: float = field(default_factory=time.time)
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    # Subsystem tracking
    web_fetch_used: bool = False
    planner_used: bool = False
    critic_used: bool = False
    chunker_used: bool = False

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def timed_out(self) -> bool:
        return self.elapsed > MAX_REQUEST_TIME

    def can_call_llm(self) -> bool:
        return self.llm_calls < MAX_LLM_CALLS and not self.timed_out

    def can_call_tool(self) -> bool:
        return self.tool_calls < MAX_TOOL_CALLS and not self.timed_out

    def record_llm_call(self, tokens_in: int = 0, tokens_out: int = 0):
        self.llm_calls += 1
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    def record_tool_call(self):
        self.tool_calls += 1

    def should_skip_planner(self) -> bool:
        """Skip planner if budget is tight."""
        return self.elapsed > MAX_REQUEST_TIME * 0.5 or self.llm_calls >= MAX_LLM_CALLS - 1

    def should_skip_critic(self) -> bool:
        """Skip critic if budget is tight."""
        return self.elapsed > MAX_REQUEST_TIME * 0.7 or self.llm_calls >= MAX_LLM_CALLS

    def summary(self) -> str:
        return (
            f"llm={self.llm_calls}/{MAX_LLM_CALLS} "
            f"tool={self.tool_calls}/{MAX_TOOL_CALLS} "
            f"time={self.elapsed:.1f}s/{MAX_REQUEST_TIME}s"
        )
