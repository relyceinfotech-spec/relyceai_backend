"""
Cost Tracker — Per-request and aggregate LLM cost tracking.
Uses Qwen pricing from OpenRouter.
"""
import threading
from typing import Optional, Dict
from dataclasses import dataclass


# OpenRouter Qwen pricing (USD per 1M tokens)
PRICING = {
    "qwen/qwen3-235b-a22b": {"prompt": 0.20, "completion": 0.60},
    "qwen/qwen3-30b-a3b": {"prompt": 0.10, "completion": 0.30},
    "qwen/qwen3.5-coder-32b": {"prompt": 0.10, "completion": 0.30},
    "qwen/qwen-2.5-coder-32b-instruct": {"prompt": 0.10, "completion": 0.30},
    "qwen/qwen3-32b": {"prompt": 0.10, "completion": 0.30},
    "deepseek/deepseek-chat-v3-0324": {"prompt": 0.30, "completion": 0.90},
    "qwen/qwen-2.5-32b-instruct": {"prompt": 0.10, "completion": 0.30},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"prompt": 0.20, "completion": 0.60}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD for a single LLM call."""
    pricing = PRICING.get(model, DEFAULT_PRICING)
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    return prompt_cost + completion_cost


class CostTracker:
    """Aggregate cost tracking across all requests."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.total_requests: int = 0
        self.total_llm_calls: int = 0
        self._per_model: Dict[str, Dict] = {}

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Record a single LLM call."""
        cost = estimate_cost(model, prompt_tokens, completion_tokens)

        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_usd += cost
            self.total_llm_calls += 1

            if model not in self._per_model:
                self._per_model[model] = {
                    "calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "cost_usd": 0.0,
                }
            self._per_model[model]["calls"] += 1
            self._per_model[model]["prompt_tokens"] += prompt_tokens
            self._per_model[model]["completion_tokens"] += completion_tokens
            self._per_model[model]["cost_usd"] += cost

    def record_request(self):
        """Mark a request as processed."""
        with self._lock:
            self.total_requests += 1

    def get_metrics(self) -> dict:
        """Return cost snapshot."""
        with self._lock:
            avg_tokens = (
                (self.total_prompt_tokens + self.total_completion_tokens)
                / self.total_requests
                if self.total_requests else 0
            )
            avg_cost = (
                self.total_cost_usd / self.total_requests
                if self.total_requests else 0.0
            )
            ces = (
                (self.total_prompt_tokens + self.total_completion_tokens)
                / self.total_completion_tokens
                if self.total_completion_tokens else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "total_llm_calls": self.total_llm_calls,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "avg_tokens_per_request": round(avg_tokens, 1),
                "avg_cost_per_request": round(avg_cost, 6),
                "cost_efficiency_score": round(ces, 2),
                "per_model": dict(self._per_model),
            }

    def reset(self):
        with self._lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_cost_usd = 0.0
            self.total_requests = 0
            self.total_llm_calls = 0
            self._per_model.clear()


_instance: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    global _instance
    if _instance is None:
        _instance = CostTracker()
    return _instance
