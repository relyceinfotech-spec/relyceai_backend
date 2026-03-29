"""
Smart Model Router — Routes tasks to the optimal model based on complexity.
Uses existing config models: FAST_MODEL, CODING_MODEL, UI_MODEL.

Routing rules:
  simple chat     → FAST_MODEL (cheap, fast)
  coding/arch     → CODING_MODEL (strong reasoning)
  UI generation   → UI_MODEL (code-focused)
  long doc        → FAST_MODEL (handles context well)
  extraction/critic → MEMORY_EXTRACTION_MODEL (lightweight)
"""
from app.config import FAST_MODEL, CODING_MODEL, UI_MODEL, MEMORY_EXTRACTION_MODEL


# Sub-intents that need strong reasoning
_REASONING_INTENTS = {
    "code_generation", "code_review", "debugging",
    "system_design", "analysis", "reasoning",
    "architecture", "optimization",
}

# Sub-intents that need UI code generation
_UI_INTENTS = {
    "ui_generation", "web_design", "frontend",
    "html_css", "react_component",
}

# Simple intents that can use cheap model
_SIMPLE_INTENTS = {
    "greeting", "chitchat", "general", "clarification",
    "acknowledgement", "farewell",
}


def select_model(
    sub_intent: str = "general",
    token_estimate: int = 0,
    has_web_content: bool = False,
    is_thinking_pass: bool = False,
) -> str:
    """
    Select optimal model for the task.
    Returns model identifier string for OpenRouter.
    """
    # Thinking/planning pass always uses fast model
    if is_thinking_pass:
        return FAST_MODEL

    # UI generation needs code-focused model
    if sub_intent in _UI_INTENTS:
        return UI_MODEL

    # Complex reasoning needs strong model
    if sub_intent in _REASONING_INTENTS:
        return CODING_MODEL

    # Simple queries use cheap model
    if sub_intent in _SIMPLE_INTENTS:
        return FAST_MODEL

    # Short queries (<150 tokens est.) use fast model
    if token_estimate < 150:
        return FAST_MODEL

    # Default: fast model
    return FAST_MODEL


def get_model_tier(model: str) -> str:
    """Returns cost tier for logging: fast, reasoning, ui."""
    if model == UI_MODEL:
        return "ui"
    if model == CODING_MODEL:
        return "reasoning"
    return "fast"
