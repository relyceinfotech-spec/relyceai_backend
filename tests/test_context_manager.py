from app.llm.context_manager import build_segmented_context, estimate_tokens


def _naive_tokens(system_prompt: str, context_messages, user_query: str) -> int:
    total = estimate_tokens(system_prompt) + estimate_tokens(user_query)
    for m in context_messages:
        total += estimate_tokens(str(m.get("content", "")))
    return total


def test_segmented_context_reduces_token_load_with_long_history_and_tools():
    system_prompt = "S" * 1800
    context_messages = []
    for i in range(24):
        role = "user" if i % 2 == 0 else "assistant"
        context_messages.append({"role": role, "content": ("message_" + str(i) + " ") * 180})

    tool_results = [{"tool": "search_web", "status": "success", "data": "X" * 2600} for _ in range(6)]
    user_query = "Analyze top competitors and extract pricing tiers"

    naive = _naive_tokens(system_prompt, context_messages, user_query)
    out = build_segmented_context(
        system_prompt=system_prompt,
        user_query=user_query,
        context_messages=context_messages,
        tool_results=tool_results,
        max_recent_messages=6,
        max_total_estimated_tokens=4500,
    )

    assert out.estimated_tokens < naive
    assert out.compressed_tool_items <= 3
    assert out.dropped_messages >= 1


def test_segmented_context_preserves_user_query_and_system_prompt():
    out = build_segmented_context(
        system_prompt="You are a reliable assistant.",
        user_query="What is the latest CPI reading?",
        context_messages=[{"role": "user", "content": "Earlier question"}],
    )

    assert out.messages[0]["role"] == "system"
    assert "reliable assistant" in out.messages[0]["content"]
    assert out.messages[-1]["role"] == "user"
    assert "latest CPI" in out.messages[-1]["content"]
