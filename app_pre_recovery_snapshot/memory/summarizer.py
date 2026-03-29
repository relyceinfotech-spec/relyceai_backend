"""
Conversation Summarization Service.
Compresses older chat messages into a dense context block to save tokens.
"""
from typing import List, Dict, Any

SUMMARY_PROMPT = """
You are compressing a conversation history.

Previous summary:
{previous_summary}

New messages to digest:
[See below]

Update the summary.

Rules:
- preserve user goals
- preserve technical details and architecture choices
- preserve unresolved questions
- remove greetings, small talk, and filler
- maximum 150 tokens
"""

async def generate_summary(llm_client: Any, messages: List[Dict[str, str]], previous_summary: str = "") -> str:
    """
    Generate a highly compressed summary of the provided conversation block.
    """
    if not messages:
        return ""

    formatted_prompt = SUMMARY_PROMPT.format(previous_summary=previous_summary if previous_summary else "None")

    summary_input = [
        {"role": "system", "content": formatted_prompt}
    ]
    summary_input.extend(messages)

    try:
        response = await llm_client.chat.completions.create(
            # Using a fast, cheap model for summarization tasks
            model="qwen/qwen3.5-flash-02-23",
            messages=summary_input,
            max_tokens=200,
            temperature=0.2,
        )
        summary = response.choices[0].message.content or ""
        
        # Enforce hard length cap to prevent prompt inflation
        if len(summary) > 800:
            summary = summary[:800]
            
        return summary
    except Exception as e:
        print(f"[Summarizer] Error generating summary: {e}")
        return ""
