"""
LLM Prompt Context Assembler.
Extracts the session summary and recents to form the optimal request payload.
"""
from typing import List, Dict

# Defines the number of recent chat messages to preserve intact against summarization
RECENT_MESSAGE_LIMIT = 6

def build_context(
    system_prompt: str,
    summary: str,
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Construct optimized prompt context using the memory architecture:
    [SYSTEM]
    [SUMMARY]
    [RECENT_MESSAGES]
    """
    context = []

    # 1. System Overlay
    context.append({
        "role": "system",
        "content": system_prompt
    })

    # 2. Summary Injection
    if summary and summary.strip():
        context.append({
            "role": "system",
            "content": f"Conversation memory (summary of prior events):\n{summary}"
        })

    # 3. Recent Chat Window
    # The `messages` passed here might legitimately just be the total DB results if we simply fetch the last 50, 
    # but the Summary background task continuously deletes anything beyond the RECENT_MESSAGE_LIMIT. 
    # Just to be strictly safe, we isolate the recent slice right here as well.
    recent_messages = messages[-RECENT_MESSAGE_LIMIT:]
    context.extend(recent_messages)

    return context
