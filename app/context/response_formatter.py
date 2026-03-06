"""
Response Formatter — Dynamic answer formatting based on task type.
Appends task-specific structure instructions to the system prompt.

Pipeline position: after Reasoning Scaffold, before Model Router.

Supported task types:
  coding → explanation + code + example
  architecture → overview + components + recommendation
  research → summary + key findings + conclusion
  debugging → problem + cause + fix
  web_analysis → summary + key findings + conclusion
  general_chat → clear explanation (default)
"""


def format_instruction(task_type: str) -> str:
    """
    Return a formatting instruction for the given task type.
    Injected into system prompt to guide response structure.
    """
    instructions = {
        "coding": (
            "\n\nStructure your response as:\n"
            "1. Brief explanation of the approach\n"
            "2. Code implementation\n"
            "3. Usage example (if applicable)\n"
            "Keep explanations concise. Prioritize working code."
        ),
        "architecture": (
            "\n\nStructure your response as:\n"
            "1. Overview — what the system does\n"
            "2. Components — key parts and their roles\n"
            "3. Recommendation — the suggested approach with trade-offs\n"
            "Use bullet points for components."
        ),
        "research": (
            "\n\nStructure your response as:\n"
            "1. Summary — the core answer in 1-2 sentences\n"
            "2. Key findings — the most important details\n"
            "3. Conclusion — actionable takeaway\n"
            "Be thorough but avoid unnecessary repetition."
        ),
        "debugging": (
            "\n\nStructure your response as:\n"
            "1. Problem — what's happening\n"
            "2. Cause — why it's happening\n"
            "3. Fix — exact steps or code to resolve it\n"
            "Be direct. Skip background unless essential."
        ),
        "web_analysis": (
            "\n\nStructure your response as:\n"
            "1. Summary — main topic in 1-2 sentences\n"
            "2. Key points — the most important details\n"
            "3. Conclusion — what the user should take away"
        ),
        "document_analysis": (
            "\n\nStructure your response as:\n"
            "1. Main arguments or topics identified\n"
            "2. Key details and supporting evidence\n"
            "3. Summary of findings"
        ),
        "creative": (
            "\n\nBe creative and engaging. Use vivid language. "
            "Present ideas in a clear, inspiring format."
        ),
    }

    return instructions.get(task_type, "")
