"""
Response Critic — Lightweight post-generation quality check.
Only triggers for complex/risky sub-intents (code, architecture, research).
Uses fast model (qwen3.5-flash), ~120-250ms added latency.

Rules:
  - Never runs more than once (no recursive loops)
  - Only triggers for flagged sub-intents
  - Short responses (<200 chars) skip critic
  - Critic output is short (max 600 tokens for corrections)
"""
from typing import Optional

from app.config import MEMORY_EXTRACTION_MODEL  # Using same fast model


# Sub-intents that warrant critic review
CRITIC_SUB_INTENTS = {
    "code_generation",
    "code_review",
    "debugging",
    "system_design",
    "analysis",
    "research",
    "reasoning",
}

# Minimum response length to trigger critic (~50 tokens)
MIN_RESPONSE_LENGTH = 200

CRITIC_PROMPT = """You are reviewing an AI assistant's response for quality.

User question:
{question}

AI response:
{response}

Evaluate:
1. Is the answer logically correct?
2. Are critical steps or details missing?
3. Does it contain hallucinated information (fake APIs, wrong syntax, impossible configs)?

If the answer is correct and complete, respond with exactly:
APPROVED

If problems exist, provide ONLY the corrected version of the response. Do not explain what was wrong — just output the fixed response.

Keep corrections concise."""


def needs_critic(sub_intent: str, response_text: str) -> bool:
    """
    Determine if a response should be reviewed by the critic.
    Only triggers for complex/risky sub-intents with substantial responses.
    """
    if sub_intent not in CRITIC_SUB_INTENTS:
        return False

    if len(response_text) < MIN_RESPONSE_LENGTH:
        return False

    return True


async def run_critic(
    user_query: str,
    response_text: str,
    timeout: float = 2.0,
) -> Optional[str]:
    """
    Run the critic on a response. Returns:
      - None if APPROVED (no changes needed)
      - Corrected text if issues found
    
    Uses fast model with strict timeout.
    Never raises — returns None on any failure.
    """
    import asyncio

    try:
        from app.llm.router import get_openrouter_client
        client = get_openrouter_client()

        prompt = CRITIC_PROMPT.format(
            question=user_query[:300],
            response=response_text[:2000],  # Cap to save tokens
        )

        async def _call_critic():
            result = await client.chat.completions.create(
                model=MEMORY_EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1,
            )
            return (result.choices[0].message.content or "").strip()

        raw = await asyncio.wait_for(_call_critic(), timeout=timeout)

        if not raw:
            return None

        # Check if approved
        if raw.upper().startswith("APPROVED"):
            print(f"[Critic] Response APPROVED ({len(response_text)} chars)")
            return None

        # Critic provided correction
        print(f"[Critic] Response CORRECTED ({len(response_text)} → {len(raw)} chars)")
        return raw

    except asyncio.TimeoutError:
        print(f"[Critic] Timed out ({timeout}s), using original response")
        return None
    except Exception as e:
        print(f"[Critic] Failed (non-blocking): {e}")
        return None
