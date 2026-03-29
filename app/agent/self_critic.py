"""
Self Critic
Pre-finalization quality review for TASK and RESEARCH responses.

Runs a focused LLM call to check the produced answer for:
  - Completeness (does it fully answer the query?)
  - Logical consistency
  - Missing critical information

If issues are found, returns them so the caller can do a repair pass.
Only triggered for TASK/RESEARCH action types to avoid latency on simple questions.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from app.llm.prompts import SELF_CRITIC_SYSTEM_PROMPT, SELF_CRITIC_REPAIR_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# ============================================
# CRITIC PROMPT
# ============================================


# ============================================
# ACTION TYPES ELIGIBLE FOR CRITIC
# ============================================

CRITIC_ELIGIBLE_TYPES = {"TASK", "RESEARCH", "ACTION"}


# ============================================
# MAIN CRITIC FUNCTION
# ============================================

async def run_self_critic(
    answer: str,
    query: str,
    client: Any,
    model: str,
    action_type: str = "TASK",
    max_answer_chars: int = 2000,
) -> Dict:
    """
    Run the self-critic on a completed answer.

    Returns:
        {"pass": True}  - answer is good, no changes needed.
        {"pass": False, "issues": [...]}  - answer has problems.
        {"pass": True, "skipped": True}   - critic was bypassed (wrong action type).

    Args:
        answer: The LLM-generated answer text to review.
        query: The original user query.
        client: The async LLM client.
        model: Model identifier to use for the critic call.
        action_type: If not in CRITIC_ELIGIBLE_TYPES, skips the critic.
        max_answer_chars: Truncate the answer to avoid excessive tokens.
    """
    if action_type not in CRITIC_ELIGIBLE_TYPES:
        return {"pass": True, "skipped": True}

    if not client:
        return {"pass": True, "skipped": True, "reason": "no client"}

    # Truncate answer to control token usage
    answer_preview = answer[:max_answer_chars]
    if len(answer) > max_answer_chars:
        answer_preview += "\n[...truncated for review...]"

    user_content = (
        f"Query: {query}\n\n"
        f"Answer:\n{answer_preview}"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SELF_CRITIC_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)

        passed = bool(result.get("pass", True))
        issues = result.get("issues", [])

        if passed:
            logger.debug("[Self-Critic] Answer passed review.")
        else:
            logger.info(f"[Self-Critic] Issues found: {issues}")

        return result

    except Exception as e:
        logger.warning(f"[Self-Critic] Review failed: {e}. Passing through.")
        return {"pass": True, "skipped": True, "reason": str(e)}


# ============================================
# REPAIR PROMPT BUILDER
# ============================================

def build_critic_repair_prompt(issues: list, query: str) -> str:
    """
    Build a concise repair instruction to inject into the LLM for a correction pass.
    """
    issues_text = "\n".join(f"- {issue}" for issue in issues)
    return SELF_CRITIC_REPAIR_PROMPT_TEMPLATE.format(issues_text=issues_text, query=query)

