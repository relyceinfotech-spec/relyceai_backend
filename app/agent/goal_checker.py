"""
Goal Checker
Uses LLM to determine if the agent's goal is satisfied by the current observations.
"""
from typing import List
import json
from app.llm.prompts import GOAL_CHECKER_SYSTEM_PROMPT
from app.config import FAST_MODEL


class GoalChecker:
    def __init__(self, client, model_to_use: str = FAST_MODEL):
        self.client = client
        self.model_to_use = model_to_use

    async def is_goal_satisfied(
        self,
        goal: str,
        observations: List[str],
        source_count: int = 0,
        min_sources: int = 2,
        min_findings: int = 2,
    ) -> bool:
        """
        Query the LLM to check if the goal is satisfied, with hard evidence thresholds.
        """
        if len(observations) < min_findings:
            print(
                f"[GoalChecker] Satisfied: False | Reason: insufficient findings "
                f"({len(observations)}/{min_findings})"
            )
            return False

        if source_count < min_sources:
            print(
                f"[GoalChecker] Satisfied: False | Reason: insufficient sources "
                f"({source_count}/{min_sources})"
            )
            return False
        system_prompt = GOAL_CHECKER_SYSTEM_PROMPT

        user_content = (
            f"INITIAL GOAL: {goal}\n\n"
            f"OBSERVED FINDINGS:\n{chr(10).join(observations)}\n\n"
            "Has the goal been fully met? Consider if the user's intent is satisfied."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            satisfied = bool(data.get("satisfied", False))
            reason = data.get("reason", "No reason provided.")

            print(f"[GoalChecker] Satisfied: {satisfied} | Reason: {reason}")
            return satisfied
        except Exception as e:
            print(f"[GoalChecker] LLM call failed: {e}. Defaulting to NOT satisfied.")
            return False


