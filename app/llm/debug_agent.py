"""
Relyce AI - Autonomous Debug Agent
Multi-step reasoning pipeline for when users are stuck on debugging.

Architecture:
- Activates when frustration is high + intent is debugging
- Injects a systematic debugging system prompt
- Structures the AI's response as: Hypothesize → Analyze → Experiment → Solve
- Makes the AI feel like a pair-programming partner, not a chatbot
"""
from typing import Dict, List, Optional
from app.llm.emotion_engine import EmotionalState


class DebugAgent:
    """Autonomous debugging assistant that structures multi-step reasoning."""

    # Activation thresholds
    FRUSTRATION_THRESHOLD = 0.5
    CONFUSION_THRESHOLD = 0.3
    QUALIFYING_INTENTS = {"debugging", "coding_complex", "system_design"}

    def should_activate(
        self,
        emotions: List[str],
        state: EmotionalState,
        sub_intent: str
    ) -> bool:
        """
        Determine if debug mode should activate.
        Triggers when: frustration is high + intent is debug-related + some confusion present.
        """
        if sub_intent not in self.QUALIFYING_INTENTS:
            return False

        # Primary trigger: high frustration during debugging
        if state.frustration > self.FRUSTRATION_THRESHOLD and sub_intent == "debugging":
            return True

        # Secondary trigger: repeated confusion during complex coding
        if (state.confusion > self.CONFUSION_THRESHOLD and
            state.frustration > 0.3 and
            sub_intent in ("debugging", "coding_complex")):
            return True

        # Tertiary: explicit frustration keywords in recent emotions
        if "frustrated" in emotions and sub_intent == "debugging":
            return True

        return False

    def get_debug_system_prompt(self, sub_intent: str = "debugging") -> str:
        """
        Generate the special system prompt for autonomous debugging mode.
        Structures the AI to think like a senior engineer pair programming.
        """
        if sub_intent == "system_design":
            return self._get_design_debug_prompt()

        return """**🔍 AUTONOMOUS DEBUGGING MODE ACTIVE**

You are now acting as an expert debugging partner. The user is stuck and possibly frustrated.
Follow this EXACT systematic approach:

## Step 1: UNDERSTAND
Restate the user's problem in 1-2 sentences to confirm understanding. If anything is unclear, ask ONE targeted clarifying question.

## Step 2: HYPOTHESIZE
List 2-3 most likely root causes, ordered by probability:
- **Most likely:** [cause] — because [evidence from their description]
- **Also possible:** [cause] — because [reasoning]

## Step 3: DIAGNOSE
For the most likely cause, explain:
- What specific code/config to check
- What output/behavior would confirm this hypothesis
- Quick diagnostic commands or code snippets they can run

## Step 4: FIX
Provide the concrete fix:
- Show the exact code changes needed
- Explain WHY this fixes the root cause
- Include any edge cases to watch for

## Step 5: PREVENT
One sentence on how to prevent this type of bug in the future.

**IMPORTANT RULES:**
- Be direct and action-oriented. No fluff.
- If you suspect multiple issues, address the most likely one first.
- Use code blocks with exact file paths when possible.
- Make the user feel like they have a senior engineer sitting next to them."""

    def _get_design_debug_prompt(self) -> str:
        """System prompt for debugging system design issues."""
        return """**🏗️ DESIGN DEBUGGING MODE ACTIVE**

The user is struggling with a system design problem. Act as a principal engineer reviewing their architecture.

## Your Approach:
1. **Identify the constraint** they're hitting (scalability? complexity? cost?)
2. **Challenge assumptions** — what are they assuming that might be wrong?
3. **Propose 2 alternatives** with clear trade-offs
4. **Recommend one** with concrete reasoning
5. **Show the path** — specific next steps to implement

Be opinionated. Senior engineers have opinions. Share yours with reasoning."""

    def get_info_message(self) -> str:
        """Status message to send to the user when debug mode activates."""
        return "[INFO] 🔍 Entering deep debugging mode — analyzing systematically..."


# Singleton
debug_agent = DebugAgent()
