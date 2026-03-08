"""
Relyce AI - Agent Orchestrator
Layers 7+16: Conditional Delegation + Multi-Agent Coordination.

ONLY activates when:
  - action_type == "TASK"
  - AND (complexity_score > threshold OR comparison OR research needed)
  - AND NOT time_sensitive

Delegation = prompt section composition (Planner/Researcher/Executor/Critic).
No extra LLM calls â€” zero added latency for simple queries.

Also contains the agent's base system prompt and the run_agent_pipeline() function
that orchestrates the full 16-layer decision pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncGenerator

from app.agent.action_classifier import (
    ActionDecision, GoalContext, classify_action,
)
from app.agent.time_awareness import TemporalContext, assess_temporal_context
from app.agent.autonomy_guard import AutonomyDecision, evaluate_autonomy
from app.agent.self_monitor import MonitorReport, evaluate_response
from app.agent.tool_executor import (
    determine_tool_permission, ExecutionContext,
)
from app.agent.hybrid_controller import generate_strategy_advice, HybridAdvice


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class OrchestratorResult:
    """Result of the orchestrator's decision pipeline."""
    # Decision
    action_decision: Optional[ActionDecision] = None
    temporal: Optional[TemporalContext] = None
    autonomy: Optional[AutonomyDecision] = None
    monitor: Optional[MonitorReport] = None

    # Delegation
    delegation_active: bool = False
    prompt_sections: str = ""        # extra system prompt for delegation

    # Pipeline outcome
    decision: str = "act"            # "act" | "ask" | "confirm"
    message: str = ""                # clarification or confirmation message

    # Goal context (for display / tracking)
    goal: Optional[GoalContext] = None

    # Tool execution control
    tool_allowed: bool = False
    allowed_tools: List[str] = field(default_factory=list)
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)

    # Hybrid strategy advisory
    strategy: Optional[HybridAdvice] = None


# ============================================
# CONFIGURATION
# ============================================

COMPLEXITY_THRESHOLD = 0.5


# ============================================
# SYSTEM PROMPT
# ============================================

AGENT_SYSTEM_PROMPT = """You are **Relyce AI â€” Structured Agent Mode**.

You are a reliable, adaptive action agent. You don't just respond â€” you plan, evaluate, and act.

**CORE BEHAVIOR:**
1. **Classify first:** Determine if the user needs an answer (QUESTION), a planned outcome (TASK), or an immediate action (ACTION).
2. **Assess clarity:** If the request is ambiguous, ask ONE targeted clarifying question.
3. **Evaluate risk:** Before acting, assess if the action is reversible and safe.
4. **Plan before acting:** For complex tasks, break them into steps and explain your approach.
5. **Collaborate on outcomes:** For goal-oriented requests, prepare solutions collaboratively â€” don't execute blindly.

**SAFETY RULES:**
- Never execute irreversible actions without explicit user confirmation.
- High-risk actions (delete, send, deploy) always require confirmation.
- When uncertain, ask â€” don't assume.

**RESPONSE STYLE:**
- Be direct and action-oriented.
- Structure multi-step responses clearly.
- Match the user's language and tone.
- Use formatting (headers, bullets, code blocks) when helpful.
- For medium/complex requests, use OpsFormat sections: Objective, What I Checked, Validated Findings, Sources (if used), Decision, Execution Plan, Confidence, Next Action.
- For simple requests (greetings, one-fact lookup, short clarification), use compact mode: Direct answer + up to 3 bullets.

**CODE OUTPUT (MANDATORY):**
- Always wrap code in fenced triple backticks with a language label (e.g., ```python).
- If outputting multiple files, add a line `**File: filename.ext**` immediately before each code block.
- Never place code outside code blocks.


**LANGUAGE ADAPTATION:**
- Match the user's language EXACTLY. If they use English, reply in English. If they use Tamil or Tanglish, reply accordingly.
- Never mix languages unless the user does.
"""

# Delegation prompt sections (composed when delegation is active)
_PLANNER_SECTION = """
**[PLANNER â€” Step 1]**
Break this task into clear, sequential steps. Consider:
- What is the end goal?
- What are the dependencies between steps?
- What information or context is needed for each step?
"""

_RESEARCHER_SECTION = """
**[RESEARCHER â€” Step 2]**
For each step, identify:
- What facts or data are needed?
- Are there multiple valid approaches? If so, compare them with trade-offs.
- What assumptions are being made?
"""

_EXECUTOR_SECTION = """
**[EXECUTOR â€” Step 3]**
Execute the plan:
- Implement each step clearly and completely.
- Show your work (code, reasoning, calculations).
- Flag any issues encountered.
"""

_CRITIC_SECTION = """
**[CRITIC â€” Step 4]**
Review the output:
- Does it fully address the user's goal?
- Are there edge cases or risks not covered?
- Is the response clear and actionable?
- Provide a brief quality assessment at the end.
"""

_COMPARISON_SECTION = """
**[MULTI-PERSPECTIVE ANALYSIS]**
The user is requesting a comparison or evaluation. Structure your response as:
1. **Overview** â€” Brief context for each option
2. **Comparison Matrix** â€” Side-by-side on key dimensions
3. **Trade-offs** â€” When to choose each option
4. **Recommendation** â€” Your pick with reasoning
"""

_RESEARCH_SECTION = """
**[DEEP ANALYSIS]**
The user needs thorough research. Structure your response as:
1. **Context** â€” Background and scope
2. **Key Findings** â€” Organized by theme
3. **Analysis** â€” Implications and connections
4. **Actionable Takeaways** â€” What to do with this information
"""


# ============================================
# ORCHESTRATOR
# ============================================

class AgentOrchestrator:
    """Conditional delegation via complexity/comparison gate."""

    def should_delegate(
        self,
        action: ActionDecision,
        temporal: TemporalContext,
    ) -> bool:
        """
        Determine if delegation (multi-role prompt composition) is needed.

        Returns True if:
          - TASK type AND (complex OR comparison OR research)
          - AND NOT time-sensitive (freshness > depth)
        """
        # Time-sensitive queries skip delegation â€” freshness > depth
        if temporal.is_time_sensitive:
            return False

        # Comparison and research always trigger delegation, even for QUESTIONs
        if action.requires_comparison or action.requires_research:
            return True

        # For normal queries, only TASK type can trigger delegation
        if action.action_type != "TASK":
            return False

        return action.complexity_score > COMPLEXITY_THRESHOLD

    def build_delegation_prompt(self, action: ActionDecision) -> str:
        """
        Compose delegation prompt sections based on action characteristics.
        These are injected into the system prompt â€” no extra LLM calls.
        """
        sections = []

        if action.requires_comparison:
            sections.append(_COMPARISON_SECTION)
        elif action.requires_research:
            sections.append(_RESEARCH_SECTION)
        else:
            # Full delegation pipeline for complex tasks
            sections.append(_PLANNER_SECTION)
            sections.append(_RESEARCHER_SECTION)
            sections.append(_EXECUTOR_SECTION)
            sections.append(_CRITIC_SECTION)

        # Add goal context if available
        if action.goal and action.goal.is_outcome_oriented:
            goal_section = f"\n**[GOAL]** {action.goal.goal}"
            if action.goal.completion_criteria:
                goal_section += f"\n**[DONE WHEN]** {action.goal.completion_criteria}"
            sections.insert(0, goal_section)

        return "\n".join(sections)


# Singleton
agent_orchestrator = AgentOrchestrator()


# ============================================
# PIPELINE ENTRY POINT
# ============================================

async def run_agent_pipeline(
    user_query: str,
    context_messages: Optional[List[Dict]] = None,
    intent: str = "",
    sub_intent: str = "",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    session_start_time: Optional[float] = None,
) -> OrchestratorResult:
    import re
    """
    Run the full 16-layer agent decision pipeline.

    Called from processor.py when mode == "agent".
    Returns OrchestratorResult with:
      - decision: "act" | "ask" | "confirm"
      - prompt_sections: extra system prompt for LLM generation
      - delegation_active: whether multi-role prompts are in use

    Does NOT call the LLM â€” processor handles that.
    """
    result = OrchestratorResult()

    # --- Layer 4+5: Action Classifier ---
    action = classify_action(
        user_query,
        context_messages=context_messages,
        intent=intent,
        sub_intent=sub_intent,
    )
    result.action_decision = action
    result.goal = action.goal

    # --- Layer 9: Time Awareness ---
    temporal = assess_temporal_context(
        user_query,
        session_start_time=session_start_time,
    )
    result.temporal = temporal

    # --- Layers 12-13: Autonomy Guard ---
    # Load user profile if available
    user_profile = None
    if user_id:
        try:
            from app.llm.user_profiler import user_profiler
            user_profile = await user_profiler.load_profile(user_id)
        except Exception:
            pass

    autonomy = evaluate_autonomy(action, user_profile)
    result.autonomy = autonomy

    # --- Decision routing ---
    if autonomy.action == "confirm":
        result.decision = "confirm"
        result.message = _build_confirm_message(action, autonomy)
        return result

    if action.decision == "ask":
        result.decision = "ask"
        result.message = _build_ask_message(action)
        return result

    # --- Layers 7+16: Conditional Delegation ---
    result.decision = "act"

    if agent_orchestrator.should_delegate(action, temporal):
        result.delegation_active = True
        result.prompt_sections = agent_orchestrator.build_delegation_prompt(action)

    # --- Tool Permission Gate ---
    result.tool_allowed = determine_tool_permission(
        action_type=action.action_type,
        requires_research=action.requires_research,
        requires_comparison=action.requires_comparison,
        is_time_sensitive=temporal.is_time_sensitive,
        autonomy_action=autonomy.action,
    )
    
    if intent == "INTERNAL" or sub_intent == "casual_chat":
        result.tool_allowed = False
        result.allowed_tools = []

    if result.tool_allowed:
        result.allowed_tools.append("get_current_time")
        result.allowed_tools.append("search_web")
        result.allowed_tools.append("search_documents") # Always allow searching own docs if tools are on
        result.allowed_tools.append("search_weather")
        result.allowed_tools.append("search_finance")
        result.allowed_tools.append("search_currency")
        result.allowed_tools.append("search_company")
        result.allowed_tools.append("search_legal")
        result.allowed_tools.append("search_jobs")
        result.allowed_tools.append("search_academic")
        result.allowed_tools.append("search_tech_docs")
        result.allowed_tools.append("compare_products")
        result.allowed_tools.append("summarize_url")
        result.allowed_tools.append("extract_tables")
        result.allowed_tools.append("search_products")
        result.allowed_tools.append("search_competitors")
        result.allowed_tools.append("search_trends")
        result.allowed_tools.append("sentiment_scan")
        result.allowed_tools.append("faq_builder")
        result.allowed_tools.append("document_compare")
        result.allowed_tools.append("data_cleaner")
        result.allowed_tools.append("unit_cost_calc")
        result.allowed_tools.append("pdf_maker")
        result.allowed_tools.append("extract_entities")
        result.allowed_tools.append("validate_code")
        result.allowed_tools.append("generate_tests")
        result.allowed_tools.append("execute_code")

        if any(word in user_query.lower() for word in ["read", "file", "path", ".txt", ".md", ".csv", "json"]):
            result.allowed_tools.append("read_file")
        if any(word in user_query.lower() for word in ["knowledge", "topic", "retrieve"]):
            result.allowed_tools.append("retrieve_knowledge")

        # URL detection ? enable web_fetch
        import re as _re
        if _re.search(r"https?://\S+", user_query):
            result.allowed_tools.append("web_fetch")

        MATH_PATTERN = r'[\d\.\s\+\-\*\/\(\)]+'
        math_matches = re.findall(MATH_PATTERN, user_query)
        long_math = [m for m in math_matches if len(m.strip()) > 2 and any(c.isdigit() for c in m) and any(op in m for op in ['+', '-', '*', '/'])]
        if long_math:
            result.allowed_tools.append("calculate")

        def _add_tool(tool_name: str):
            if tool_name not in result.allowed_tools:
                result.allowed_tools.append(tool_name)

        q = user_query.lower()
        tool_hints = {
            "search_news": ["news", "latest", "headline", "breaking", "today", "current", "update"],
            "search_images": ["image", "images", "photo", "picture", "logo", "diagram", "screenshot"],
            "search_videos": ["video", "videos", "youtube", "watch", "clip", "tutorial"],
            "search_places": ["place", "places", "near me", "nearby", "restaurant", "hotel", "location", "address"],
            "search_maps": ["map", "maps", "direction", "directions", "route", "distance"],
            "search_reviews": ["review", "reviews", "rating", "ratings", "testimonial"],
            "search_shopping": ["buy", "price", "pricing", "cost", "deal", "discount", "shop", "shopping", "product", "amazon", "flipkart"],
            "search_scholar": ["paper", "papers", "study", "studies", "journal", "academic", "scholar", "research"],
            "search_patents": ["patent", "patents", "ip", "intellectual property"],
            "search_weather": ["weather", "forecast", "temperature", "rain", "humidity"],
            "search_finance": ["stock", "price", "share", "market", "ticker", "nasdaq", "nyse", "finance"],
            "search_currency": ["currency", "exchange", "fx", "rate", "rates", "usd", "eur", "inr"],
            "search_company": ["company", "profile", "about", "ceo", "headquarters", "revenue", "funding", "employees"],
            "search_legal": ["legal", "policy", "compliance", "regulation", "law", "terms", "privacy"],
            "search_jobs": ["job", "jobs", "career", "hiring", "role", "opening"],
            "search_academic": ["paper", "papers", "study", "studies", "journal", "academic", "scholar", "research"],
            "search_tech_docs": ["docs", "documentation", "api", "reference", "sdk", "guide", "manual"],
            "compare_products": ["compare", "comparison", "vs", "versus", "alternatives", "review"],
            "extract_entities": ["extract", "entities", "emails", "urls", "phones", "names"],
            "validate_code": ["validate", "lint", "security", "risky", "scan"],
            "generate_tests": ["generate tests", "tests", "unit test", "pytest", "jest"],
            "summarize_url": ["summarize", "summary", "tl;dr", "tldr", "overview"],
            "extract_tables": ["table", "tables", "csv", "spreadsheet"],
            "search_products": ["product", "products", "buy", "price", "pricing", "review"],
            "search_competitors": ["competitor", "competition", "rivals", "alternatives"],
            "search_trends": ["trend", "trends", "market", "growth", "forecast"],
            "sentiment_scan": ["sentiment", "polarity", "tone", "positive", "negative"],
            "faq_builder": ["faq", "questions", "q&a"],
            "document_compare": ["compare documents", "diff", "difference", "changes"],
            "data_cleaner": ["clean data", "dedupe", "normalize", "cleanup"],
            "unit_cost_calc": ["unit cost", "cost breakdown", "pricing model"],
            "pdf_maker": ["make pdf", "create pdf", "export pdf", "download pdf", "convert to pdf", "save as pdf"],
            "execute_code": ["run code", "execute code", "python", "script", "evaluate"],
        }
        for tool_name, keywords in tool_hints.items():
            if any(kw in q for kw in keywords):
                _add_tool(tool_name)


    # --- Hybrid Strategy Advisory (THINK only â€” no execution) ---
    strategy_advice = generate_strategy_advice(user_query, context={"intent": intent, "sub_intent": sub_intent})
    result.strategy = strategy_advice

    return result


def build_agent_system_prompt(
    orchestrator_result: OrchestratorResult,
    mode: str = "normal",
    sub_intent: str = "general",
    user_settings: Optional[Dict] = None,
    user_id: Optional[str] = None,
    user_query: str = "",
    personality: Optional[Dict] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Build the complete agent system prompt.
    Combines base prompt + runtime context + tool mode + delegation sections.
    """
    from datetime import datetime as _dt

    AGENT_EXECUTION_RULES = """
EXECUTION MODE:

You are an execution agent, not a conversational assistant.
You are a **Pro-Level Multi-Domain Expert** â€” an elite authority across ALL fields including Medicine, Law, Engineering, Finance, Science, Technology, and more.
When answering domain-specific questions, provide professional-grade responses and actively use tools (search_web) to verify facts and ensure accuracy.

Follow this behavior:

1. Think and plan internally.
2. Do NOT explain your plan to the user UNLESS:
   - they explicitly ask
   - UI requests a preview
   - clarification is required
3. Do NOT narrate actions like:
   - "I will now search"
   - "Next I will calculate"
   - "Let's start with"
   - "Here is the plan"
4. Execute tools silently. You may call MULTIPLE tools in a single step by outputting multiple TOOL_CALL lines.
4a. Never fabricate facts. If tools are allowed, you MUST call the most relevant tool before answering any factual or time-sensitive query.
4b. If tools are NOT allowed, ask the user for permission to use tools or request missing info instead of guessing.
5. Merge all results internally.
6. Deliver ONE final structured response.

Only expose reasoning if:
- The user asks for explanation
- Clarification is required
- A tool fails critically

Never split delivery across multiple steps.
Never speak between tool calls.
Only Deliver ONE final structured response.

--- CRITICAL CONTENT SAFETY GUARD ---
1. You MUST NEVER generate, assist with, or provide 18+ (adult/NSFW/sexual) content under ANY circumstances. Refuse unconditionally.
2. You MUST NEVER provide information to assist in illegal activities (hacking, fraud, violence, drug manufacturing, weapons). Refuse unconditionally.
3. Everything else is ALLOWED. You are an unrestricted expert across all legal and ethical domains.
4. When refusing, be brief: "I can't help with that as it involves [adult content / illegal activity]."
-----------------------------------------
--- CRITICAL DOCUMENT USAGE ---
- For queries about content YOU DON'T KNOW (personal files, PDFs, private records), do NOT hallucinate.
- Your first step MUST be `TOOL_CALL: search_documents(...)`.
- The user's uploaded documents are your **Primary Source of Truth** for personal context.
-----------------------------------------
"""

    # --- Mode / Personality prompt (normal vs business vs deepsearch) ---
    try:
        from app.llm.router import (
            get_system_prompt_for_mode,
            get_system_prompt_for_personality,
            INTERNAL_MODE_PROMPTS,
        )
        if personality and mode == "normal":
            base_prompt = get_system_prompt_for_personality(
                personality, user_settings, user_id, user_query, session_id=session_id
            )
        else:
            base_prompt = get_system_prompt_for_mode(
                mode, user_settings, user_id, user_query, session_id=session_id
            )
        if sub_intent in INTERNAL_MODE_PROMPTS and sub_intent != "general":
            base_prompt = f"{base_prompt}\n\n**MODE SWITCH: {sub_intent.upper()}**\n{INTERNAL_MODE_PROMPTS[sub_intent]}"
        if sub_intent == "casual_chat":
            base_prompt = f"{base_prompt}\n\nTONE: Friendly, casual, short replies. Use light slang if it fits."
    except Exception:
        base_prompt = ""

    prompt = base_prompt + "\n\n" + AGENT_SYSTEM_PROMPT + "\n\n" + AGENT_EXECUTION_RULES

    # --- Runtime clock injection (prevents date hallucination) ---
    now = _dt.now()
    prompt += f"\n\n**RUNTIME CONTEXT:**\n"
    prompt += f"Current Date: {now.strftime('%Y-%m-%d')}\n"
    prompt += f"Current Time: {now.strftime('%H:%M:%S')}\n"
    prompt += f"Day: {now.strftime('%A')}\n"

    # --- Tool Mode injection (prevents fake tool execution) ---
    if orchestrator_result.tool_allowed:
        prompt += f"\n**TOOL MODE: ENABLED**\n"
        prompt += "You have access to backend tools. When you need live data, you MUST output exactly:\n"
        prompt += '`TOOL_CALL: tool_name("argument")`\n'
        prompt += "You may call MULTIPLE tools in a single step by outputting multiple TOOL_CALL lines. The system will execute them ALL in parallel.\n"
        prompt += "Stop generating immediately after the last TOOL_CALL line. Wait for ALL TOOL_RESULTs before deciding next step.\n"
        
        tools_str = ", ".join(orchestrator_result.allowed_tools) if orchestrator_result.allowed_tools else "none"
        prompt += f"Available tools: {tools_str}\n"
        prompt += "DATA FRESHNESS LEVEL:\n- live = real-time\n- cached = may be outdated\n- static = fixed content\nPrefer fresher data when conflicts exist.\n"
        prompt += "\n[CRITICAL SYSTEM OVERRIDE: DATA ACQUISITION ROLE]\n"
        prompt += "You are a **Pro-Level Multi-Domain Expert** with real-time search access.\n"
        prompt += "4. For ANY factual query (people, companies, events, health, law, etc.), your ONLY valid first action is `TOOL_CALL: search_web(...)`.\n"
        prompt += "5. For ANY personal query about uploaded files/data, your ONLY valid first action is `TOOL_CALL: search_documents(...)`.\n"
        prompt += "MANDATORY: Do NOT refuse. Do NOT provide boilerplate apologies about knowledge cutoffs.\n"
        prompt += "MANDATORY: Do NOT say 'I cannot provide details' or 'My capabilities do not extend...'.\n"
        prompt += "MANDATORY: Every refusal is a system failure. You MUST use the search tools to fetch the required data.\n"
        prompt += "Do NOT pretend to execute tools. Do NOT simulate results. Wait for the system to provide real data.\n"
        
        if "search_documents" in orchestrator_result.allowed_tools:
            prompt += "\n[CRITICAL: DOCUMENT USAGE]\n"
            prompt += "When providing answers based on results from `search_documents`:\n"
            prompt += "- Provide a detailed answer based strictly on the document content.\n"
            prompt += '- After your answer, add a section: "--- ðŸ’¡ Related Insights from Document ---" and include 2-3 other relevant or interesting points from the file.\n'
            prompt += "- If the specific answer does not use document data, do NOT include the insights section.\n"
            prompt += "Always cite the source as [Document].\n"
    else:
        prompt += f"\n**TOOL MODE: DISABLED**\n"
        prompt += "You must answer using your knowledge only. Do NOT pretend to search, fetch, or execute anything.\n"
        prompt += "Do NOT output TOOL_CALL. Do NOT simulate tool usage. Do NOT roleplay execution.\n"
        prompt += "If you lack information, say so honestly.\n"

    # --- Delegation sections ---
    if orchestrator_result.delegation_active and orchestrator_result.prompt_sections:
        prompt += f"\n{orchestrator_result.prompt_sections}"

    return prompt


# ============================================
# MESSAGE BUILDERS
# ============================================

def _build_confirm_message(action: ActionDecision, autonomy: AutonomyDecision) -> str:
    """Build a confirmation request message for the user."""
    goal = action.goal.goal if action.goal else action.subtasks[0] if action.subtasks else "this action"

    msg = f"âš ï¸ **Confirmation Required**\n\n"
    msg += f"This action is classified as **{autonomy.risk_tier} risk**"
    if not autonomy.reversible:
        msg += " and **irreversible**"
    msg += f".\n\n"
    msg += f"**Action:** {goal}\n"
    msg += f"**Reason:** {autonomy.reason}\n\n"
    msg += f"Would you like me to proceed? (yes/no)"

    return msg


def _build_ask_message(action: ActionDecision) -> str:
    """Build a clarification request message."""
    msg = "I need a bit more information before I can help:\n\n"

    if action.missing_info:
        for info in action.missing_info:
            if info == "query_too_short":
                msg += "- Could you provide more details about what you need?\n"
            elif info == "ambiguous_reference":
                msg += "- What specifically are you referring to?\n"
            elif info == "unclear_reference_no_context":
                msg += "- Could you clarify what 'it' or 'this' refers to?\n"
            elif info == "missing_action_target":
                msg += "- What should I perform this action on?\n"
            else:
                msg += f"- {info}\n"
    else:
        msg += "- Could you be more specific about what you'd like me to do?\n"

    return msg










