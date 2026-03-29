"""
Graph Builder
Compiles dynamic LLM strategy strings (or linear inputs) into a deterministic PlanGraph. 
"""
import asyncio
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.state.plan_graph import PlanGraph, PlanNode
from app.agent.role_cognition import resolve_plan_node_role


def _apply_node_role(node: PlanNode) -> PlanNode:
    resolution = resolve_plan_node_role(
        node_id=node.node_id,
        action_type=node.action_type,
        declared_role=getattr(node, "role", None),
    )
    node.role = resolution.role
    node.role_fallback_applied = bool(resolution.role_fallback_applied)
    node.role_resolution_source = str(resolution.role_resolution_source or "action_mapping")
    if resolution.warning:
        print(f"[Graph Builder] Role fallback applied for {node.node_id}: {resolution.warning}")
    return node


def _infer_tool_name(text: str) -> Optional[str]:
    q = str(text or "").strip().lower()
    if not q:
        return None
    if any(token in q for token in ["hi", "hello", "hey", "thanks", "thank you"]):
        return None
    if any(token in q for token in ["weather", "forecast", "temperature", "rain"]):
        return "search_weather"
    if any(token in q for token in ["currency", "exchange rate", "usd", "inr", "eur", "gbp"]):
        return "search_currency"
    if any(token in q for token in ["stock", "share price", "market cap", "nasdaq", "nyse", "finance"]):
        return "search_finance"
    if any(token in q for token in ["document", "pdf", "file", "upload", "attachment"]):
        return "search_documents"
    if any(token in q for token in ["patent", "uspto"]):
        return "search_patents"
    if any(token in q for token in ["scholar", "paper", "journal", "study", "research paper"]):
        return "search_scholar"
    if any(token in q for token in ["job", "jobs", "hiring", "career"]):
        return "search_jobs"
    if any(token in q for token in ["law", "legal", "regulation", "compliance"]):
        return "search_legal"
    if any(token in q for token in ["company", "competitor", "startup", "ceo", "founder"]):
        return "search_company"
    if any(token in q for token in ["news", "latest", "today", "current", "recent", "new"]):
        return "search_news"
    if any(token in q for token in ["code", "python", "javascript", "bug", "debug", "stack trace"]):
        return "validate_code"
    if any(token in q for token in ["calculate", "calc", "sum", "multiply", "equation", "+", "-", "*", "/"]):
        return "calculate"
    if any(token in q for token in ["compare product", "best product", "pricing comparison"]):
        return "compare_products"
    if any(token in q for token in ["search", "find", "look up", "research", "reserch", "analyze", "verify", "source", "citation"]):
        return "search_web"
    if len(q.split()) <= 3:
        return None
    return "search_web"


def _make_plan_node(node_id: str, instruction: str, dependencies: Optional[List[str]] = None) -> PlanNode:
    tool_name = _infer_tool_name(instruction)
    payload: Dict[str, Any] = {"instruction": instruction}
    action_type = "REASONING"
    if tool_name:
        action_type = "TOOL_CALL"
        payload["tool"] = tool_name
    return _apply_node_role(PlanNode(
        node_id=node_id,
        action_type=action_type,
        payload=payload,
        dependencies=dependencies or [],
    ))


PLANNER_TOOL_ALLOWLIST = {
    "search_web",
    "search_news",
    "search_scholar",
    "search_legal",
    "search_company",
    "summarize_url",
    "search_finance",
    "search_weather",
    "search_jobs",
    "search_patents",
}


def _strip_fenced_json(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def _extract_completion_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content  # type: ignore[attr-defined]
        if isinstance(content, str):
            return content
    except Exception:
        pass
    try:
        if isinstance(response, dict):
            return str((((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")).strip()
    except Exception:
        pass
    return ""


def _sanitize_search_instruction(text: str) -> str:
    q = str(text or "").strip()
    if not q:
        return ""
    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(r"[\"'`]+", "", q)
    q = re.sub(r"\b(hey|hi|hello|hii|yo|macha|bro|buddy|pls|please)\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(can you|could you|tell me|search|research|reserch|find)\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b(i\s*(got|fgot|forgot|heard)\s*(an|a)?\s*news\s*(that)?)\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip(" ,.-")
    if not q:
        return ""

    stop_tokens = {
        "an", "a", "the", "that", "this", "about", "please", "plz",
        "haas", "hass", "thaty", "tht", "idk", "umm", "uhh",
        "hwy", "hw", "wey", "macha",
        "tell", "me", "can", "you", "could", "would",
    }
    keep_short = {"us", "uk", "un", "eu", "uae", "g7", "g20"}
    cleaned_tokens: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z0-9]+", q):
        t = token.lower()
        if t in stop_tokens:
            continue
        if len(t) < 3 and t not in keep_short and not t.isdigit():
            continue
        if t in seen:
            continue
        seen.add(t)
        cleaned_tokens.append(t)

    if cleaned_tokens:
        return " ".join(cleaned_tokens)[:180].strip()
    return q[:180].strip()


def _sanitize_step_instruction(tool: str, instruction: str) -> str:
    raw_tool = str(tool or "").strip().lower()
    raw_instruction = str(instruction or "").strip()
    if raw_tool == "summarize_url":
        if not raw_instruction:
            return "__TOP_RESULT_URL__"
        lowered = raw_instruction.lower()
        if lowered in {
            "top_result_url",
            "first_result_url",
            "__top_result_url__",
            "__top_result_url_2__",
            "__top_result_url_3__",
            "__next_result_url__",
        }:
            if lowered == "__top_result_url_2__":
                return "__TOP_RESULT_URL_2__"
            if lowered == "__top_result_url_3__":
                return "__TOP_RESULT_URL_3__"
            if lowered == "__next_result_url__":
                return "__TOP_RESULT_URL_2__"
            return "__TOP_RESULT_URL__"
        match = re.search(r"https?://[^\s\"'<>]+", raw_instruction)
        if match:
            return match.group(0).rstrip(".,;)")
        return "__TOP_RESULT_URL__"
    return _sanitize_search_instruction(raw_instruction)


def _is_conflict_news_query(query: str) -> bool:
    q = str(query or "").lower()
    conflict_tokens = {
        "war", "fight", "conflict", "attack", "strikes", "missile", "drone",
        "ceasefire", "military", "battle", "tension", "sanction",
    }
    geo_tokens = {
        "iran", "israel", "pakistan", "india", "russia", "ukraine", "gaza",
        "hamas", "hezbollah", "usa", "us", "middle east",
    }
    return any(t in q for t in conflict_tokens) and any(g in q for g in geo_tokens)


def _country_phrase_for_conflict(query: str) -> str:
    q = str(query or "").lower()
    countries = []
    for token in ("iran", "israel", "pakistan", "india", "russia", "ukraine", "usa"):
        if token in q:
            countries.append(token)
    if not countries:
        return _sanitize_search_instruction(query) or "recent conflict"
    if len(countries) == 1 and countries[0] == "iran":
        # Most common user intent in this product history.
        return "Iran Israel conflict"
    if len(countries) >= 2:
        return " ".join(countries[:2]) + " conflict"
    return countries[0] + " conflict"


def _build_graph_from_steps(
    *,
    session_id: str,
    task_id: str,
    query: str,
    steps: List[Dict[str, str]],
    final_instruction: str = "",
    reasoning_notes: Optional[List[str]] = None,
) -> PlanGraph:
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    node_ids: List[str] = []
    search_node_ids: List[str] = []
    prev_id: Optional[str] = None
    for idx, step in enumerate(steps, start=1):
        node_id = f"P{idx}"
        tool = str(step.get("tool") or "search_web").strip().lower()
        instruction = _sanitize_step_instruction(tool, str(step.get("instruction") or ""))
        if not instruction:
            continue
        if tool not in PLANNER_TOOL_ALLOWLIST:
            inferred = _infer_tool_name(instruction) or "search_web"
            tool = inferred if inferred in PLANNER_TOOL_ALLOWLIST else "search_web"
        dependencies: List[str] = [prev_id] if prev_id else []
        if tool == "summarize_url" and search_node_ids:
            for dep_id in search_node_ids:
                if dep_id not in dependencies:
                    dependencies.append(dep_id)
        graph.add_node(
            _apply_node_role(PlanNode(
                node_id=node_id,
                action_type="TOOL_CALL",
                payload={"tool": tool, "instruction": instruction},
                dependencies=dependencies,
            ))
        )
        node_ids.append(node_id)
        if tool.startswith("search_"):
            search_node_ids.append(node_id)
        prev_id = node_id

    if not node_ids:
        graph.add_node(
            _apply_node_role(PlanNode(
                node_id="P1",
                action_type="TOOL_CALL",
                payload={"tool": "search_web", "instruction": _sanitize_search_instruction(query)},
            ))
        )
        node_ids = ["P1"]

    graph.add_node(
        _apply_node_role(PlanNode(
            node_id="FINAL",
            action_type="REASONING",
            payload={"instruction": final_instruction or f"Synthesize findings for: {query}"},
            dependencies=node_ids,
        ))
    )
    notes = [str(n).strip() for n in (reasoning_notes or []) if str(n).strip()]
    if notes:
        graph.metadata["reasoning_notes"] = notes[:6]
    return graph


def _build_research_fallback_steps(query: str, use_news: bool, wants_multi_source: bool) -> List[Dict[str, str]]:
    cleaned = _sanitize_search_instruction(query) or query
    base = cleaned
    q_lower = str(query or "").lower()
    timeliness_markers = ("latest", "recent", "current", "today", "new", "new data", "update", "updated", "as of")
    now_year = datetime.now(timezone.utc).year
    recency_suffix = f" {now_year} {max(2000, now_year - 1)}" if any(t in q_lower for t in timeliness_markers) else ""
    if _is_conflict_news_query(query):
        phrase = _country_phrase_for_conflict(query)
        step_one = {"tool": "search_news", "instruction": f"{phrase} timeline June 2025 to 2026 latest updates"}
        step_two = {"tool": "search_web", "instruction": f"{phrase} key events casualties missiles ceasefire official statements"}
        step_three = {"tool": "summarize_url", "instruction": "__TOP_RESULT_URL__"}
        return [step_one, step_two, step_three] if wants_multi_source else [step_one, step_two]

    if use_news:
        step_one = {"tool": "search_news", "instruction": f"{base}{recency_suffix} latest verified updates"}
        step_two = {"tool": "search_web", "instruction": f"{base}{recency_suffix} timeline key events official statements Reuters BBC"}
    else:
        step_one = {"tool": "search_web", "instruction": f"{base}{recency_suffix}".strip()}
        step_two = {"tool": "search_web", "instruction": f"{base}{recency_suffix} timeline key events official statements reliable sources"}
    if wants_multi_source:
        return [step_one, step_two]
    return [step_one]


def _build_fallback_reasoning_notes(query: str, steps: List[Dict[str, str]]) -> List[str]:
    cleaned = _sanitize_search_instruction(query) or str(query or "").strip()
    cleaned = cleaned[:180]
    notes: List[str] = []
    if cleaned:
        notes.append(f'I interpreted your request as: "{cleaned}".')
    if steps:
        first = str(steps[0].get("instruction") or "").strip()
        if first:
            notes.append(f"I'll begin with a targeted search: {first}.")
        if len(steps) > 1:
            second = str(steps[1].get("instruction") or "").strip()
            if second:
                notes.append(f"Then I'll cross-check with: {second}.")
    notes.append("After collecting sources, I'll synthesize the most supported answer.")
    return notes[:4]


def _adaptive_read_depth(query: str, context: str = "") -> int:
    q = str(query or "").lower()
    c = str(context or "").lower()
    depth = 2
    deep_markers = {
        "deep", "detailed", "comprehensive", "full breakdown", "compare", "analysis",
        "timeline", "cross-check", "verify", "fact check", "sources",
    }
    conflict_markers = {"war", "fight", "conflict", "attack", "sanction", "ceasefire"}
    if any(marker in q for marker in deep_markers):
        depth = max(depth, 3)
    if any(marker in q for marker in conflict_markers):
        depth = max(depth, 3)
    if any(flag in c for flag in ("citation_coverage_low", "source_mix_incomplete", "freshness_low")):
        depth = max(depth, 3)
    if "cross_source_conflict" in c:
        depth = max(depth, 4)
    m = re.search(r"read-depth target:\s*([1-9])", c)
    if m:
        try:
            depth = max(depth, int(m.group(1)))
        except Exception:
            pass
    return max(2, min(4, depth))


def _ensure_read_steps(
    steps: List[Dict[str, str]],
    *,
    min_reads: int = 2,
    prefer_max_steps: int = 4,
) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = str(step.get("tool") or "").strip().lower()
        instruction = str(step.get("instruction") or "").strip()
        if not tool or not instruction:
            continue
        normalized.append({"tool": tool, "instruction": instruction})

    if not normalized:
        return normalized

    search_steps = [s for s in normalized if str(s.get("tool") or "").strip().lower().startswith("search_")]
    read_steps = [s for s in normalized if str(s.get("tool") or "").strip().lower() == "summarize_url"]
    if not search_steps:
        return normalized

    while len(read_steps) < max(1, int(min_reads)):
        idx = len(read_steps) + 1
        placeholder = "__TOP_RESULT_URL__" if idx == 1 else f"__TOP_RESULT_URL_{idx}__"
        read_steps.append({"tool": "summarize_url", "instruction": placeholder})

    max_search_slots = max(1, int(prefer_max_steps) - int(min_reads))
    return search_steps[:max_search_slots] + read_steps[:max(1, int(min_reads))]


async def _plan_steps_with_llm(
    *,
    query: str,
    context: str,
    client: Optional[Any],
    model_to_use: Optional[str],
    read_depth_target: int = 2,
) -> Optional[Dict[str, Any]]:
    if client is None:
        return None

    planner_model = str(model_to_use or "").strip() or "deepseek/deepseek-chat"
    planner_prompt = (
        "You are a query planning engine for a research assistant.\n"
        "Return ONLY valid JSON with this schema:\n"
        '{"reasoning_notes":["..."],'
        '"steps":[{"tool":"search_news|search_web|search_scholar|search_company|search_legal|summarize_url","instruction":"..."}],'
        '"final_instruction":"..."}\n'
        "Rules:\n"
        "1) Rewrite noisy user phrasing into concise factual search queries.\n"
        "2) Choose 2-5 steps. Prefer search_news first for current events.\n"
        "3) For research/news requests, include at least TWO summarize_url steps at the end:\n"
        "   first '__TOP_RESULT_URL__', second '__TOP_RESULT_URL_2__'.\n"
        "   If read_depth_target >= 3, include additional summarize_url placeholders.\n"
        "4) Do not copy filler words from the user.\n"
        "5) Keep each instruction under 120 characters.\n"
        "6) For conflict/geopolitical topics, include timeline/events/official-statements in step intents.\n"
        "7) Provide 2-4 short reasoning_notes in natural first-person style.\n"
        "8) Output JSON only."
    )
    user_payload = {
        "query": query,
        "context": context or "",
        "read_depth_target": max(2, min(4, int(read_depth_target or 2))),
    }

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=planner_model,
                messages=[
                    {"role": "system", "content": planner_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                temperature=0.1,
                stream=False,
            ),
            timeout=7.0,
        )
    except Exception:
        return None

    raw = _strip_fenced_json(_extract_completion_text(response))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None

    steps_raw = parsed.get("steps")
    if not isinstance(steps_raw, list):
        return None

    steps: List[Dict[str, str]] = []
    for item in steps_raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip().lower()
        instruction = _sanitize_step_instruction(tool, str(item.get("instruction") or ""))
        if not instruction:
            continue
        if tool not in PLANNER_TOOL_ALLOWLIST:
            inferred = _infer_tool_name(instruction) or "search_web"
            tool = inferred if inferred in PLANNER_TOOL_ALLOWLIST else "search_web"
        steps.append({"tool": tool, "instruction": instruction[:120]})
        if len(steps) >= 5:
            break

    if not steps:
        return None

    depth = max(2, min(4, int(read_depth_target or 2)))
    prefer_max = min(6, max(4, depth + 2))
    steps = _ensure_read_steps(steps, min_reads=depth, prefer_max_steps=prefer_max)

    notes: List[str] = []
    raw_notes = parsed.get("reasoning_notes")
    if isinstance(raw_notes, list):
        for item in raw_notes:
            note = str(item or "").strip()
            if note:
                notes.append(note[:240])
            if len(notes) >= 6:
                break
    if not notes:
        analysis = str(parsed.get("analysis") or "").strip()
        if analysis:
            notes = [seg.strip()[:240] for seg in re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z])", analysis) if seg.strip()][:4]

    final_instruction = str(parsed.get("final_instruction") or "").strip()
    return {"steps": steps, "final_instruction": final_instruction, "reasoning_notes": notes}


def classify_query(query: str) -> str:
    """Lightweight query classifier used by the recovered TaskManager path."""
    q = str(query or "").strip().lower()
    if not q:
        return "general"
    if any(token in q for token in ["weather", "forecast", "temperature", "rain"]):
        return "weather"
    if any(token in q for token in ["calculate", "solve", "+", "-", "*", "/", "equation"]):
        return "math"
    if any(token in q for token in ["who is", "what is", "when was", "where is", "founder", "ceo"]):
        return "simple_fact"
    if any(token in q for token in ["compare", "analyze", "research", "reserch", "verify", "sources", "citation"]):
        return "research"
    return "general"

def build_linear_plan_graph(session_id: str, task_id: str, steps: List[str]) -> PlanGraph:
    """
    Fallback builder for simple linear sequences. 
    A -> B -> C.
    """
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    
    prev_id = None
    for idx, step in enumerate(steps):
        node_id = f"N{idx+1}"
        deps = [prev_id] if prev_id else []

        node = _make_plan_node(node_id=node_id, instruction=step, dependencies=deps)
        graph.add_node(node)
        prev_id = node_id
        
    return graph

async def compile_plan_graph(
    session_id: str,
    task_id: str,
    query: str,
    context: Optional[str] = None,
    client: Optional[Any] = None,
    model_to_use: Optional[str] = None,
) -> PlanGraph:
    """
    Intelligently compiles a query into a dependency graph.
    In a full production setting, this would use a fast, focused LLM call (like Haiku) 
    to emit a hard-structured JSON DAG. 
    
    For MVP Phase 5, we parse known heuristic patterns into graphs.
    """
    # 1. Very basic routing heuristically 
    lower_query = query.lower()

    is_researchy = any(token in lower_query for token in ["research", "reserch", "analyze", "verify", "sources", "citation"])
    is_timely = any(token in lower_query for token in ["latest", "today", "current", "recent", "new", "news", "headline", "breaking"])

    if is_researchy or is_timely:
        use_news = any(token in lower_query for token in ["news", "latest", "today", "headline", "breaking"])
        wants_multi_source = (
            is_researchy
            and any(token in lower_query for token in ["sources", "verify", "cross", "compare", "research", "reserch", "analysis"])
        )
        read_depth = _adaptive_read_depth(query, context or "")
        prefer_max_steps = min(6, max(4, int(read_depth) + 2))
        llm_plan = await _plan_steps_with_llm(
            query=query,
            context=context or "",
            client=client,
            model_to_use=model_to_use,
            read_depth_target=read_depth,
        )
        if llm_plan:
            llm_steps = _ensure_read_steps(
                llm_plan.get("steps") or [],
                min_reads=read_depth,
                prefer_max_steps=prefer_max_steps,
            )
            return _build_graph_from_steps(
                session_id=session_id,
                task_id=task_id,
                query=query,
                steps=llm_steps,
                final_instruction=str(llm_plan.get("final_instruction") or ""),
                reasoning_notes=llm_plan.get("reasoning_notes") or [],
            )
        fallback_steps = _ensure_read_steps(
            _build_research_fallback_steps(query=query, use_news=use_news, wants_multi_source=wants_multi_source),
            min_reads=read_depth,
            prefer_max_steps=prefer_max_steps,
        )
        return _build_graph_from_steps(
            session_id=session_id,
            task_id=task_id,
            query=query,
            steps=fallback_steps,
            reasoning_notes=_build_fallback_reasoning_notes(query, fallback_steps),
        )
    
    if "and" in lower_query and "then" in lower_query:
        # Example pattern: "search X and search Y then summarize"
        # N1: Search X, N2: Search Y, N3: Summarize (depends on N1, N2)
        parts = lower_query.split("then")
        parallels = parts[0].split("and")
        
        graph = PlanGraph(graph_id=task_id, session_id=session_id)
        
        parallel_ids = []
        for idx, p in enumerate(parallels):
            n_id = f"P{idx+1}"
            node = _make_plan_node(node_id=n_id, instruction=p.strip())
            graph.add_node(node)
            parallel_ids.append(n_id)
            
        n_final = _apply_node_role(PlanNode(
            node_id="FINAL", 
            action_type="REASONING", 
            payload={"instruction": parts[1].strip()},
            dependencies=parallel_ids
        ))
        graph.add_node(n_final)
        return graph

    # Fallback to single monolithic node for simple queries
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    node = _make_plan_node(node_id="N1", instruction=_sanitize_search_instruction(query) or query)
    graph.add_node(node)
    return graph
