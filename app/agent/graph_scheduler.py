"""
Deterministic Graph Scheduler
Executes a PlanGraph in deterministic dependency order.
Allows safe parallel execution for independent researcher tool nodes.
"""
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple, Set, Callable
import asyncio
import json
import time
import re
from urllib.parse import urlparse

from app.state.plan_graph import PlanGraph, PlanNode, NodeStatus
from app.state.transaction_manager import begin_transaction, commit_transaction, rollback_transaction, is_memory_suppressed
from app.agent.tool_executor import (
    parse_tool_calls,
    execute_tool,
    format_tool_result,
    ToolCall,
    ToolResult,
    ExecutionContext,
    TOOLS,
)
from app.agent.repair_engine import repair_cycle, generate_repair_strategy, build_repair_prompt
from app.chat.mode_mapper import normalize_chat_mode
from app.agent.role_cognition import resolve_plan_node_role
from app.agent.tool_confidence_store import get_tool_confidence_store


def _safe_preview(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _emit_info(event: str, **payload: Any) -> str:
    body = {"event": event}
    body.update(payload)
    return f"[INFO]{json.dumps(body)}"


def _node_stream_meta(node: PlanNode, mode_name: str) -> Dict[str, Any]:
    return {
        "node_id": str(getattr(node, "node_id", "") or ""),
        "role": str(getattr(node, "role", "executor") or "executor"),
        "mode": str(mode_name or "smart"),
        "role_fallback_applied": bool(getattr(node, "role_fallback_applied", False)),
        "role_resolution_source": str(getattr(node, "role_resolution_source", "unknown") or "unknown"),
    }

def _is_soft_tool_failure(tool_result: Any) -> bool:
    reason = str(getattr(tool_result, "error", "") or "").lower()
    source = str(getattr(tool_result, "source", "") or "").lower()
    if "search_cooldown" in reason or "search_cooldown" in source:
        return True
    if "rate limited" in reason:
        return True
    if "no_readable_url_candidates" in reason:
        return True
    if _is_non_actionable_scrape_error(reason):
        return True
    return False


def _is_non_actionable_scrape_error(reason: str) -> bool:
    lower = str(reason or "").strip().lower()
    if not lower:
        return False
    markers = (
        "robots.txt",
        "does not allow automated scraping",
        "website does not allow automated scraping",
        "http 403",
        "forbidden",
        "access denied",
        "no_readable_url_candidates",
    )
    return any(m in lower for m in markers)


def _should_silence_tool_error(tool_name: str, status: str, error: str) -> bool:
    if str(status or "").strip().lower() not in {"failed", "error", "blocked", "throttled"}:
        return False
    if not _is_retrieval_tool(tool_name):
        return False
    return _is_non_actionable_scrape_error(error)

def _soft_tool_result_payload(tool_name: str, tool_result: Any) -> Dict[str, Any]:
    reason = str(getattr(tool_result, "error", "") or "temporary tool throttle")
    return {
        "command": tool_name,
        "result": {"warning": "tool_throttled", "reason": reason},
        "source": getattr(tool_result, "source", tool_name),
        "confidence": "low",
        "trust": "unverified",
    }


def _normalize_result_items(raw_items: Any, *, max_items: int = 10) -> List[Dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    out: List[Dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = _safe_preview(item.get("title") or item.get("name") or item.get("snippet") or "", 120)
        url = str(item.get("link") or item.get("url") or item.get("source") or "").strip()
        if not title and not url:
            continue
        if not title and url:
            title = _safe_preview(url, 120)
        normalized: Dict[str, str] = {"title": title}
        if url:
            normalized["url"] = url
        out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _extract_tool_result_metadata(tool_name: str, tool_result: Any) -> Dict[str, Any]:
    data = getattr(tool_result, "data", None)
    metadata: Dict[str, Any] = {}
    tool_lower = str(tool_name or "").strip().lower()

    raw_items: Any = None
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        for key in ("results", "items", "organic", "news"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                raw_items = candidate
                break

    result_items = _normalize_result_items(raw_items)
    if result_items:
        metadata["result_items"] = result_items
        if isinstance(raw_items, list):
            metadata["result_count"] = len(raw_items)
        else:
            metadata["result_count"] = len(result_items)

    if isinstance(data, dict):
        read_title = _safe_preview(data.get("title") or data.get("name") or "", 140)
        read_url = str(data.get("url") or data.get("link") or "").strip()
        if read_title:
            metadata["read_title"] = read_title
        if read_url:
            metadata["read_url"] = read_url

        summary_hint = _safe_preview(data.get("summary") or data.get("snippet") or "", 180)
        if summary_hint:
            metadata["result_hint"] = summary_hint

    if "search" in tool_lower and "result_count" in metadata:
        metadata.setdefault("result_hint", f'Found {int(metadata["result_count"])} results')

    return metadata


def _dedupe_result_items(raw_items: Any, *, max_items: int = 20) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or item.get("link") or item.get("source") or "").strip().lower()
        title = str(item.get("title") or item.get("name") or "").strip().lower()
        snippet = str(item.get("snippet") or item.get("summary") or "").strip().lower()
        key = url or f"{title}|{snippet[:140]}"
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(dict(item))
        if len(deduped) >= max_items:
            break
    return deduped


def _dedupe_tool_result_data(data: Any) -> Any:
    if isinstance(data, list):
        return _dedupe_result_items(data)
    if isinstance(data, dict):
        updated = dict(data)
        touched = False
        for key in ("results", "items", "organic", "news", "articles"):
            value = updated.get(key)
            if isinstance(value, list):
                updated[key] = _dedupe_result_items(value)
                touched = True
        if touched:
            return updated
    return data


def _build_parallel_merge_snapshot(
    successful_parallel_results: List[Tuple[PlanNode, str, Any]],
    *,
    max_sources: int = 12,
) -> Dict[str, Any]:
    seen: Set[str] = set()
    merged_sources: List[Dict[str, str]] = []
    for _, tool_name, tool_result in successful_parallel_results:
        metadata = _extract_tool_result_metadata(tool_name, tool_result)
        for item in metadata.get("result_items") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            key = (url.lower() if url else title.lower())
            if not key or key in seen:
                continue
            seen.add(key)
            entry = {"title": title}
            if url:
                entry["url"] = url
            merged_sources.append(entry)
            if len(merged_sources) >= max_sources:
                break
        if len(merged_sources) >= max_sources:
            break
    return {
        "tool_runs": len(successful_parallel_results),
        "unique_sources": len(merged_sources),
        "sources": merged_sources,
    }


def _emit_tool_result_info(
    *,
    node_id: str,
    tool: str,
    status: str,
    args_preview: str,
    tool_result: Any = None,
    error: str = "",
    role: str = "",
    mode: str = "",
    role_fallback_applied: bool = False,
    role_resolution_source: str = "",
    silent: bool = False,
) -> str:
    payload: Dict[str, Any] = {
        "node_id": node_id,
        "tool": tool,
        "status": status,
        "args_preview": args_preview,
    }
    if role:
        payload["role"] = role
    if mode:
        payload["mode"] = mode
    payload["role_fallback_applied"] = bool(role_fallback_applied)
    if role_resolution_source:
        payload["role_resolution_source"] = role_resolution_source
    if error:
        payload["error"] = error
    effective_silent = bool(silent) or _should_silence_tool_error(tool, status, error)
    if effective_silent:
        payload["silent"] = True
    if tool_result is not None:
        payload.update(_extract_tool_result_metadata(tool, tool_result))
    return _emit_info("tool_result", **payload)


def _is_retrieval_tool(tool_name: str) -> bool:
    name = str(tool_name or "").strip().lower()
    if not name:
        return False
    if name.startswith("search_"):
        return True
    return name in {"summarize_url", "web_fetch", "extract_tables"}


def _extract_http_urls(value: Any, *, max_urls: int = 20) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()

    def _walk(node: Any) -> None:
        if len(out) >= max_urls:
            return
        if isinstance(node, str):
            text = node.strip()
            if text.startswith("http://") or text.startswith("https://"):
                if text not in seen:
                    seen.add(text)
                    out.append(text)
            return
        if isinstance(node, list):
            for item in node:
                _walk(item)
                if len(out) >= max_urls:
                    return
            return
        if isinstance(node, dict):
            for key in ("url", "link", "source"):
                candidate = str(node.get(key) or "").strip()
                if candidate.startswith("http://") or candidate.startswith("https://"):
                    if candidate not in seen:
                        seen.add(candidate)
                        out.append(candidate)
                        if len(out) >= max_urls:
                            return
            for key in ("result", "results", "items", "organic", "news", "data"):
                _walk(node.get(key))
                if len(out) >= max_urls:
                    return

    _walk(value)
    return out


def _extract_first_http_url(value: Any) -> str:
    urls = _extract_http_urls(value, max_urls=1)
    return urls[0] if urls else ""


def _domain_from_url(value: str) -> str:
    try:
        host = (urlparse(str(value or "").strip()).netloc or "").lower()
    except Exception:
        host = ""
    return host[4:] if host.startswith("www.") else host


_READ_BLOCKLIST_DOMAINS: Set[str] = {
    "reddit.com",
    "old.reddit.com",
    "np.reddit.com",
    "books.google.com",
    "scholar.google.com",
    "onlinelibrary.wiley.com",
    "advanced.onlinelibrary.wiley.com",
}


def _is_blocked_read_domain(url: str) -> bool:
    host = _domain_from_url(url)
    return bool(host and host in _READ_BLOCKLIST_DOMAINS)


def _collect_previous_read_domains(graph: PlanGraph) -> Set[str]:
    out: Set[str] = set()
    for node in graph.nodes.values():
        if str(getattr(node, "action_type", "")).upper() != "TOOL_CALL":
            continue
        tool = str((node.payload or {}).get("tool") or "").strip().lower()
        if tool != "summarize_url":
            continue
        if str(getattr(node, "status", "")).upper() != "COMPLETED":
            continue
        url = _extract_first_http_url(node.result)
        host = _domain_from_url(url)
        if host:
            out.add(host)
    return out


def _wants_domain_diversity(raw_instruction: str) -> bool:
    instruction = str(raw_instruction or "").strip().lower()
    return instruction in {"__top_result_url_2__", "__top_result_url_3__", "__next_result_url__"}


def _resolve_summarize_url_arg(node: PlanNode, graph: PlanGraph, user_query: str) -> str:
    explicit = str(node.payload.get("instruction") or "").strip()
    if explicit.startswith("http://") or explicit.startswith("https://"):
        return explicit

    candidate_urls: List[str] = []
    for dep_id in reversed(node.dependencies or []):
        dep_node = graph.get_node(dep_id)
        if not dep_node:
            continue
        candidate_urls.extend(_extract_http_urls(dep_node.result, max_urls=20))

    if not candidate_urls:
        query_url = _extract_first_http_url(user_query)
        if query_url:
            candidate_urls = [query_url]
    if not candidate_urls:
        return ""

    unique_urls: List[str] = []
    seen: Set[str] = set()
    for url in candidate_urls:
        key = str(url).strip()
        if not key or key in seen:
            continue
        if _is_blocked_read_domain(key):
            continue
        seen.add(key)
        unique_urls.append(key)

    if _wants_domain_diversity(explicit):
        used_domains = _collect_previous_read_domains(graph)
        for url in unique_urls:
            host = _domain_from_url(url)
            if host and host not in used_domains:
                return url

    return unique_urls[0]


_QUERY_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "from", "this", "that", "about", "latest", "recent", "news",
    "update", "updates", "official", "statement", "statements", "timeline", "events", "event",
    "reliable", "sources", "additional", "verified", "information", "data",
}


def _query_terms(value: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for token in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if len(token) < 3 or token in _QUERY_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 10:
            break
    return out


def _collect_search_rows(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)][:10]
    if isinstance(data, dict):
        for key in ("results", "items", "news", "organic", "data"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, dict)][:10]
    return []


def _is_low_relevance_search_result(tool_result: Any, query: str) -> bool:
    rows = _collect_search_rows(getattr(tool_result, "data", None))
    if not rows:
        return False
    terms = _query_terms(query)
    if not terms:
        return False

    considered = 0
    hits = 0
    for row in rows[:5]:
        text = " ".join(
            str(row.get(key) or "") for key in ("title", "snippet", "link", "url", "source")
        ).lower()
        if not text.strip():
            continue
        considered += 1
        if any(term in text for term in terms):
            hits += 1
    if considered == 0:
        return False
    return (hits / considered) < 0.25


def _rewrite_search_query(raw_query: str, user_query: str) -> str:
    source = str(raw_query or "").strip() or str(user_query or "").strip()
    tokens = _query_terms(source)
    base = " ".join(tokens[:8]) if tokens else re.sub(r"\s+", " ", source).strip()
    if not base:
        return ""
    lower = base.lower()
    if not any(t in lower for t in ("latest", "recent", "today", "current", "news")):
        base = f"{base} latest verified updates"
    if not any(t in lower for t in ("timeline", "events", "official")):
        base = f"{base} timeline official statements"
    return re.sub(r"\s+", " ", base).strip()[:180]


def _alternate_search_tool(tool_name: str) -> str:
    name = str(tool_name or "").strip().lower()
    if name == "search_news":
        return "search_web"
    if name == "search_web":
        return "search_news"
    return name


async def _maybe_retry_low_relevance_search(
    *,
    node: PlanNode,
    tool_name: str,
    tool_call: ToolCall,
    tool_result: Any,
    exec_ctx: Any,
    user_query: str,
    session_scope: str,
) -> Tuple[str, ToolCall, Any, bool]:
    name = str(tool_name or "").strip().lower()
    if name not in {"search_web", "search_news", "search_scholar"}:
        return tool_name, tool_call, tool_result, False
    if not getattr(tool_result, "success", False):
        return tool_name, tool_call, tool_result, False
    if bool((node.payload or {}).get("_rewrite_retry_done")):
        return tool_name, tool_call, tool_result, False

    query_used = str(tool_call.args or node.payload.get("instruction") or user_query or "").strip()
    if not _is_low_relevance_search_result(tool_result, query_used):
        return tool_name, tool_call, tool_result, False

    rewritten = _rewrite_search_query(query_used, user_query)
    if not rewritten or rewritten.lower() == query_used.lower():
        node.payload["_rewrite_retry_done"] = True
        return tool_name, tool_call, tool_result, False

    retry_tool_name = _alternate_search_tool(name)
    retry_call = ToolCall(
        name=retry_tool_name,
        args=rewritten,
        raw=f'TOOL_CALL: {retry_tool_name}("{rewritten}")',
        session_id=session_scope,
    )
    retry_result = await execute_tool(retry_call, exec_ctx)
    if getattr(retry_result, "success", False):
        node.payload["_rewrite_retry_done"] = True
        node.payload["tool"] = retry_tool_name
        node.payload["instruction"] = rewritten
        return retry_tool_name, retry_call, retry_result, True

    node.payload["_rewrite_retry_done"] = True
    return tool_name, tool_call, tool_result, False


def _select_fallback_retrieval_tool(allowed_tools: Set[str]) -> Optional[str]:
    candidates = [c for c in ("search_web", "search_news", "search_scholar", "search_documents") if c in allowed_tools]
    if not candidates:
        return None
    store = get_tool_confidence_store()
    ordering = {name: idx for idx, name in enumerate(("search_web", "search_news", "search_scholar", "search_documents"))}
    ranked: List[Tuple[float, str]] = []
    for name in candidates:
        rec = store.get(name)
        if rec is None:
            # Keep gentle prior for exploration.
            score = 0.55 - (0.02 * float(ordering.get(name, 0)))
        else:
            # Never hard-disable a tool; keep exploration floor.
            score = max(0.30, float(rec.selection_score()))
        ranked.append((score, name))
    ranked.sort(key=lambda row: (-row[0], ordering.get(row[1], 99)))
    return str(ranked[0][1])


def _choose_reliability_preferred_retrieval_tool(
    *,
    current_tool: str,
    allowed_tools: Set[str],
    min_score: float = 0.30,
    min_improvement: float = 0.08,
) -> str:
    current = str(current_tool or "").strip().lower()
    if not _is_retrieval_tool(current):
        return current
    candidates = [name for name in ("search_web", "search_news", "search_scholar", "search_documents") if name in allowed_tools]
    if not candidates:
        return current
    store = get_tool_confidence_store()
    scored: List[Tuple[str, float]] = []
    for name in candidates:
        rec = store.get(name)
        if rec is None:
            score = 0.55
        else:
            score = max(min_score, float(rec.selection_score()))
        scored.append((name, score))
    scored.sort(key=lambda row: row[1], reverse=True)
    best_name, best_score = scored[0]
    current_score = next((s for n, s in scored if n == current), 0.55)
    if best_name != current and current_score <= min_score and (best_score - current_score) >= min_improvement:
        return best_name
    return current


def _append_tool_runtime(
    *,
    graph: PlanGraph,
    tool_name: str,
    success: bool,
    latency_ms: int,
    hard_error: bool = False,
) -> None:
    graph.metadata = dict(graph.metadata or {})
    rows = list(graph.metadata.get("tool_runtime_stats") or [])
    rows.append(
        {
            "tool": str(tool_name or "").strip().lower(),
            "success": bool(success),
            "latency_ms": max(0, int(latency_ms or 0)),
            "hard_error": bool(hard_error),
        }
    )
    graph.metadata["tool_runtime_stats"] = rows[-500:]


def _resolve_allowed_tools(agent_result: Any, state: Any = None) -> Tuple[Set[str], bool]:
    mode_name = normalize_chat_mode(str(getattr(state, "mode", "") or ""))
    unrestricted = mode_name in {"smart", "agent", "research_pro"}
    if unrestricted and state is not None:
        # Hybrid/Agent product mode: do not permission-block planner tools here.
        # ToolExecutor remains the source of truth for execution safety/errors.
        return set(TOOLS.keys()), True

    allowed = set(getattr(agent_result, "allowed_tools", []) or [])
    tool_allowed = bool(getattr(agent_result, "tool_allowed", False))
    return allowed, tool_allowed


def _build_planned_tool_call(node: PlanNode, user_query: str, session_scope: str, graph: PlanGraph) -> Tuple[str, ToolCall]:
    tool_name = str(node.payload.get("tool") or "").strip()
    tool_args = str(node.payload.get("instruction") or user_query or "").strip()
    if tool_name == "summarize_url":
        resolved_url = _resolve_summarize_url_arg(node, graph, user_query)
        if resolved_url:
            tool_args = resolved_url
    return tool_name, ToolCall(
        name=tool_name,
        args=tool_args,
        raw=f'TOOL_CALL: {tool_name}("{tool_args}")',
        session_id=session_scope,
    )


async def _execute_planned_tool_node(node: PlanNode, user_query: str, session_scope: str, exec_ctx: Any, graph: PlanGraph) -> Tuple[PlanNode, str, ToolCall, Any, int]:
    tool_name, tool_call = _build_planned_tool_call(node, user_query, session_scope, graph)
    if str(tool_name or "").strip().lower() == "summarize_url" and not str(tool_call.args or "").strip():
        return (
            node,
            tool_name,
            tool_call,
            ToolResult(
                tool_name=tool_name,
                success=False,
                data={"warning": "no_readable_url_candidates"},
                source=tool_name,
                confidence="low",
                trust="unverified",
                error="no_readable_url_candidates",
            ),
            0,
        )
    started = time.time()
    tool_result = await execute_tool(tool_call, exec_ctx)
    latency_ms = int(max(0, round((time.time() - started) * 1000)))
    return node, tool_name, tool_call, tool_result, latency_ms

async def run_plan_graph(
    graph: PlanGraph,
    strategy: Any,
    user_query: str,
    messages: List[Dict[str, str]],
    agent_result: Any,
    client: Any,
    model_to_use: str,
    create_kwargs: Dict[str, Any],
    state: Any = None,
    mode_name: str = "smart",
    resolve_node_policy: Optional[Callable[[str], Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Executes a PlanGraph sequentially.
    Matches the original generator loop but orchestrates over DAG nodes instead of indices.
    """
    exec_ctx = agent_result.execution_context
    session_id = graph.session_id
    task_scope = f"{graph.session_id}:{graph.graph_id}"
    mode_label = normalize_chat_mode(str(mode_name or "smart"))
    role_flow: List[Dict[str, Any]] = list((graph.metadata or {}).get("role_flow") or [])
    role_trace_map: Dict[str, Dict[str, Any]] = {}

    def _ensure_node_role(node: PlanNode) -> Dict[str, Any]:
        resolved = resolve_plan_node_role(
            node_id=str(getattr(node, "node_id", "")),
            action_type=str(getattr(node, "action_type", "")),
            declared_role=str(getattr(node, "role", "") or ""),
        )
        node.role = resolved.role
        node.role_fallback_applied = bool(resolved.role_fallback_applied)
        node.role_resolution_source = str(resolved.role_resolution_source or "action_mapping")
        if resolved.warning:
            print(f"[Graph Scheduler] Role fallback applied ({node.node_id}): {resolved.warning}")
        return {
            "role": str(node.role),
            "role_fallback_applied": bool(node.role_fallback_applied),
            "role_resolution_source": str(node.role_resolution_source),
        }

    def _begin_role_trace(node: PlanNode) -> Dict[str, Any]:
        role_meta = _ensure_node_role(node)
        started_at_ms = int(time.time() * 1000)
        row = {
            "node_id": str(node.node_id),
            "action_type": str(node.action_type),
            "role": role_meta["role"],
            "mode": mode_label,
            "status": "RUNNING",
            "role_fallback_applied": bool(role_meta["role_fallback_applied"]),
            "role_resolution_source": role_meta["role_resolution_source"],
            "started_at_ms": started_at_ms,
            "duration_ms": 0,
            "tool_calls": 0,
            "retries": 0,
            "parallel_exception_count": 0,
        }
        role_flow.append(row)
        role_trace_map[str(node.node_id)] = row
        return row

    def _finalize_role_trace(
        node: PlanNode,
        *,
        status: str,
        tool_calls: int = 0,
        retries: int = 0,
        parallel_exception_count: int = 0,
    ) -> None:
        row = role_trace_map.get(str(node.node_id))
        if row is None:
            row = _begin_role_trace(node)
        ended_at_ms = int(time.time() * 1000)
        started_at_ms = int(row.get("started_at_ms", ended_at_ms))
        row["status"] = str(status or "UNKNOWN").upper()
        row["ended_at_ms"] = ended_at_ms
        row["duration_ms"] = max(0, ended_at_ms - started_at_ms)
        row["tool_calls"] = int(max(0, tool_calls))
        row["retries"] = int(max(0, retries))
        row["parallel_exception_count"] = int(max(0, int(row.get("parallel_exception_count", 0) or 0) + int(max(0, parallel_exception_count))))

    def _persist_role_flow() -> None:
        graph.metadata = dict(graph.metadata or {})
        graph.metadata["role_flow"] = role_flow[-200:]

    # Optional: We can wrap the entire graph execution in a transaction
    # Since it's deterministic planning (ADAPTIVE_CODE_PLAN / SEQUENTIAL)
    transaction_active = False
    try:
        begin_transaction(session_id, graph.graph_id)
        transaction_active = True
    except Exception as e:
        print(f"[Graph Scheduler] Could not begin wrapper transaction: {e}")

    step_count = 0
    while True:
        if getattr(exec_ctx, 'terminate', False):
            print(f"[Graph Scheduler] Terminated mid-execution.")
            break

        ready_nodes = graph.get_ready_nodes()
        
        if not ready_nodes:
            if graph.is_fully_completed():
                print(f"[Graph Scheduler] All nodes executed successfully.")
            else:
                print(f"[Graph Scheduler] Graph stalled (failures or unconnected nodes). Terminating loop.")
                yield "\n\n**Agent Guardrail:** Execution sequence stalled. A required subtask failed or was blocked by the transaction policy constraints."
            _persist_role_flow()
            break

        for candidate in ready_nodes:
            role_meta = _ensure_node_role(candidate)
            if str(candidate.node_id).upper() != "FINAL" and role_meta["role"] == "synthesizer":
                candidate.status = NodeStatus.FAILED
                _finalize_role_trace(candidate, status="FAILED", tool_calls=0, retries=0)
                yield _emit_info(
                    "progress",
                    agent_state="invalid_role_assignment",
                    node_id=candidate.node_id,
                    role=role_meta["role"],
                    mode=mode_label,
                    role_fallback_applied=bool(role_meta["role_fallback_applied"]),
                    role_resolution_source=role_meta["role_resolution_source"],
                    label=f"Invalid synthesizer role on non-FINAL node {candidate.node_id}",
                )
        parallel_tool_nodes: List[PlanNode] = []
        for node in ready_nodes:
            if node.status != NodeStatus.READY:
                continue
            if node.action_type != "TOOL_CALL":
                continue
            if not node.payload.get("tool"):
                continue
            role_meta = _ensure_node_role(node)
            if role_meta["role"] != "researcher":
                continue
            parallel_tool_nodes.append(node)
        _persist_role_flow()
        if len(parallel_tool_nodes) > 1:
            for node in parallel_tool_nodes:
                node.status = NodeStatus.RUNNING
                _begin_role_trace(node)
                if callable(resolve_node_policy):
                    try:
                        resolved_policy = resolve_node_policy(str(getattr(node, "role", "executor")))
                        if isinstance(resolved_policy, tuple) and len(resolved_policy) >= 2 and isinstance(resolved_policy[1], dict):
                            node.payload["_effective_policy_meta"] = dict(resolved_policy[1])
                        elif isinstance(resolved_policy, dict):
                            node.payload["_effective_policy_meta"] = dict(resolved_policy)
                    except Exception:
                        node.payload["_effective_policy_meta"] = {"policy_resolution_error": True}
            step_count += len(parallel_tool_nodes)
            print(f"[Graph Scheduler] Executing {len(parallel_tool_nodes)} independent planned tools in parallel.")
            allowed_tools, tools_enabled = _resolve_allowed_tools(agent_result, state)
            if not tools_enabled:
                for node in parallel_tool_nodes:
                    tool_name = str(node.payload.get("tool") or "").strip()
                    graph.mark_failed(node.node_id)
                    _finalize_role_trace(node, status="FAILED", tool_calls=0, retries=0)
                    node_meta = _node_stream_meta(node, mode_label)
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id} blocked: planned tool {tool_name} is not allowed for this query context.",
                        }
                    )
                    yield _emit_tool_result_info(
                        node_id=node.node_id,
                        tool=tool_name,
                        status="blocked",
                        args_preview=_safe_preview(node.payload.get("instruction") or user_query),
                        error="Tool not allowed for this query context",
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                _persist_role_flow()
                continue

            had_blocked_without_fallback = False
            allow_fallback_swaps = state is not None
            for node in parallel_tool_nodes:
                tool_name = str(node.payload.get("tool") or "").strip()
                if tool_name in allowed_tools:
                    continue
                fallback_tool = (
                    _select_fallback_retrieval_tool(allowed_tools)
                    if allow_fallback_swaps and _is_retrieval_tool(tool_name)
                    else None
                )
                if fallback_tool and fallback_tool != tool_name:
                    node.payload["tool"] = fallback_tool
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id}: fallback tool swap {tool_name} -> {fallback_tool} due to permission gate.",
                        }
                    )
                    yield _emit_info(
                        "planning",
                        topic=f"Tool {tool_name} blocked; using {fallback_tool} instead",
                        node_id=node.node_id,
                        role=str(getattr(node, "role", "executor")),
                        mode=mode_label,
                        role_fallback_applied=bool(getattr(node, "role_fallback_applied", False)),
                        role_resolution_source=str(getattr(node, "role_resolution_source", "unknown")),
                    )
                    continue
                had_blocked_without_fallback = True
                graph.mark_failed(node.node_id)
                _finalize_role_trace(node, status="FAILED", tool_calls=0, retries=0)
                node_meta = _node_stream_meta(node, mode_label)
                messages.append(
                    {
                        "role": "system",
                        "content": f"Node {node.node_id} blocked: planned tool {tool_name} is not allowed for this query context.",
                    }
                )
                yield _emit_info(
                    "tool_blocked",
                    node_id=node.node_id,
                    tool=tool_name,
                    reason="Tool not allowed for this query context",
                    role=node_meta["role"],
                    mode=node_meta["mode"],
                    role_fallback_applied=node_meta["role_fallback_applied"],
                    role_resolution_source=node_meta["role_resolution_source"],
                )
                yield _emit_tool_result_info(
                    node_id=node.node_id,
                    tool=tool_name,
                    status="blocked",
                    args_preview=_safe_preview(node.payload.get("instruction") or user_query),
                    error="Tool not allowed for this query context",
                    role=node_meta["role"],
                    mode=node_meta["mode"],
                    role_fallback_applied=node_meta["role_fallback_applied"],
                    role_resolution_source=node_meta["role_resolution_source"],
                )
            if had_blocked_without_fallback:
                _persist_role_flow()
                continue
            for node in parallel_tool_nodes:
                tool_name = str(node.payload.get("tool") or "").strip()
                node_meta = _node_stream_meta(node, mode_label)
                yield _emit_info(
                    "tool_call",
                    node_id=node.node_id,
                    tool=tool_name,
                    args_preview=_safe_preview(node.payload.get("instruction") or user_query),
                    role=node_meta["role"],
                    mode=node_meta["mode"],
                    role_fallback_applied=node_meta["role_fallback_applied"],
                    role_resolution_source=node_meta["role_resolution_source"],
                )
            raw_results = await asyncio.gather(
                *[
                    _execute_planned_tool_node(node, user_query, session_id, exec_ctx, graph)
                    for node in parallel_tool_nodes
                ],
                return_exceptions=True,
            )

            results: List[Tuple[PlanNode, str, ToolCall, Any, int]] = []
            parallel_exception_count = 0
            for node, raw_result in zip(parallel_tool_nodes, raw_results):
                if isinstance(raw_result, Exception):
                    parallel_exception_count += 1
                    node_meta = _node_stream_meta(node, mode_label)
                    tool_name = str(node.payload.get("tool") or "").strip()
                    error_text = _safe_preview(raw_result, 220)
                    _append_tool_runtime(
                        graph=graph,
                        tool_name=tool_name,
                        success=False,
                        latency_ms=0,
                        hard_error=True,
                    )
                    if _is_retrieval_tool(tool_name):
                        graph.mark_completed(
                            node.node_id,
                            {
                                "command": tool_name,
                                "result": {"warning": "parallel_tool_exception", "reason": error_text},
                                "source": tool_name,
                                "confidence": "low",
                                "trust": "unverified",
                            },
                        )
                        _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0, parallel_exception_count=1)
                    else:
                        graph.mark_failed(node.node_id)
                        _finalize_role_trace(node, status="FAILED", tool_calls=1, retries=0, parallel_exception_count=1)
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id} failed during parallel execution via planned tool {tool_name}: {error_text}",
                        }
                    )
                    print(f"[Graph Scheduler] Parallel tool exception ({node.node_id}/{tool_name}): {error_text}")
                    yield _emit_tool_result_info(
                        node_id=node.node_id,
                        tool=tool_name,
                        status="failed",
                        args_preview=_safe_preview(node.payload.get("instruction") or user_query),
                        error=error_text,
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                    continue
                results.append(raw_result)

            successful_parallel_results: List[Tuple[PlanNode, str, Any]] = []
            for node, tool_name, tool_call, tool_result, latency_ms in results:
                node_meta = _node_stream_meta(node, mode_label)
                exec_ctx.tool_results.append(tool_result)
                if tool_result.success:
                    _append_tool_runtime(
                        graph=graph,
                        tool_name=tool_name,
                        success=True,
                        latency_ms=int(latency_ms),
                        hard_error=False,
                    )
                    tool_name, tool_call, tool_result, retried = await _maybe_retry_low_relevance_search(
                        node=node,
                        tool_name=tool_name,
                        tool_call=tool_call,
                        tool_result=tool_result,
                        exec_ctx=exec_ctx,
                        user_query=user_query,
                        session_scope=session_id,
                    )
                    if retried:
                        exec_ctx.tool_results.append(tool_result)
                        yield _emit_info(
                            "planning",
                            topic=f"Refined search query for higher relevance ({tool_name})",
                            node_id=node.node_id,
                            role=node_meta["role"],
                            mode=node_meta["mode"],
                            role_fallback_applied=node_meta["role_fallback_applied"],
                            role_resolution_source=node_meta["role_resolution_source"],
                        )
                    tool_result.data = _dedupe_tool_result_data(getattr(tool_result, "data", None))
                    graph.mark_completed(
                        node.node_id,
                        {
                            "command": tool_name,
                            "result": tool_result.data,
                            "source": getattr(tool_result, "source", tool_name),
                            "confidence": getattr(tool_result, "confidence", "medium"),
                            "trust": getattr(tool_result, "trust", "verified"),
                        },
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id} completed via planned tool {tool_name}. Result: {str(tool_result.data)[:8000]}",
                        }
                    )
                    successful_parallel_results.append((node, tool_name, tool_result))
                    _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=1 if retried else 0)
                    yield _emit_tool_result_info(
                        node_id=node.node_id,
                        tool=tool_name,
                        status="ok",
                        args_preview=_safe_preview(tool_call.args),
                        tool_result=tool_result,
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                else:
                    _append_tool_runtime(
                        graph=graph,
                        tool_name=tool_name,
                        success=False,
                        latency_ms=int(latency_ms),
                        hard_error=not _is_soft_tool_failure(tool_result),
                    )
                    if _is_soft_tool_failure(tool_result):
                        graph.mark_completed(node.node_id, _soft_tool_result_payload(tool_name, tool_result))
                        print(f"[Graph Scheduler] Planned tool {tool_name} throttled: {tool_result.error or 'unknown error'}")
                        messages.append(
                            {
                                "role": "system",
                                "content": f"Node {node.node_id} throttled via planned tool {tool_name}: {tool_result.error or 'unknown error'}",
                            }
                        )
                        _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0)
                        yield _emit_tool_result_info(
                            node_id=node.node_id,
                            tool=tool_name,
                            status="throttled",
                            args_preview=_safe_preview(node.payload.get("instruction") or user_query),
                            tool_result=tool_result,
                            error=str(tool_result.error or "unknown error")[:200],
                            role=node_meta["role"],
                            mode=node_meta["mode"],
                            role_fallback_applied=node_meta["role_fallback_applied"],
                            role_resolution_source=node_meta["role_resolution_source"],
                        )
                    else:
                        if _is_retrieval_tool(tool_name):
                            graph.mark_completed(node.node_id, _soft_tool_result_payload(tool_name, tool_result))
                            print(f"[Graph Scheduler] Planned retrieval tool {tool_name} failed; continuing with fallback synthesis.")
                            messages.append(
                                {
                                    "role": "system",
                                    "content": f"Node {node.node_id} fallback-completed after {tool_name} failure: {tool_result.error or 'unknown error'}",
                                }
                            )
                            _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0)
                            yield _emit_tool_result_info(
                                node_id=node.node_id,
                                tool=tool_name,
                                status="failed",
                                args_preview=_safe_preview(node.payload.get("instruction") or user_query),
                                tool_result=tool_result,
                                error=str(tool_result.error or "unknown error")[:200],
                                role=node_meta["role"],
                                mode=node_meta["mode"],
                                role_fallback_applied=node_meta["role_fallback_applied"],
                                role_resolution_source=node_meta["role_resolution_source"],
                            )
                        else:
                            graph.mark_failed(node.node_id)
                            print(f"[Graph Scheduler] Planned tool {tool_name} failed: {tool_result.error or 'unknown error'}")
                            messages.append(
                                {
                                    "role": "system",
                                    "content": f"Node {node.node_id} failed via planned tool {tool_name}: {tool_result.error or 'unknown error'}",
                                }
                            )
                            _finalize_role_trace(node, status="FAILED", tool_calls=1, retries=0)
                            yield _emit_tool_result_info(
                                node_id=node.node_id,
                                tool=tool_name,
                                status="failed",
                                args_preview=_safe_preview(node.payload.get("instruction") or user_query),
                                tool_result=tool_result,
                                error=str(tool_result.error or "unknown error")[:200],
                                role=node_meta["role"],
                                mode=node_meta["mode"],
                                role_fallback_applied=node_meta["role_fallback_applied"],
                                role_resolution_source=node_meta["role_resolution_source"],
                            )
            if successful_parallel_results:
                merge_snapshot = _build_parallel_merge_snapshot(successful_parallel_results)
                if int(merge_snapshot.get("unique_sources", 0) or 0) > 0:
                    graph.metadata = dict(graph.metadata or {})
                    graph.metadata["parallel_merge_snapshot"] = merge_snapshot
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"Parallel merge dedup: {merge_snapshot.get('tool_runs', 0)} tool runs -> "
                                f"{merge_snapshot.get('unique_sources', 0)} unique sources."
                            ),
                        }
                    )
                    yield _emit_info(
                        "parallel_merge",
                        tool_runs=int(merge_snapshot.get("tool_runs", 0) or 0),
                        unique_sources=int(merge_snapshot.get("unique_sources", 0) or 0),
                        mode=mode_label,
                    )
            graph.metadata = dict(graph.metadata or {})
            graph.metadata["parallel_batches_total"] = int(graph.metadata.get("parallel_batches_total", 0) or 0) + 1
            graph.metadata["parallel_tools_total"] = int(graph.metadata.get("parallel_tools_total", 0) or 0) + len(parallel_tool_nodes)
            graph.metadata["parallel_success_total"] = int(graph.metadata.get("parallel_success_total", 0) or 0) + len(successful_parallel_results)
            graph.metadata["parallel_exception_count_total"] = int(graph.metadata.get("parallel_exception_count_total", 0) or 0) + int(parallel_exception_count)
            yield _emit_info(
                "parallel_batch_stats",
                parallel_tools=len(parallel_tool_nodes),
                parallel_success=len(successful_parallel_results),
                parallel_exception_count=int(parallel_exception_count),
                mode=mode_label,
            )
            _persist_role_flow()
            continue

        # Fall back to single-node execution when work is dependent or reasoning-led.
        eligible_nodes = [candidate for candidate in ready_nodes if candidate.status == NodeStatus.READY]
        if not eligible_nodes:
            _persist_role_flow()
            continue
        node = eligible_nodes[0]
        node.status = NodeStatus.RUNNING
        _begin_role_trace(node)
        if callable(resolve_node_policy):
            try:
                resolved_policy = resolve_node_policy(str(getattr(node, "role", "executor")))
                if isinstance(resolved_policy, tuple) and len(resolved_policy) >= 2 and isinstance(resolved_policy[1], dict):
                    node.payload["_effective_policy_meta"] = dict(resolved_policy[1])
                elif isinstance(resolved_policy, dict):
                    node.payload["_effective_policy_meta"] = dict(resolved_policy)
            except Exception:
                node.payload["_effective_policy_meta"] = {"policy_resolution_error": True}
        step_count += 1
        node_meta = _node_stream_meta(node, mode_label)
        
        print(
            f"[Graph Scheduler] Executing Node {node.node_id} "
            f"(Action: {node.action_type}, Role: {node_meta['role']}) | Step {step_count}"
        )

        if node.action_type == "TOOL_CALL" and node.payload.get("tool"):
            tool_name, tool_call = _build_planned_tool_call(node, user_query, task_scope, graph)
            allowed_tools, tools_enabled = _resolve_allowed_tools(agent_result, state)
            if tools_enabled and node_meta["role"] == "researcher" and _is_retrieval_tool(tool_name):
                prior_tool = str(tool_name)
                reliability_tool = _choose_reliability_preferred_retrieval_tool(
                    current_tool=tool_name,
                    allowed_tools=allowed_tools,
                )
                if reliability_tool and reliability_tool != tool_name:
                    node.payload["tool"] = reliability_tool
                    tool_name, tool_call = _build_planned_tool_call(node, user_query, task_scope, graph)
                    yield _emit_info(
                        "planning",
                        topic=f"Reliability routing preferred {reliability_tool} over {prior_tool}",
                        node_id=node.node_id,
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
            if not tools_enabled or tool_name not in allowed_tools:
                allow_fallback_swap = state is not None
                fallback_tool = (
                    _select_fallback_retrieval_tool(allowed_tools)
                    if allow_fallback_swap and _is_retrieval_tool(tool_name)
                    else None
                )
                if fallback_tool and fallback_tool != tool_name and tools_enabled:
                    node.payload["tool"] = fallback_tool
                    tool_name, tool_call = _build_planned_tool_call(node, user_query, task_scope, graph)
                    print(f"[Graph Scheduler] Planned tool {tool_name} selected via fallback permission swap.")
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id}: fallback tool swap applied due to permission gate.",
                        }
                    )
                    yield _emit_info(
                        "planning",
                        topic=f"Tool blocked by gate; switched to {tool_name}",
                        node_id=node.node_id,
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                else:
                    graph.mark_failed(node.node_id)
                    _finalize_role_trace(node, status="FAILED", tool_calls=0, retries=0)
                    print(f"[Graph Scheduler] Planned tool {tool_name} blocked by permission gate.")
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id} blocked: planned tool {tool_name} is not allowed.",
                        }
                    )
                    yield _emit_info(
                        "tool_blocked",
                        node_id=node.node_id,
                        tool=tool_name,
                        reason="Tool blocked by permission gate",
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                    yield _emit_tool_result_info(
                        node_id=node.node_id,
                        tool=tool_name,
                        status="blocked",
                        args_preview=_safe_preview(tool_call.args),
                        error="Tool blocked by permission gate",
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                    _persist_role_flow()
                    continue
            print(f"[Graph Scheduler] Deterministically executing planned tool {tool_name}")
            yield _emit_info(
                "tool_call",
                node_id=node.node_id,
                tool=tool_name,
                args_preview=_safe_preview(tool_call.args),
                role=node_meta["role"],
                mode=node_meta["mode"],
                role_fallback_applied=node_meta["role_fallback_applied"],
                role_resolution_source=node_meta["role_resolution_source"],
            )
            tool_started_at = time.time()
            tool_result = await execute_tool(tool_call, exec_ctx)
            tool_latency_ms = int(max(0, round((time.time() - tool_started_at) * 1000)))
            exec_ctx.tool_results.append(tool_result)
            if tool_result.success:
                _append_tool_runtime(
                    graph=graph,
                    tool_name=tool_name,
                    success=True,
                    latency_ms=tool_latency_ms,
                    hard_error=False,
                )
                tool_name, tool_call, tool_result, retried = await _maybe_retry_low_relevance_search(
                    node=node,
                    tool_name=tool_name,
                    tool_call=tool_call,
                    tool_result=tool_result,
                    exec_ctx=exec_ctx,
                    user_query=user_query,
                    session_scope=task_scope,
                )
                if retried:
                    exec_ctx.tool_results.append(tool_result)
                    yield _emit_info(
                        "planning",
                        topic=f"Refined search query for higher relevance ({tool_name})",
                        node_id=node.node_id,
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                graph.mark_completed(
                    node.node_id,
                    {
                        "command": tool_name,
                        "result": tool_result.data,
                        "source": getattr(tool_result, "source", tool_name),
                        "confidence": getattr(tool_result, "confidence", "medium"),
                        "trust": getattr(tool_result, "trust", "verified"),
                    },
                )
                messages.append(
                    {
                        "role": "system",
                        "content": f"Node {node.node_id} completed via planned tool {tool_name}. Result: {str(tool_result.data)[:8000]}",
                    }
                )
                _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=1 if retried else 0)
                yield _emit_tool_result_info(
                    node_id=node.node_id,
                    tool=tool_name,
                    status="ok",
                    args_preview=_safe_preview(tool_call.args),
                    tool_result=tool_result,
                    role=node_meta["role"],
                    mode=node_meta["mode"],
                    role_fallback_applied=node_meta["role_fallback_applied"],
                    role_resolution_source=node_meta["role_resolution_source"],
                )
                _persist_role_flow()
                continue

            _append_tool_runtime(
                graph=graph,
                tool_name=tool_name,
                success=False,
                latency_ms=tool_latency_ms,
                hard_error=not _is_soft_tool_failure(tool_result),
            )
            if _is_soft_tool_failure(tool_result):
                graph.mark_completed(node.node_id, _soft_tool_result_payload(tool_name, tool_result))
                print(f"[Graph Scheduler] Planned tool {tool_name} throttled: {tool_result.error or 'unknown error'}")
                messages.append(
                    {
                        "role": "system",
                        "content": f"Node {node.node_id} throttled via planned tool {tool_name}: {tool_result.error or 'unknown error'}",
                    }
                )
                yield _emit_tool_result_info(
                    node_id=node.node_id,
                    tool=tool_name,
                    status="throttled",
                    args_preview=_safe_preview(tool_call.args),
                    tool_result=tool_result,
                    error=str(tool_result.error or "unknown error")[:200],
                    role=node_meta["role"],
                    mode=node_meta["mode"],
                    role_fallback_applied=node_meta["role_fallback_applied"],
                    role_resolution_source=node_meta["role_resolution_source"],
                )
                _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0)
            else:
                if _is_retrieval_tool(tool_name):
                    graph.mark_completed(node.node_id, _soft_tool_result_payload(tool_name, tool_result))
                    print(f"[Graph Scheduler] Planned retrieval tool {tool_name} failed; continuing with fallback synthesis.")
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id} fallback-completed after {tool_name} failure: {tool_result.error or 'unknown error'}",
                        }
                    )
                    yield _emit_tool_result_info(
                        node_id=node.node_id,
                        tool=tool_name,
                        status="failed",
                        args_preview=_safe_preview(tool_call.args),
                        tool_result=tool_result,
                        error=str(tool_result.error or "unknown error")[:200],
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                    _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0)
                else:
                    graph.mark_failed(node.node_id)
                    print(f"[Graph Scheduler] Planned tool {tool_name} failed: {tool_result.error or 'unknown error'}")
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Node {node.node_id} failed via planned tool {tool_name}: {tool_result.error or 'unknown error'}",
                        }
                    )
                    yield _emit_tool_result_info(
                        node_id=node.node_id,
                        tool=tool_name,
                        status="failed",
                        args_preview=_safe_preview(tool_call.args),
                        tool_result=tool_result,
                        error=str(tool_result.error or "unknown error")[:200],
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                    _finalize_role_trace(node, status="FAILED", tool_calls=1, retries=0)
            _persist_role_flow()
            continue
        
        # Build prompt injection for this node's isolated target
        target_payload = json.dumps(node.payload)
        node_role = str(node_meta.get("role") or "executor")
        if node_role == "researcher":
            role_instruction = (
                "[ROLE: RESEARCHER]\nUse tools first. Prefer evidence-rich retrieval. "
                "Output one TOOL_CALL when tool data is needed."
            )
        elif node_role == "critic":
            role_instruction = (
                "[ROLE: CRITIC]\nPrioritize strict validation and repair. "
                "If evidence is missing, call one retrieval tool. Otherwise return concise checks."
            )
        elif node_role == "synthesizer":
            role_instruction = (
                "[ROLE: SYNTHESIZER]\nThis is FINAL synthesis. Do not call tools. "
                "Produce the final user-facing answer from available evidence only."
            )
        elif node_role == "planner":
            role_instruction = (
                "[ROLE: PLANNER]\nReasoning-first step. Avoid tool calls unless absolutely required."
            )
        else:
            role_instruction = (
                "[ROLE: EXECUTOR]\nExecute the node deterministically. Use a tool only when required by the node."
            )
        messages.append({
            "role": "user",
            "content": (
                f"Graph Execution Step:\nYou must now resolve node {node.node_id}.\n"
                f"Objective: {target_payload}\n\n"
                f"{role_instruction}\n\n"
                "Respond using ONE tool call if required, or pure text."
            ),
        })

        step_output = ""
        is_tool_call = False
        # -----------------------------
        # Generate LLM action for Node
        # -----------------------------
        stream = None
        try:
            stream = await client.chat.completions.create(**create_kwargs)
        except Exception as node_err:
            # Node retry policy: one fallback retry with a minimal base prompt.
            yield _emit_info(
                "progress",
                agent_state="retrying_node",
                node_id=node.node_id,
                role=node_meta["role"],
                mode=node_meta["mode"],
                role_fallback_applied=node_meta["role_fallback_applied"],
                role_resolution_source=node_meta["role_resolution_source"],
                label=f"Retrying node {node.node_id} after transient model error",
                reason=str(node_err)[:140],
            )
            fallback_kwargs = dict(create_kwargs)
            fallback_messages = list(fallback_kwargs.get("messages", []))
            if fallback_messages:
                fallback_messages[0] = {
                    "role": "system",
                    "content": "You are in fallback mode for a failed DAG node. Respond deterministically and avoid unnecessary tool calls.",
                }
                fallback_kwargs["messages"] = fallback_messages
            try:
                stream = await client.chat.completions.create(**fallback_kwargs)
            except Exception as retry_err:
                graph.mark_failed(node.node_id)
                _finalize_role_trace(node, status="FAILED", tool_calls=0, retries=1)
                messages.append({
                    "role": "system",
                    "content": f"Node {node.node_id} failed after retry: {retry_err}",
                })
                _persist_role_flow()
                continue

        async for chunk in stream:
            if hasattr(chunk, "usage") and getattr(chunk, "usage", None):
                r_tokens = getattr(chunk.usage, "reasoning_tokens", 0)
                if isinstance(chunk.usage, dict):
                    r_tokens = chunk.usage.get("reasoning_tokens", 0)
                if r_tokens:
                    yield f"[INFO]INTEL:{{\"reasoning_tokens\": {r_tokens}}}"
            if hasattr(chunk, "choices") and chunk.choices and getattr(chunk.choices[0].delta, 'content', None):
                token = chunk.choices[0].delta.content
                step_output += token
                if "TOOL_CALL" in step_output:
                    is_tool_call = True

        factual_markers = ("who is", "founder", "ceo", "affiliation", "cbse", "matric", "price", "latest", "today", "current")
        q_low = (user_query or "").lower()
        needs_forced_tool = any(m in q_low for m in factual_markers)
        role_allows_force = node_role not in {"planner", "synthesizer"}
        if agent_result.tool_allowed and role_allows_force and needs_forced_tool and "TOOL_CALL:" not in step_output:
            messages.append({"role": "assistant", "content": step_output})
            messages.append({
                "role": "user",
                "content": "Factual query detected. Output ONLY one TOOL_CALL now. Use search_web first.",
            })
            node.status = NodeStatus.PENDING
            _finalize_role_trace(node, status="PENDING", tool_calls=0, retries=1)
            _persist_role_flow()
            continue

        # Intercept tool parsing
        had_tool_call = "TOOL_CALL:" in step_output
        if node_role == "synthesizer" and had_tool_call:
            step_output = re.sub(r"TOOL_CALL:.*", "", step_output, flags=re.IGNORECASE).strip()
            had_tool_call = False
        while "TOOL_CALL:" in step_output:
            first_call_idx = step_output.find("TOOL_CALL:")
            
            # Find boundary for first block
            newline_idx = step_output.find("\n", first_call_idx)
            
            # Since consecutive toolcalls might be on same line without \n natively emitted by AI...
            # Search for the *NEXT* TOOL_CALL natively or use end of string
            next_call_idx = step_output.find("TOOL_CALL:", first_call_idx + 10)

            # Determine slice boundaries to parse
            if newline_idx != -1 and (next_call_idx == -1 or newline_idx < next_call_idx):
                end_idx = newline_idx
            elif next_call_idx != -1:
                end_idx = next_call_idx
            else:
                end_idx = len(step_output)
                
            truncated_step_output = step_output[:end_idx].strip()
            
            # Strip the extracted block from the string stream buffer to advance parsing loop naturally!
            step_output = step_output[end_idx:]

            # parse_tool_calls returns [ToolCall] now...
            tools_found = parse_tool_calls(truncated_step_output)
            
            if not tools_found:
                 break # Ensure safe exit if parse failed on garbage data to avoid infinite loop
                 
            for tool_call in tools_found:
            
                # Non-transactional gate check inside graph scope
                from app.state.transaction_manager import TOOL_CLASSIFICATOR, ToolClass
                if tool_call and transaction_active:
                    t_class = TOOL_CLASSIFICATOR.get(tool_call.name, ToolClass.NON_TRANSACTIONAL)
                    if t_class == ToolClass.NON_TRANSACTIONAL:
                        exec_ctx.forced_finalize = True
                        exec_ctx.degraded = True
                        print(f"[Graph Scheduler] Banned NON_TRANSACTIONAL tool {tool_call.name} in transaction.")
                        graph.mark_failed(node.node_id)
                        _finalize_role_trace(node, status="FAILED", tool_calls=0, retries=0)
                        continue

                tool_executed_correctly = False

                allowed_tools, tools_enabled = _resolve_allowed_tools(agent_result, state)
                if tool_call and tools_enabled and tool_call.name in allowed_tools:
                    print(f"Executing tool {tool_call.name}")
                    yield _emit_info(
                        "tool_call",
                        node_id=node.node_id,
                        tool=tool_call.name,
                        args_preview=_safe_preview(tool_call.args),
                        role=node_meta["role"],
                        mode=node_meta["mode"],
                        role_fallback_applied=node_meta["role_fallback_applied"],
                        role_resolution_source=node_meta["role_resolution_source"],
                    )
                    
                    # Tool executes...
                    tool_started_at = time.time()
                    tool_result = await execute_tool(tool_call, exec_ctx)
                    tool_latency_ms = int(max(0, round((time.time() - tool_started_at) * 1000)))
                    exec_ctx.tool_results.append(tool_result)
                    
                    if tool_result.success:
                        _append_tool_runtime(
                            graph=graph,
                            tool_name=tool_call.name,
                            success=True,
                            latency_ms=tool_latency_ms,
                            hard_error=False,
                        )
                        graph.mark_completed(node.node_id, {"command": tool_call.name, "result": str(tool_result.data)})
                        messages.append({
                            "role": "system",
                            "content": f"Node {node.node_id} successfully completed. Result: {str(tool_result.data)[:8000]}"
                        })
                        yield _emit_tool_result_info(
                            node_id=node.node_id,
                            tool=tool_call.name,
                            status="ok",
                            args_preview=_safe_preview(tool_call.args),
                            tool_result=tool_result,
                            role=node_meta["role"],
                            mode=node_meta["mode"],
                            role_fallback_applied=node_meta["role_fallback_applied"],
                            role_resolution_source=node_meta["role_resolution_source"],
                        )
                        _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0)
                        tool_executed_correctly = True
                    else:
                        _append_tool_runtime(
                            graph=graph,
                            tool_name=tool_call.name,
                            success=False,
                            latency_ms=tool_latency_ms,
                            hard_error=not _is_soft_tool_failure(tool_result),
                        )
                        if _is_soft_tool_failure(tool_result) or _is_retrieval_tool(tool_call.name):
                            graph.mark_completed(node.node_id, _soft_tool_result_payload(tool_call.name, tool_result))
                            messages.append({
                                "role": "system",
                                "content": f"Node {node.node_id} fallback-completed after {tool_call.name} issue: {tool_result.error or 'unknown error'}"
                            })
                            fallback_status = "throttled" if _is_soft_tool_failure(tool_result) else "failed"
                            yield _emit_tool_result_info(
                                node_id=node.node_id,
                                tool=tool_call.name,
                                status=fallback_status,
                                args_preview=_safe_preview(tool_call.args),
                                tool_result=tool_result,
                                error=str(tool_result.error or "unknown error")[:200],
                                role=node_meta["role"],
                                mode=node_meta["mode"],
                                role_fallback_applied=node_meta["role_fallback_applied"],
                                role_resolution_source=node_meta["role_resolution_source"],
                            )
                            _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0)
                            tool_executed_correctly = True
                            continue
                        yield _emit_tool_result_info(
                            node_id=node.node_id,
                            tool=tool_call.name,
                            status="failed",
                            args_preview=_safe_preview(tool_call.args),
                            tool_result=tool_result,
                            error=str(tool_result.error or "unknown error")[:200],
                            role=node_meta["role"],
                            mode=node_meta["mode"],
                            role_fallback_applied=node_meta["role_fallback_applied"],
                            role_resolution_source=node_meta["role_resolution_source"],
                        )
                        # Node-Scoped Repair Loop
                        is_repaired = False
                        if strategy and strategy.repair_policy.get("enabled", False):
                            repair_max = strategy.repair_policy.get("max_attempts", 2)
                            print(f"[Graph Scheduler] Node {node.node_id} tool failed. Executing repair loop.")
                            
                            repair_cycle_result = repair_cycle(
                                failure=tool_result.error or "unknown",
                                context={"node_id": node.node_id, "generated_code": truncated_step_output},
                                max_attempts=repair_max,
                            )
                            
                            if repair_cycle_result.status == "repair_needed" and repair_cycle_result.final_failure_type:
                                repair_strat = generate_repair_strategy(tool_result.error or "unknown")
                                repair_prompt_text = build_repair_prompt(
                                    original_code=truncated_step_output,
                                    error=tool_result.error or "unknown",
                                    repair_strategy=repair_strat,
                                )
                                messages.append({"role": "assistant", "content": truncated_step_output})
                                messages.append({"role": "user", "content": repair_prompt_text})
                                
                                # We reset the node to PENDING so the next loop will retry it
                                node.status = NodeStatus.PENDING
                                _finalize_role_trace(node, status="PENDING", tool_calls=1, retries=1)
                                continue
                                
                            elif repair_cycle_result.status == "repair_failed":
                                # Strict rollback if repair fails on this node
                                print(f"[Graph Scheduler] Repair cycle exhausted on Node {node.node_id}. Triggering LIFO Rollback.")
                                if transaction_active:
                                    await rollback_transaction(session_id, graph.graph_id)
                                    transaction_active = False # Killed
                                    
                        if not is_repaired:
                            graph.mark_failed(node.node_id)
                            _finalize_role_trace(node, status="FAILED", tool_calls=1, retries=0)
                            messages.append({
                                "role": "system",
                                "content": f"Node {node.node_id} failed permanently. All dependent branches are now BLOCKED."
                            })
                            
                else:
                    # Text-only success (Tool hallucinated or blocked)
                    print(f"[Graph Scheduler] Tool disabled or malformed. Resolving as text completion.")
                    if is_tool_call and tool_call:
                        yield _emit_tool_result_info(
                            node_id=node.node_id,
                            tool=tool_call.name,
                            status="blocked",
                            args_preview=_safe_preview(tool_call.args),
                            error="Tool restricted by policy for this prompt",
                            role=node_meta["role"],
                            mode=node_meta["mode"],
                            role_fallback_applied=node_meta["role_fallback_applied"],
                            role_resolution_source=node_meta["role_resolution_source"],
                        )
                    
                if not tool_executed_correctly and node.status != NodeStatus.FAILED and node.status != NodeStatus.PENDING:
                        node_obj = graph.get_node(node.node_id)
                        current_result = node_obj.result or []
                        if not isinstance(current_result, list):
                            current_result = [current_result]
                            
                        # Safely extract tool data without crashing if tool_result wasn't assigned
                        t_name = tool_call.name if tool_call else "unknown_tool"
                        
                        try:
                            # If tool_result exists in scope, use its data
                            t_res_data = str(tool_result.data)
                        except NameError:
                            # If tool_result was never initialized because the tool wasn't authorized
                            t_res_data = "Blocked by safety policy or malformed call."
                            
                        current_result.append({"command": t_name, "result": t_res_data})
        if not had_tool_call:
            # Reasoning-only node: persist text output so processor can fall back
            # when no tool results exist for synthesis.
            graph.mark_completed(node.node_id, {"raw_output": step_output})
            _finalize_role_trace(node, status="COMPLETED", tool_calls=0, retries=0)
            if step_output and step_output.strip():
                messages.append({
                    "role": "assistant",
                    "content": step_output
                })
        elif node.status == NodeStatus.RUNNING:
            _finalize_role_trace(node, status="COMPLETED", tool_calls=1, retries=0)

        _persist_role_flow()

        # End of Node Loop Execution
    
    if transaction_active and graph.is_fully_completed():
        commit_transaction(session_id, graph.graph_id)
    _persist_role_flow()



