"""
Relyce AI - Tool Executor (Production Hardened)
Tool Execution Layer: Real backend tool execution with standardized contract.

Architecture:
  - App controls execution, not the LLM
  - All tools return a standard contract: {status, data, source, confidence}
  - Registry carries metadata for autonomy/permission integration
  - Timeout guard prevents agent freeze
  - Contract-based validation (deterministic)

Tools:
  - get_current_time()  → real datetime
  - search_web(query)   → serper API search (organic results)
"""
from __future__ import annotations

import asyncio
import json
import re
import os
import ast
import operator
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================
# CONFIGURATION
# ============================================

MAX_TOOL_CALLS = 2        # max tool invocations per request
MAX_RETRIES = 2           # retry failed tools up to N times
TOOL_TIMEOUT = 3          # seconds before tool execution is aborted
MAX_TOOL_PAYLOAD = 4000   # max characters returned by a tool

def truncate_payload(data: Any) -> Any:
    """Truncates massive string payloads to prevent token explosion."""
    if isinstance(data, str) and len(data) > MAX_TOOL_PAYLOAD:
        return data[:MAX_TOOL_PAYLOAD] + "... [truncated]"
    return data


# ============================================
# STANDARD TOOL CONTRACT
# ============================================
# Every tool MUST return:
# {
#     "status":     "success" | "failure",
#     "data":       <payload> | None,
#     "source":     "<tool_name>",
#     "confidence": "high" | "medium" | "low"
# }


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class ToolCall:
    """Parsed tool call from LLM output."""
    name: str = ""
    args: str = ""
    raw: str = ""
    session_id: str = ""
    user_id: str = ""


@dataclass
class ToolResult:
    """Result of a tool execution (internal tracking)."""
    tool_name: str = ""
    success: bool = False
    data: Any = None
    source: str = ""
    confidence: str = "low"
    trust: str = "verified"
    error: str = ""
    retries: int = 0


@dataclass
class ExecutionContext:
    """Tracks tool execution state for a single request."""
    tool_calls_made: int = 0
    retry_count: int = 0              # total retries across all tools
    tool_results: List[ToolResult] = field(default_factory=list)
    degraded: bool = False            # True if any tool failed
    degradation_reasons: List[str] = field(default_factory=list)
    forced_finalize: bool = False     # True if step limit hit
    untrusted_seen: bool = False      # tracking for system prompt injection
    
    # User Control Extensions
    terminate: bool = False
    confirmation: Optional[bool] = None
    step_count: int = 0
    memory_hits: int = 0         
    pause_event: Optional[asyncio.Event] = None
    loop_break_triggered: bool = False
    user_id: str = ""              # owner of the request
    
    # --- Controller Discipline Tracking ---
    executed_tools: set = field(default_factory=set)
    final_delivery: bool = False

    @property
    def total_operations(self) -> int:
        """Total operations: tool calls + retries (for completion guard)."""
        return self.tool_calls_made + self.retry_count


# ============================================
# TOOL IMPLEMENTATIONS
# ============================================

def _tool_get_current_time(args: str = "") -> Dict:
    """Returns the real current date and time (standard contract)."""
    now = datetime.now()
    return {
        "status": "success",
        "data": {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "iso": now.isoformat(),
            "display": now.strftime("%B %d, %Y at %I:%M %p"),
        },
        "source": "clock",
        "confidence": "high",
    }


# Session-based search latency control
LAST_SEARCH_TIME: Dict[str, float] = {}

def can_call_search(session_id: str) -> bool:
    if not session_id:
        return True
    now = time.time()
    last = LAST_SEARCH_TIME.get(session_id, 0)
    if now - last < 5:
        return False
    LAST_SEARCH_TIME[session_id] = now
    return True


async def _tool_search_web(args: str = "", session_id: str = "") -> Dict:
    """
    Executes a real web search via Serper API.
    Extracts organic results from Serper response format.
    Returns standard contract.
    """
    if not can_call_search(session_id):
        return {
            "status": "failure",
            "data": "Search cooldown active. Please wait a moment before searching again.",
            "source": "search_cooldown",
            "confidence": "low",
        }

    try:
        from app.llm.router import execute_serper_batch, get_tools_for_mode
        tools = get_tools_for_mode("normal")

        # Use the Search endpoint
        endpoint = tools.get("Search", tools.get("search", ""))
        if not endpoint:
            return {
                "status": "failure",
                "data": None,
                "source": "serper",
                "confidence": "low",
            }

        raw_result = await execute_serper_batch(endpoint, [args], param_key="q")

        # Parse Serper response: result is a list of dicts with "organic" key
        cleaned = []
        if isinstance(raw_result, list) and len(raw_result) > 0:
            first = raw_result[0]
            organic = first.get("organic", []) if isinstance(first, dict) else []
            if not organic:
                return {
                    "status": "failure",
                    "data": None,
                    "source": "serper_no_organic",
                    "confidence": "low",
                }
            for r in organic[:5]:
                if isinstance(r, dict):
                    cleaned.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "link": r.get("link", ""),
                    })
        elif isinstance(raw_result, dict):
            # Single response (non-batch)
            organic = raw_result.get("organic", [])
            if not organic:
                return {
                    "status": "failure",
                    "data": None,
                    "source": "serper_no_organic",
                    "confidence": "low",
                }
            for r in organic[:5]:
                if isinstance(r, dict):
                    cleaned.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "link": r.get("link", ""),
                    })

        if cleaned:
            return {
                "status": "success",
                "data": cleaned,
                "source": "serper",
                "confidence": "high",
            }
        else:
            return {
                "status": "failure",
                "data": None,
                "source": "serper",
                "confidence": "low",
            }

    except Exception as e:
        return {
            "status": "failure",
            "data": None,
            "source": "serper",
            "confidence": "low",
        }


SAFE_ROOT = "data/"

async def _tool_read_file(args: str = "") -> Dict:
    """Reads a local file returning first 5000 chars securely."""
    try:
        abs_path = os.path.abspath(args.strip())
        abs_safe = os.path.abspath(SAFE_ROOT)
        
        if not abs_path.startswith(abs_safe):
            return {
                "status": "failure",
                "data": f"Access denied. Path '{args}' is outside safe sandbox.",
                "source": "local_file",
                "confidence": "low"
            }
            
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return {
                "status": "failure",
                "data": f"File not found: '{args}'",
                "source": "local_file",
                "confidence": "low"
            }
            
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()[:5000]

        return {
            "status": "success",
            "data": content,
            "source": "local_file",
            "confidence": "low", # Force LLM to treat as UNTRUSTED
            "trust": "unverified"
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "local_file",
            "confidence": "low"
        }

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _tool_calculate(args: str = "") -> Dict:
    """Evaluates numerical expression securely using AST (no eval)."""
    try:
        def eval_node(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")): # fallback for python < 3.8
                return node.n
            elif isinstance(node, getattr(ast, "BinOp")):
                return OPS[type(node.op)](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, getattr(ast, "UnaryOp")):
                return OPS[type(node.op)](eval_node(node.operand))
            else:
                raise ValueError("Unsafe or unsupported mathematical expression")

        tree = ast.parse(args.strip(), mode='eval')
        result = eval_node(tree.body)

        return {
            "status": "success",
            "data": result,
            "source": "calculator",
            "confidence": "high"
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "calculator",
            "confidence": "low"
        }

async def _tool_retrieve_knowledge(args: str = "") -> Dict:
    """Fetches knowledge locally using mocked knowledge store interface."""
    try:
        # Mocked for now: logic to connect to true KB should be linked here
        data = f"Knowledge successfully retrieved for topic: {args}"

        return {
            "status": "success",
            "data": data,
            "source": "knowledge_base",
            "confidence": "high",
            "freshness": "cached"
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "knowledge_base",
            "confidence": "low"
        }

async def _tool_search_documents(args: str = "", user_id: str = "", session_id: str = "") -> Dict:
    """
    Searches user-uploaded documents (RAG) semantically.
    Returns the most relevant text chunks from the indexed documents.
    """
    if not user_id or user_id == "anonymous":
        return {
            "status": "failure",
            "data": "Search documents requires a registered user account.",
            "source": "rag_search",
            "confidence": "low"
        }

    if not session_id:
        return {
            "status": "failure",
            "data": "Search documents requires a valid session_id.",
            "source": "rag_search",
            "confidence": "low"
        }

    try:
        from app.rag.retrieval import retrieve_rag_context
        # We use retrieve_rag_context which handles Weaviate connection and embedding
        context = await retrieve_rag_context(user_id=user_id, query=args, session_id=session_id, top_k=5)
        
        if not context:
            return {
                "status": "success",
                "data": "No matching information found in your documents.",
                "source": "rag_search",
                "confidence": "medium"
            }

        return {
            "status": "success",
            "data": context,
            "source": "rag_search",
            "confidence": "high"
        }
    except Exception as e:
        print(f"[ToolExecutor] RAG search error: {e}")
        return {
            "status": "failure",
            "data": f"Error searching documents: {str(e)}",
            "source": "rag_search",
            "confidence": "low"
        }


# ============================================
# TOOL REGISTRY (with metadata)
# ============================================
# Each tool carries integration metadata for:
#   - Autonomy Guard (risk, reversible)
#   - Permission Gate (freshness)
#   - Execution (func, is_async)

TOOLS: Dict[str, Dict] = {
    "get_current_time": {
        "func": _tool_get_current_time,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_web": {
        "func": _tool_search_web,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "read_file": {
        "func": _tool_read_file,
        "is_async": True,
        "reversible": True,
        "risk": "medium",
        "freshness": "static",
    },
    "calculate": {
        "func": _tool_calculate,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "retrieve_knowledge": {
        "func": _tool_retrieve_knowledge,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "cached",
    },
    "web_fetch": {
        "func": None,  # Lazy-loaded from app.input_processing.web_fetch
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_documents": {
        "func": _tool_search_documents,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
}


# ============================================
# TOOL PERMISSION GATE
# ============================================

def determine_tool_permission(
    action_type: str,
    requires_research: bool,
    requires_comparison: bool,
    is_time_sensitive: bool,
    autonomy_action: str,
) -> bool:
    """
    The APP decides if tools are allowed — not the LLM.

    Tools are ENABLED when:
      - action_type == "ACTION" (immediate execution)
      - OR requires_research (needs external data)
      - OR is_time_sensitive (needs live/fresh data)

    Comparison fast-path:
      - Comparisons do NOT get tool eligibility unless time-sensitive.

    Autonomy gate:
      - Only "confirm" blocks tools (high-risk needs user approval first).
    """
    # Comparison fast-path: skip tools unless freshness needed
    if requires_comparison and not is_time_sensitive:
        return False

    # Only block tools when autonomy requires explicit confirmation (high-risk)
    if autonomy_action == "confirm":
        return False

    # Full Agent Autonomy: Default to True for all other queries
    # Allow the LLM to decide whether it needs tools (e.g. search_web for facts).
    return True


# ============================================
# TOOL CALL PARSING
# ============================================

def parse_tool_calls(text: str) -> List[ToolCall]:
    """
    Parse one or multiple TOOL_CALLs from LLM output sequentially.
    Expected format: TOOL_CALL: tool_name("args") or TOOL_CALL: tool_name(args)
    Supports multiple parallel generations concatenated or separated by spaces/newlines.
    """
    calls = []

    # Regex setup (allows matching sequentially within the same string segment)
    # Extracts the full chunk raw match, plus the tool name and argument block.
    pattern_with_quotes = r'TOOL_CALL:\s*(\w+)\s*\(\s*"([^"]*)"\s*\)'
    pattern_without_quotes = r'TOOL_CALL:\s*(\w+)\s*\(([^)]*)\)'

    # Process all quote variants first across entire string...
    # Warning: finditer returns all sequences so we don't accidentally swallow items!
    
    for match in re.finditer(pattern_with_quotes, text):
        calls.append(ToolCall(
            name=match.group(1),
            args=match.group(2),
            raw=match.group(0)
        ))
        # Strip processed segments to avoid overlap processing with the second regex pass
        text = text.replace(match.group(0), "")

    # Process unquoted tool matches sequentially locally (like boolean expressions or calculations without string wrapper).
    for match in re.finditer(pattern_without_quotes, text):
        calls.append(ToolCall(
            name=match.group(1),
            args=match.group(2).strip().strip('\"\''),
            raw=match.group(0)
        ))

    return calls


# ============================================
# CONTRACT-BASED VALIDATION
# ============================================

def validate_tool_result(result: Dict) -> bool:
    """
    Contract-based validation. Deterministic.
    A tool result is valid IFF:
      - result is not None
      - status == "success"
      - data is truthy (non-empty, non-None)
    """
    return (
        result is not None
        and result.get("status") == "success"
        and bool(result.get("data"))
    )


# ============================================
# TIMEOUT GUARD
# ============================================

async def safe_execute(tool_func, *args, timeout: int = TOOL_TIMEOUT, is_async: bool = True, **kwargs):
    """
    Execute a tool with timeout protection.
    Prevents agent freeze from slow APIs.
    Returns standard contract on timeout.
    """
    try:
        if is_async:
            return await asyncio.wait_for(tool_func(*args, **kwargs), timeout=timeout)
        else:
            # Sync tools run directly (they're instant, like get_current_time)
            return tool_func(*args, **kwargs)
    except asyncio.TimeoutError:
        return {
            "status": "failure",
            "data": "Execution timed out.",
            "source": "timeout",
            "confidence": "low",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "error",
            "confidence": "low",
        }


# ============================================
# TOOL EXECUTION (with retry + timeout)
# ============================================

import inspect

async def execute_tool(tool_call: ToolCall, exec_ctx: Optional[ExecutionContext] = None) -> ToolResult:
    """
    Execute a single tool with retry logic and timeout guard.
    Returns ToolResult with success/failure and data.
    Tracks retries in exec_ctx if provided.
    """
    result = ToolResult(tool_name=tool_call.name)

    if tool_call.name not in TOOLS:
        result.error = f"Unknown tool: {tool_call.name}"
        return result

    # --- PHASE 4D: Transaction State Capture ---
    if exec_ctx and hasattr(exec_ctx, "session_id") and exec_ctx.session_id and hasattr(exec_ctx, "task_id") and exec_ctx.task_id:
        from app.state.transaction_manager import get_transaction, add_rollback_action, CompensatingAction, TOOL_CLASSIFICATOR, ToolClass
        ctx = get_transaction(exec_ctx.session_id, exec_ctx.task_id)
        if ctx and ctx.is_active:
            t_class = TOOL_CLASSIFICATOR.get(tool_call.name)
            if t_class == ToolClass.REVERSIBLE:
                action = CompensatingAction(tool_call.name, "GENERIC_RESTORE", {"args": tool_call.args})
                add_rollback_action(exec_ctx.session_id, exec_ctx.task_id, action)
            elif t_class == ToolClass.COMPENSATABLE:
                action = CompensatingAction(tool_call.name, "GENERIC_COMPENSATE", {"args": tool_call.args})
                add_rollback_action(exec_ctx.session_id, exec_ctx.task_id, action)

    tool_meta = TOOLS[tool_call.name]
    handler = tool_meta["func"]
    is_async = tool_meta["is_async"]

    # Lazy-load web_fetch handler
    if tool_call.name == "web_fetch" and handler is None:
        from app.input_processing.web_fetch import _tool_web_fetch
        handler = _tool_web_fetch
        TOOLS["web_fetch"]["func"] = handler

    # --- PHASE 6: Sandbox Isolation Routing ---
    from app.config import SANDBOX_ENABLED
    # Tools that are safe to sandbox (have no in-process side effects needed by orchestrator)
    _SANDBOX_ELIGIBLE = {"read_file", "calculate", "retrieve_knowledge"}
    
    if SANDBOX_ENABLED and tool_call.name in _SANDBOX_ELIGIBLE:
        from app.sandbox.sandbox_manager import get_sandbox_manager
        sandbox = get_sandbox_manager()
        sandbox_result = await sandbox.execute_tool_sandboxed(
            tool_name=tool_call.name,
            tool_args=tool_call.args,
            session_id=tool_call.session_id,
        )
        result.success = sandbox_result.success
        result.data = sandbox_result.data
        result.source = sandbox_result.source
        result.confidence = sandbox_result.confidence
        result.trust = sandbox_result.trust
        result.error = sandbox_result.error
        if sandbox_result.timed_out:
            if exec_ctx:
                exec_ctx.degraded = True
                exec_ctx.degradation_reasons.append(f"sandbox_timeout:{tool_call.name}")
        if sandbox_result.crashed:
            if exec_ctx:
                exec_ctx.degraded = True
                exec_ctx.degradation_reasons.append(f"sandbox_crash:{tool_call.name}")
        return result
    # --- END PHASE 6 ---
    
    sig = inspect.signature(handler)
    kwargs = {}
    if "session_id" in sig.parameters:
        kwargs["session_id"] = tool_call.session_id
    if "user_id" in sig.parameters:
        kwargs["user_id"] = tool_call.user_id

    for attempt in range(MAX_RETRIES):
        raw = await safe_execute(handler, tool_call.args, timeout=TOOL_TIMEOUT, is_async=is_async, **kwargs)

        if validate_tool_result(raw):
            result.success = True
            result.data = truncate_payload(raw["data"])
            result.source = raw.get("source", tool_call.name)
            result.confidence = raw.get("confidence", "medium")
            result.trust = raw.get("trust", "verified")
            
            # Confidence downgrade if retries used
            if attempt > 0 and result.confidence != "low":
                result.confidence = "medium" if result.confidence == "high" else "low"
                
            result.retries = attempt
            return result
        else:
            result.retries = attempt + 1
            result.error = raw.get("source", "") if raw else "no response"
            # Track retry in execution context
            if exec_ctx:
                exec_ctx.retry_count += 1

    # All retries exhausted
    result.success = False
    result.source = tool_call.name
    result.confidence = "low"
    if not result.error:
        result.error = "Tool returned empty or invalid data after retries"
    return result


# ============================================
# STRUCTURED TOOL RESULT INJECTION
# ============================================

def format_tool_result(result: ToolResult) -> str:
    """
    Format a tool result for injection into the LLM context.
    Uses structured tags so the LLM can assess reliability.
    """
    if result.success:
        data_str = json.dumps(result.data, indent=2, default=str)
        return (
            f"TOOL_RESULT: {result.tool_name}\n"
            f"STATUS: success\n"
            f"SOURCE: {result.source}\n"
            f"CONFIDENCE: {result.confidence}\n"
            f"DATA:\n{data_str}"
        )
    else:
        return (
            f"TOOL_RESULT: {result.tool_name}\n"
            f"STATUS: failure\n"
            f"SOURCE: {result.source}\n"
            f"CONFIDENCE: {result.confidence}\n"
            f"REASON: {result.error}"
        )
