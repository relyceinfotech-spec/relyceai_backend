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
  - get_current_time()  â†’ real datetime
  - search_web(query)   â†’ serper API search (organic results)
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
from urllib.parse import urlparse
from app.safety.content_policy import classify_nsfw, has_prompt_injection_markers


# ============================================
# CONFIGURATION
# ============================================

MAX_TOOL_CALLS = 2        # max tool invocations per request
MAX_RETRIES = 2           # retry failed tools up to N times
TOOL_TIMEOUT = 3          # seconds before tool execution is aborted
MAX_TOOL_PAYLOAD = 4000   # max characters returned by a tool
TOOL_COOLDOWN_SECONDS = 2.0
_RATE_LIMITED_TOOLS = {
    "search_web", "search_news", "search_images", "search_videos", "search_places", "search_maps",
    "search_reviews", "search_shopping", "search_scholar", "search_patents", "search_weather",
    "search_finance", "search_currency", "search_company", "search_legal", "search_jobs",
    "search_academic", "search_tech_docs", "compare_products", "search_products", "search_competitors",
    "search_trends", "summarize_url", "extract_tables", "faq_builder", "web_fetch"
}
_TOOL_LAST_CALL = {}

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


SERPER_LIST_KEYS = ["organic", "news", "images", "videos", "places", "maps", "reviews", "shopping", "scholar", "patents"]

def _extract_serper_items(raw_result: Any) -> List[Dict[str, Any]]:
    """Normalize Serper responses into a compact list of items."""
    def _find_list(obj: Any) -> List[Dict[str, Any]]:
        if isinstance(obj, dict):
            for key in SERPER_LIST_KEYS:
                val = obj.get(key)
                if isinstance(val, list) and val:
                    return val
        return []

    results: List[Dict[str, Any]] = []
    payloads = raw_result if isinstance(raw_result, list) else [raw_result]
    for payload in payloads:
        for item in _find_list(payload):
            if not isinstance(item, dict):
                continue
            out: Dict[str, Any] = {}
            for k in ("title", "snippet", "link", "source", "date", "imageUrl", "thumbnail", "price", "rating", "address"):
                v = item.get(k)
                if v:
                    out[k] = v
            if "link" not in out and item.get("url"):
                out["link"] = item.get("url")
            if out:
                results.append(out)
            if len(results) >= 5:
                return results
    return results


async def _tool_serper_generic(args: str = "", session_id: str = "", tool_key: str = "Search") -> Dict:
    """
    Executes a Serper tool endpoint and returns a compact list of results.
    """
    if not can_call_search(session_id):
        return {
            "status": "failure",
            "data": "Search cooldown active. Please wait a moment before searching again.",
            "source": "search_cooldown",
            "confidence": "low",
        }

    try:
        from app.llm.router import execute_serper_batch
        from app.config import SERPER_TOOLS

        endpoint = SERPER_TOOLS.get(tool_key) or SERPER_TOOLS.get(tool_key.title())
        if not endpoint:
            return {
                "status": "failure",
                "data": None,
                "source": "serper",
                "confidence": "low",
            }

        param_key = "url" if tool_key.lower() == "webpage" else "q"
        raw_result = await execute_serper_batch(endpoint, [args], param_key=param_key)
        cleaned = _extract_serper_items(raw_result)

        if cleaned:
            return {
                "status": "success",
                "data": cleaned,
                "source": f"serper_{tool_key.lower()}",
                "confidence": "high",
            }
        return {
            "status": "failure",
            "data": None,
            "source": "serper_no_results",
            "confidence": "low",
        }
    except Exception:
        return {
            "status": "failure",
            "data": None,
            "source": "serper",
            "confidence": "low",
        }


async def _tool_search_news(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="News")

async def _tool_search_images(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Images")

async def _tool_search_videos(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Videos")

async def _tool_search_places(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Places")

async def _tool_search_maps(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Maps")

async def _tool_search_reviews(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Reviews")

async def _tool_search_shopping(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Shopping")

async def _tool_search_scholar(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Scholar")


def _decorate_query(args: str, required_tokens: List[str], prefix: str) -> str:
    q = (args or "").strip()
    if not q:
        return ""
    ql = q.lower()
    if any(tok in ql for tok in required_tokens):
        return q
    return f"{prefix} {q}"

async def _tool_search_weather(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["weather", "forecast", "temperature", "rain", "humidity"], "weather")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_finance(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["stock", "price", "share", "market", "ticker", "nasdaq", "nyse"], "stock price")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_currency(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["exchange", "fx", "currency", "rate", "rates", "usd", "eur", "inr"], "exchange rate")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_company(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["company", "profile", "about", "headquarters", "revenue", "funding", "ceo", "employees", "overview"], "company profile")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_legal(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["legal", "policy", "compliance", "regulation", "law", "terms", "privacy"], "legal policy")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_jobs(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["job", "jobs", "career", "hiring", "role", "opening", "vacancy"], "jobs hiring")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_academic(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["paper", "study", "journal", "doi", "preprint", "arxiv"], "research paper")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Scholar")

async def _tool_search_tech_docs(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["docs", "documentation", "api", "reference", "sdk", "guide", "manual"], "official documentation")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_compare_products(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["compare", "comparison", "vs", "versus", "alternatives", "review"], "compare")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")


async def _tool_summarize_url(args: str = "", session_id: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        url = str(payload.get("url", "")) if isinstance(payload, dict) else str(args or "")
        if not url.strip():
            return {
                "status": "failure",
                "data": "Missing url",
                "source": "summarize_url",
                "confidence": "low",
            }
        from app.input_processing.web_fetch import fetch_and_extract
        fetched = await fetch_and_extract(url)
        if not fetched or fetched.get("status") != "success":
            return fetched or {
                "status": "failure",
                "data": "Fetch failed",
                "source": "summarize_url",
                "confidence": "low",
            }
        data = fetched.get("data", {})
        content = str(data.get("content", ""))
        sentences = re.split(r"(?<=[.!?])\s+", content)
        summary = " ".join([s.strip() for s in sentences if s.strip()][:3]).strip()
        if len(summary) > 800:
            summary = summary[:800].rsplit(" ", 1)[0] + "..."
        key_points = [line for line in content.splitlines() if line.strip()][:6]
        return {
            "status": "success",
            "data": {
                "url": data.get("url", url),
                "title": data.get("title", ""),
                "summary": summary,
                "key_points": key_points,
            },
            "source": "summarize_url",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "summarize_url",
            "confidence": "low",
        }

async def _tool_extract_tables(args: str = "", session_id: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        text = ""
        url = ""
        if isinstance(payload, dict):
            text = str(payload.get("text", ""))
            url = str(payload.get("url", ""))
        else:
            text = str(args or "")
        if not text.strip() and url:
            from app.input_processing.web_fetch import fetch_and_extract
            fetched = await fetch_and_extract(url)
            if not fetched or fetched.get("status") != "success":
                return fetched or {
                    "status": "failure",
                    "data": "Fetch failed",
                    "source": "extract_tables",
                    "confidence": "low",
                }
            text = str((fetched.get("data") or {}).get("content", ""))

        if not text.strip():
            return {
                "status": "failure",
                "data": "No text provided",
                "source": "extract_tables",
                "confidence": "low",
            }

        tables = []
        lines = [ln for ln in text.splitlines()]
        i = 0
        while i < len(lines) - 1:
            if "|" in lines[i] and "|" in lines[i + 1] and re.search(r"\|\s*-+", lines[i + 1]):
                headers = [h.strip() for h in lines[i].split("|") if h.strip()]
                i += 2
                rows = []
                while i < len(lines) and "|" in lines[i]:
                    row = [c.strip() for c in lines[i].split("|") if c.strip()]
                    if row:
                        rows.append(row)
                    i += 1
                if headers and rows:
                    tables.append({"headers": headers, "rows": rows})
                continue
            i += 1

        csv_preview = ""
        if tables:
            headers = tables[0]["headers"]
            rows = tables[0]["rows"]
            csv_lines = [",".join(headers)]
            for row in rows:
                csv_lines.append(",".join(row))
            csv_preview = "\n".join(csv_lines)

        return {
            "status": "success",
            "data": {"tables": tables, "csv_preview": csv_preview},
            "source": "extract_tables",
            "confidence": "medium" if tables else "low",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "extract_tables",
            "confidence": "low",
        }

async def _tool_search_products(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["product", "products", "buy", "price", "pricing", "review", "compare"], "product")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_competitors(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["competitor", "competition", "alternatives", "rivals", "similar"], "competitors")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")

async def _tool_search_trends(args: str = "", session_id: str = "") -> Dict:
    query = _decorate_query(args, ["trend", "trends", "market", "growth", "forecast", "industry"], "market trends")
    return await _tool_serper_generic(query, session_id=session_id, tool_key="Search")


def _tool_sentiment_scan(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        text = str(payload.get("text", "")) if isinstance(payload, dict) else str(args or "")
        if not text.strip():
            return {
                "status": "failure",
                "data": "Empty text payload",
                "source": "sentiment_scan",
                "confidence": "low",
            }
        pos_words = {"good", "great", "excellent", "positive", "love", "amazing", "awesome", "happy", "success", "win", "improve"}
        neg_words = {"bad", "poor", "terrible", "negative", "hate", "awful", "sad", "fail", "loss", "bug", "issue"}
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        pos = sum(1 for t in tokens if t in pos_words)
        neg = sum(1 for t in tokens if t in neg_words)
        score = (pos - neg) / max(1, (pos + neg))
        label = "neutral"
        if score > 0.2:
            label = "positive"
        elif score < -0.2:
            label = "negative"
        return {
            "status": "success",
            "data": {"score": score, "label": label, "positive": pos, "negative": neg},
            "source": "sentiment_scan",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "sentiment_scan",
            "confidence": "low",
        }

async def _tool_faq_builder(args: str = "", session_id: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        text = ""
        url = ""
        if isinstance(payload, dict):
            text = str(payload.get("text", ""))
            url = str(payload.get("url", ""))
        else:
            text = str(args or "")
        if not text.strip() and url:
            from app.input_processing.web_fetch import fetch_and_extract
            fetched = await fetch_and_extract(url)
            if not fetched or fetched.get("status") != "success":
                return fetched or {
                    "status": "failure",
                    "data": "Fetch failed",
                    "source": "faq_builder",
                    "confidence": "low",
                }
            text = str((fetched.get("data") or {}).get("content", ""))

        if not text.strip():
            return {
                "status": "failure",
                "data": "No text provided",
                "source": "faq_builder",
                "confidence": "low",
            }

        questions = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip().endswith("?")]
        faqs = []
        if questions:
            for q in questions[:6]:
                faqs.append({"q": q, "a": "Answer based on the document content."})
        else:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
            stop = {"this", "that", "with", "from", "your", "have", "will", "about", "there", "their", "which", "what"}
            freq = {}
            for w in words:
                if w in stop:
                    continue
                freq[w] = freq.get(w, 0) + 1
            keywords = sorted(freq, key=freq.get, reverse=True)[:5]
            for kw in keywords:
                faqs.append({"q": f"What is {kw}?", "a": f"The document discusses {kw}."})

        return {
            "status": "success",
            "data": {"faqs": faqs},
            "source": "faq_builder",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "faq_builder",
            "confidence": "low",
        }


def _tool_document_compare(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        a = str(payload.get("doc_a", "")) if isinstance(payload, dict) else ""
        b = str(payload.get("doc_b", "")) if isinstance(payload, dict) else ""
        if not a or not b:
            return {
                "status": "failure",
                "data": "Provide doc_a and doc_b",
                "source": "document_compare",
                "confidence": "low",
            }
        a_lines = a.splitlines()
        b_lines = b.splitlines()
        import difflib
        diff = list(difflib.unified_diff(a_lines, b_lines, lineterm="", n=2))
        diff_preview = "\n".join(diff[:200])
        sm = difflib.SequenceMatcher(None, a_lines, b_lines)
        added = removed = changed = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "insert":
                added += (j2 - j1)
            elif tag == "delete":
                removed += (i2 - i1)
            elif tag == "replace":
                changed += max(i2 - i1, j2 - j1)
        return {
            "status": "success",
            "data": {
                "summary": {"added": added, "removed": removed, "changed": changed},
                "diff": diff_preview,
            },
            "source": "document_compare",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "document_compare",
            "confidence": "low",
        }


def _tool_data_cleaner(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        raw = str(payload.get("data", "")) if isinstance(payload, dict) else str(args or "")
        if not raw.strip():
            return {
                "status": "failure",
                "data": "No data provided",
                "source": "data_cleaner",
                "confidence": "low",
            }
        cleaned = []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        new_row = {}
                        for k, v in row.items():
                            nk = re.sub(r"\s+", "_", str(k).strip().lower())
                            nv = v.strip() if isinstance(v, str) else v
                            new_row[nk] = nv
                        cleaned.append(new_row)
            elif isinstance(data, dict):
                cleaned.append(data)
        except Exception:
            import csv
            from io import StringIO
            reader = csv.DictReader(StringIO(raw))
            for row in reader:
                new_row = {}
                for k, v in row.items():
                    nk = re.sub(r"\s+", "_", str(k).strip().lower())
                    nv = v.strip() if isinstance(v, str) else v
                    new_row[nk] = nv
                cleaned.append(new_row)

        deduped = []
        seen = set()
        for row in cleaned:
            key = json.dumps(row, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        return {
            "status": "success",
            "data": {"rows": deduped, "count": len(deduped)},
            "source": "data_cleaner",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "data_cleaner",
            "confidence": "low",
        }


def _tool_unit_cost_calc(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        items = payload.get("items", []) if isinstance(payload, dict) else []
        currency = payload.get("currency", "") if isinstance(payload, dict) else ""
        if not items:
            return {
                "status": "failure",
                "data": "Provide items with unit_cost and quantity",
                "source": "unit_cost_calc",
                "confidence": "low",
            }
        breakdown = []
        total = 0.0
        for item in items:
            name = str(item.get("name", "item"))
            unit = float(item.get("unit_cost", 0))
            qty = float(item.get("quantity", 0))
            item_total = unit * qty
            total += item_total
            breakdown.append({"name": name, "unit_cost": unit, "quantity": qty, "total": item_total})
        return {
            "status": "success",
            "data": {"currency": currency, "total": total, "items": breakdown},
            "source": "unit_cost_calc",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "unit_cost_calc",
            "confidence": "low",
        }

def _tool_pdf_maker(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        title = str(payload.get("title", "Document Export")) if isinstance(payload, dict) else "Document Export"
        target = str(payload.get("target", "text")) if isinstance(payload, dict) else "text"
        content = str(payload.get("content", "")).strip() if isinstance(payload, dict) else str(args or "").strip()

        if not content and target != "chat":
            return {
                "status": "failure",
                "data": "Provide content to convert into PDF.",
                "source": "pdf_maker",
                "confidence": "low",
            }

        return {
            "status": "success",
            "data": {
                "title": title,
                "target": target,
                "content": content,
                "message": "PDF payload ready for client download",
            },
            "source": "pdf_maker",
            "confidence": "high",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "pdf_maker",
            "confidence": "low",
        }

def _tool_extract_entities(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        text = str(payload.get("text", "")) if isinstance(payload, dict) else str(args or "")
        if not text.strip():
            return {
                "status": "failure",
                "data": "Empty text payload",
                "source": "extract_entities",
                "confidence": "low",
            }

        emails = sorted(set(re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.I)))
        urls = sorted(set(re.findall(r"https?://[^\s)\]]+", text, flags=re.I)))
        phones = sorted(set(re.findall(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{4}", text)))
        money = sorted(set(re.findall(r"\$\s?\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s?(?:USD|EUR|INR|GBP|AUD|CAD)\b", text, flags=re.I)))

        candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
        common = {"The", "This", "That", "These", "Those", "For", "And", "With", "From", "Into", "About"}
        people_orgs = sorted(set(c for c in candidates if c.split()[0] not in common))

        data = {
            "emails": emails,
            "urls": urls,
            "phones": phones,
            "money": money,
            "people_orgs": people_orgs,
            "counts": {
                "emails": len(emails),
                "urls": len(urls),
                "phones": len(phones),
                "money": len(money),
                "people_orgs": len(people_orgs),
            },
        }
        return {
            "status": "success",
            "data": data,
            "source": "extract_entities",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "extract_entities",
            "confidence": "low",
        }


def _tool_validate_code(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        code = str(payload.get("code", "")) if isinstance(payload, dict) else str(args or "")
        language = str(payload.get("language", "")).lower().strip() if isinstance(payload, dict) else ""
        if not code.strip():
            return {
                "status": "failure",
                "data": "Empty code payload",
                "source": "validate_code",
                "confidence": "low",
            }

        findings = []
        lines = code.splitlines()
        secret_patterns = [
            (r"api[_-]?key\s*[:=]", "Potential API key in code"),
            (r"secret\s*[:=]", "Potential secret in code"),
            (r"password\s*[:=]", "Potential password in code"),
            (r"-----BEGIN [A-Z ]+PRIVATE KEY-----", "Private key material in code"),
        ]
        risky_patterns = [
            (r"\beval\s*\(", "Use of eval"),
            (r"\bexec\s*\(", "Use of exec"),
            (r"\bsubprocess\.", "Subprocess usage"),
            (r"\bos\.system\s*\(", "Use of os.system"),
        ]
        sql_concat = [r"SELECT\s+.+\+.+", r"INSERT\s+.+\+.+", r"UPDATE\s+.+\+.+", r"DELETE\s+.+\+.+", r"WHERE\s+.+\+.+"] if language in ("python", "javascript", "typescript", "js", "ts") else []

        for idx, line in enumerate(lines, start=1):
            for pattern, message in secret_patterns:
                if re.search(pattern, line, flags=re.I):
                    findings.append({"line": idx, "severity": "high", "rule": "hardcoded_secret", "message": message})
            for pattern, message in risky_patterns:
                if re.search(pattern, line, flags=re.I):
                    findings.append({"line": idx, "severity": "medium", "rule": "risky_exec", "message": message})
            if re.search(r"TODO|FIXME", line, flags=re.I):
                findings.append({"line": idx, "severity": "low", "rule": "todo", "message": "TODO/FIXME marker found"})
            for pattern in sql_concat:
                if re.search(pattern, line, flags=re.I):
                    findings.append({"line": idx, "severity": "medium", "rule": "sql_concat", "message": "Possible SQL string concatenation"})

        summary = {
            "total": len(findings),
            "high": sum(1 for f in findings if f["severity"] == "high"),
            "medium": sum(1 for f in findings if f["severity"] == "medium"),
            "low": sum(1 for f in findings if f["severity"] == "low"),
        }
        return {
            "status": "success",
            "data": {"summary": summary, "findings": findings},
            "source": "validate_code",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "validate_code",
            "confidence": "low",
        }


def _tool_generate_tests(args: str = "") -> Dict:
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        code = str(payload.get("code", "")) if isinstance(payload, dict) else str(args or "")
        filename = str(payload.get("filename", "")).strip() if isinstance(payload, dict) else ""
        language = str(payload.get("language", "")).lower().strip() if isinstance(payload, dict) else ""

        if not language and filename:
            if filename.endswith((".py",)):
                language = "python"
            elif filename.endswith((".js", ".jsx")):
                language = "javascript"
            elif filename.endswith((".ts", ".tsx")):
                language = "typescript"

        if not language:
            language = "generic"

        if language in ("python", "py"):
            test_path = "tests/test_module.py"
            template = (
                "import pytest\n\n"
                "def test_basic_behavior():\n"
                "    # TODO: arrange\n"
                "    # TODO: act\n"
                "    # TODO: assert\n"
                "    assert True\n"
            )
            framework = "pytest"
        elif language in ("javascript", "typescript", "js", "ts"):
            test_path = "tests/module.test.js"
            template = (
                "describe('module', () => {\n"
                "  it('works as expected', () => {\n"
                "    // TODO: arrange\n"
                "    // TODO: act\n"
                "    // TODO: assert\n"
                "    expect(true).toBe(true);\n"
                "  });\n"
                "});\n"
            )
            framework = "jest"
        else:
            test_path = "tests/test_module.txt"
            template = "Add tests for the provided code."
            framework = "generic"

        data = {
            "framework": framework,
            "files": [{"path": test_path, "content": template}],
            "notes": "Template generated from code context; refine assertions and setup.",
        }
        return {
            "status": "success",
            "data": data,
            "source": "generate_tests",
            "confidence": "medium",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": str(e),
            "source": "generate_tests",
            "confidence": "low",
        }

def _tool_execute_code(args: str = "") -> Dict:
    """Executes small Python snippets in a restricted sandbox context."""
    try:
        payload = None
        if args and args.strip().startswith("{"):
            payload = json.loads(args)
        if payload and isinstance(payload, dict):
            language = str(payload.get("language", "python")).lower().strip()
            code = str(payload.get("code", ""))
            expression = str(payload.get("expression", "")).strip()
        else:
            language = "python"
            code = str(args or "")
            expression = ""

        if language not in ("python", "py"):
            return {
                "status": "failure",
                "data": {"stdout": "", "stderr": f"Unsupported language: {language}", "result": None, "locals": {}},
                "source": "execute_code",
                "confidence": "low",
            }
        if not code.strip() and not expression:
            return {
                "status": "failure",
                "data": {"stdout": "", "stderr": "Empty code payload", "result": None, "locals": {}},
                "source": "execute_code",
                "confidence": "low",
            }

        import io
        import contextlib
        import math
        import statistics

        safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
        }

        sandbox_globals = {"__builtins__": safe_builtins, "math": math, "statistics": statistics}
        sandbox_locals: Dict[str, Any] = {}

        stdout = io.StringIO()
        stderr = ""
        result_value = None

        try:
            with contextlib.redirect_stdout(stdout):
                if expression:
                    result_value = eval(expression, sandbox_globals, sandbox_locals)
                if code.strip():
                    exec(compile(code, "<sandbox>", "exec"), sandbox_globals, sandbox_locals)
        except Exception as e:
            stderr = str(e)

        output = stdout.getvalue().strip()
        locals_preview = {}
        for k, v in list(sandbox_locals.items())[:12]:
            if k.startswith("_"):
                continue
            try:
                locals_preview[k] = repr(v)
            except Exception:
                locals_preview[k] = "<unrepr>"

        if result_value is None and "result" in sandbox_locals:
            try:
                result_value = sandbox_locals.get("result")
            except Exception:
                result_value = None

        status = "success" if not stderr else "failure"
        confidence = "high" if status == "success" else "low"

        return {
            "status": status,
            "data": {
                "stdout": output,
                "stderr": stderr,
                "result": result_value,
                "locals": locals_preview,
            },
            "source": "execute_code",
            "confidence": confidence,
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": {"stdout": "", "stderr": str(e), "result": None, "locals": {}},
            "source": "execute_code",
            "confidence": "low",
        }

async def _tool_search_patents(args: str = "", session_id: str = "") -> Dict:
    return await _tool_serper_generic(args, session_id=session_id, tool_key="Patents")

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
    "search_news": {
        "func": _tool_search_news,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_images": {
        "func": _tool_search_images,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_videos": {
        "func": _tool_search_videos,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_places": {
        "func": _tool_search_places,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_maps": {
        "func": _tool_search_maps,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_reviews": {
        "func": _tool_search_reviews,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_shopping": {
        "func": _tool_search_shopping,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_scholar": {
        "func": _tool_search_scholar,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_patents": {
        "func": _tool_search_patents,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_weather": {
        "func": _tool_search_weather,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_finance": {
        "func": _tool_search_finance,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_currency": {
        "func": _tool_search_currency,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_company": {
        "func": _tool_search_company,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_legal": {
        "func": _tool_search_legal,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_jobs": {
        "func": _tool_search_jobs,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_academic": {
        "func": _tool_search_academic,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_tech_docs": {
        "func": _tool_search_tech_docs,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "compare_products": {
        "func": _tool_compare_products,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "summarize_url": {
        "func": _tool_summarize_url,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "extract_tables": {
        "func": _tool_extract_tables,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_products": {
        "func": _tool_search_products,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_competitors": {
        "func": _tool_search_competitors,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "search_trends": {
        "func": _tool_search_trends,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "sentiment_scan": {
        "func": _tool_sentiment_scan,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "faq_builder": {
        "func": _tool_faq_builder,
        "is_async": True,
        "reversible": True,
        "risk": "low",
        "freshness": "live",
    },
    "document_compare": {
        "func": _tool_document_compare,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "data_cleaner": {
        "func": _tool_data_cleaner,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "unit_cost_calc": {
        "func": _tool_unit_cost_calc,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "pdf_maker": {
        "func": _tool_pdf_maker,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "extract_entities": {
        "func": _tool_extract_entities,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "validate_code": {
        "func": _tool_validate_code,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "generate_tests": {
        "func": _tool_generate_tests,
        "is_async": False,
        "reversible": True,
        "risk": "low",
        "freshness": "static",
    },
    "execute_code": {
        "func": _tool_execute_code,
        "is_async": False,
        "reversible": True,
        "risk": "medium",
        "freshness": "static",
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
    The APP decides if tools are allowed â€” not the LLM.

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
# TOOL ARGUMENT VALIDATION
# ============================================

_SEARCH_TOOLS = {
    "search_web", "search_news", "search_images", "search_videos", "search_places", "search_maps",
    "search_reviews", "search_shopping", "search_scholar", "search_patents", "search_weather",
    "search_finance", "search_currency", "search_company", "search_legal", "search_jobs",
    "search_academic", "search_tech_docs", "compare_products", "search_products", "search_competitors",
    "search_trends", "faq_builder", "extract_tables", "summarize_url", "web_fetch"
}


def _safe_json_extract_arg(raw_args: str, keys: List[str]) -> str:
    try:
        if raw_args and raw_args.strip().startswith("{"):
            payload = json.loads(raw_args)
            if isinstance(payload, dict):
                for k in keys:
                    if payload.get(k):
                        return str(payload.get(k))
    except Exception:
        return raw_args or ""
    return raw_args or ""


def _validate_tool_args(tool_call: ToolCall) -> Optional[str]:
    args = tool_call.args or ""

    # Hard length cap to avoid abuse payloads.
    if len(args) > 8000:
        return "Tool arguments too large for safe execution."

    # Prompt-injection markers in tool args are suspicious for fetched/RAG content flows.
    if has_prompt_injection_markers(args):
        return "Tool arguments blocked: prompt-injection markers detected."

    if tool_call.name in _SEARCH_TOOLS:
        query = _safe_json_extract_arg(args, ["q", "query", "url", "text"])
        blocked, reason = classify_nsfw(query)
        if blocked:
            return reason

    # URL safety for URL-based tools.
    if tool_call.name in {"summarize_url", "web_fetch"}:
        url = _safe_json_extract_arg(args, ["url"])
        if url:
            parsed = urlparse(url.strip())
            if parsed.scheme and parsed.scheme not in {"http", "https"}:
                return "Only http/https URLs are allowed."

    return None
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

    arg_error = _validate_tool_args(tool_call)
    if arg_error:
        result.error = arg_error
        result.source = tool_call.name
        result.confidence = "low"
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

    # Rate limit heavy tools (per session)
    if tool_call.name in _RATE_LIMITED_TOOLS:
        key = ((tool_call.session_id or "anon"), tool_call.name)
        now = time.time()
        last = _TOOL_LAST_CALL.get(key)
        if last and (now - last) < TOOL_COOLDOWN_SECONDS:
            result.error = f"Rate limited: wait {TOOL_COOLDOWN_SECONDS:.1f}s before calling {tool_call.name} again."
            result.source = tool_call.name
            result.confidence = "low"
            return result
        _TOOL_LAST_CALL[key] = now
    # Lazy-load web_fetch handler
    if tool_call.name == "web_fetch" and handler is None:
        from app.input_processing.web_fetch import _tool_web_fetch
        handler = _tool_web_fetch
        TOOLS["web_fetch"]["func"] = handler

    # --- PHASE 6: Sandbox Isolation Routing ---
    from app.config import SANDBOX_ENABLED
    # Tools that are safe to sandbox (have no in-process side effects needed by orchestrator)
    _SANDBOX_ELIGIBLE = {"read_file", "calculate", "retrieve_knowledge", "execute_code", "extract_entities", "validate_code", "generate_tests", "sentiment_scan", "document_compare", "data_cleaner", "unit_cost_calc", "pdf_maker"}
    
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














