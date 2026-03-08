from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.llm.token_counter import estimate_tokens

ROUTER_DEBUG = False
ROUTER_STAGE_PRIORITY = {
    "rules": 1,
    "embedding": 2,
    "llm": 3,
    "fallback": 4,
}

CONFIDENCE_THRESHOLD = 0.62
INTENT_CONFIDENCE_FALLBACK = 0.55
INTENT_TIMEOUT_MS = 120
TRACE_MAX_BYTES = 10 * 1024 * 1024

SESSION_COOLDOWN_PENALTY = 0.75
FAIL_RATE_PENALTY = 0.6
FAIL_RATE_THRESHOLD = 0.5
MIN_FAILURE_ATTEMPTS = 4
DOMAIN_PRIORITY_BOOST = 1.15
RESEARCH_BOOST = 1.2


@dataclass
class SkillMeta:
    name: str
    purpose: str
    keywords: List[str]
    type: str = "runtime"  # runtime | advanced_runtime | meta
    domain: str = "general"


@dataclass
class SkillSelection:
    selected_skills: List[str] = field(default_factory=list)
    capsules: List[Tuple[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    router_stage: str = "fallback"
    stage_priority: int = ROUTER_STAGE_PRIORITY["fallback"]
    fallback_level: int = 0
    execution_path: str = "normal"
    intent: str = "general"
    intent_confidence: float = 0.0
    domain: str = "general"


_SKILLS: List[SkillMeta] = [
    # Design/UI
    SkillMeta("ui-ux-pro-max", "Premium UI/UX patterns and design systems.", ["ui", "ux", "design", "theme", "palette", "dashboard", "layout"], "runtime", "ui"),
    SkillMeta("frontend-design", "Professional frontend architecture and component structure.", ["frontend", "react", "component", "tailwind", "css", "interface", "responsive"], "runtime", "ui"),
    SkillMeta("web-artifacts-builder", "Complex web artifacts and reusable UI modules.", ["artifact", "widget", "shadcn", "module", "component", "ui"], "runtime", "ui"),
    # Memory/Context
    SkillMeta("memory-systems", "Persist and retrieve user/session memory.", ["remember", "memory", "preference", "profile", "recall"], "runtime", "memory"),
    SkillMeta("digital-brain", "Personal operating system for identity, goals, and voice.", ["identity", "goals", "persona", "profile", "voice", "preferences"], "runtime", "memory"),
    SkillMeta("context-optimization", "Optimize context windows and reduce token usage.", ["token", "optimize", "context", "latency", "cost"], "runtime", "context"),
    SkillMeta("context-compression", "Compress long conversations while preserving intent.", ["compress", "summary", "condense", "history", "long chat"], "runtime", "context"),
    # Engineering
    SkillMeta("react-best-practices", "Production React patterns and performance practices.", ["react", "next", "performance", "rerender", "hook", "state"], "runtime", "engineering"),
    SkillMeta("supabase-postgres-best-practices", "Supabase/Postgres schema and query best practices.", ["supabase", "postgres", "sql", "schema", "query", "index", "rls"], "runtime", "engineering"),
    SkillMeta("webapp-testing", "Web app test strategy and Playwright automation.", ["test", "testing", "playwright", "e2e", "qa", "automation"], "runtime", "engineering"),
    # Research/Logic
    SkillMeta("research-agent", "Professional multi-step research protocols and error recovery.", ["research", "analyze", "analysis", "compare", "sources", "evidence"], "runtime", "research"),
    SkillMeta("multi-agent-patterns", "Multi-agent planning and coordination patterns.", ["agent", "orchestrate", "workflow", "planner", "multi-step", "dag"], "runtime", "logic"),
    SkillMeta("systematic-debugging", "Structured debugging and root-cause analysis.", ["debug", "bug", "error", "trace", "fix", "root cause"], "runtime", "logic"),
    SkillMeta("using-superpowers", "Strict professional reliability protocols for complex execution.", ["reliability", "protocol", "strict", "quality", "robust"], "advanced_runtime", "logic"),
    # Meta/dev
    SkillMeta("skill-creator", "Create and improve custom skills.", ["create skill", "new skill", "build skill", "agent capability"], "meta", "meta"),
    SkillMeta("writing-skills", "Professional standards for writing skill docs and instructions.", ["write skill", "skill documentation", "instructions", "prompt writing"], "meta", "meta"),
]

DOMAIN_PRIORITY = {
    "ui": ["frontend-design", "ui-ux-pro-max"],
    "memory": ["memory-systems", "digital-brain"],
    "research": ["research-agent"],
    "engineering": ["react-best-practices", "systematic-debugging"],
    "logic": ["multi-agent-patterns", "systematic-debugging"],
}

INTENT_TO_DOMAIN = {
    "ui_design": "ui",
    "memory": "memory",
    "personalization": "memory",
    "context": "context",
    "engineering": "engineering",
    "debugging": "engineering",
    "research": "research",
    "analysis": "research",
    "logic": "logic",
    "meta": "meta",
    "general": "general",
}

_INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
    ("ui_design", ["ui", "dashboard", "layout", "theme", "design", "component", "tailwind"]),
    ("memory", ["remember", "my preference", "my goal", "profile", "who am i"]),
    ("personalization", ["voice", "identity", "tone", "personalize", "my style"]),
    ("context", ["compress", "summarize history", "token", "context", "reduce cost"]),
    ("engineering", ["react", "nextjs", "supabase", "postgres", "schema", "performance"]),
    ("debugging", ["debug", "error", "fix", "trace", "bug", "failing"]),
    ("research", ["research", "compare", "analysis", "market", "study", "evidence"]),
    ("logic", ["workflow", "agent", "plan", "orchestrate", "multi-step", "dag"]),
    ("meta", ["create skill", "new skill", "skill docs", "writing skills"]),
]

_BASE_DIR = Path(__file__).resolve().parents[2]
_SKILLS_DIR = _BASE_DIR.parent / "skills"
_CACHE_DIR = _BASE_DIR / "cache"
_LOG_DIR = _BASE_DIR / "logs"
_VECTOR_CACHE_FILE = _CACHE_DIR / "skill_vectors.json"
_PERF_FILE = _CACHE_DIR / "skill_performance.json"
_TRACE_FILE = _LOG_DIR / "skill_router_trace.jsonl"

_skill_vectors: Dict[str, Dict[str, int]] = {}
_skill_capsules: Dict[str, str] = {}
_skill_perf: Dict[str, Dict[str, float]] = {}
_session_recent_skills: Dict[str, List[str]] = {}
_session_last_selection: Dict[str, SkillSelection] = {}


def _debug(msg: str) -> None:
    if ROUTER_DEBUG:
        print(f"[SkillRouter] {msg}")


def _ensure_dirs() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_\-]+", (text or "").lower())


def _vectorize(text: str) -> Dict[str, int]:
    vec: Dict[str, int] = {}
    for tok in _tokenize(text):
        vec[tok] = vec.get(tok, 0) + 1
    return vec


def _jaccard(a: Dict[str, int], b: Dict[str, int]) -> float:
    sa = set(a.keys())
    sb = set(b.keys())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _load_capsule(skill_name: str) -> str:
    if skill_name in _skill_capsules:
        return _skill_capsules[skill_name]
    path = _SKILLS_DIR / skill_name / "SKILL.md"
    text = ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = ""
    clean = re.sub(r"\s+", " ", text).strip()[:700]
    capsule = f"Skill: {skill_name}\nPurpose: {clean}" if clean else f"Skill: {skill_name}"
    _skill_capsules[skill_name] = capsule
    return capsule


def _load_perf() -> None:
    global _skill_perf
    if _skill_perf:
        return
    _ensure_dirs()
    if _PERF_FILE.exists():
        try:
            _skill_perf = json.loads(_PERF_FILE.read_text(encoding="utf-8"))
        except Exception:
            _skill_perf = {}


def _save_perf() -> None:
    _ensure_dirs()
    _PERF_FILE.write_text(json.dumps(_skill_perf, indent=2), encoding="utf-8")


def _rotate_trace_if_needed() -> None:
    _ensure_dirs()
    try:
        if not _TRACE_FILE.exists():
            return
        if _TRACE_FILE.stat().st_size < TRACE_MAX_BYTES:
            return
        oldest = _TRACE_FILE.with_suffix(".jsonl.2")
        mid = _TRACE_FILE.with_suffix(".jsonl.1")
        if oldest.exists():
            oldest.unlink(missing_ok=True)
        if mid.exists():
            mid.rename(oldest)
        _TRACE_FILE.rename(mid)
    except Exception:
        pass


def preload_skill_vectors() -> bool:
    global _skill_vectors
    _ensure_dirs()
    if _skill_vectors:
        return True

    cached = None
    if _VECTOR_CACHE_FILE.exists():
        try:
            cached = json.loads(_VECTOR_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cached = None
    if isinstance(cached, dict) and "vectors" in cached:
        _skill_vectors = cached.get("vectors", {})
        if _skill_vectors:
            return True

    vectors: Dict[str, Dict[str, int]] = {}
    for skill in _SKILLS:
        base = f"{skill.name} {skill.purpose} {' '.join(skill.keywords)} {skill.domain} {skill.type}"
        vectors[skill.name] = _vectorize(base)
    _skill_vectors = vectors
    _VECTOR_CACHE_FILE.write_text(json.dumps({"vectors": vectors, "ts": int(time.time())}), encoding="utf-8")
    return True


def _complexity_path(query: str) -> str:
    q = (query or "").lower()
    complex_markers = ["build", "create", "full", "end-to-end", "architecture", "analyze", "optimize", "pipeline", "workflow"]
    if len(q.split()) > 22 or sum(1 for m in complex_markers if m in q) >= 2:
        return "dag"
    return "normal"


def _mode_caps(mode: str, execution_path: str) -> int:
    m = (mode or "normal").lower()
    if execution_path == "dag":
        if m == "agent":
            return 8
        if m in {"business", "deepsearch"}:
            return 5
        return 2
    if m == "agent":
        return 2
    if m in {"business", "deepsearch"}:
        return 2
    return 1


def _detect_intent(query: str, timeout_ms: int = INTENT_TIMEOUT_MS) -> Tuple[Optional[str], float]:
    start = time.perf_counter()
    q = (query or "").lower().strip()
    if not q:
        return "general", 1.0

    best_intent = "general"
    best_score = 0.0
    tokens = set(_tokenize(q))

    for intent, keys in _INTENT_PATTERNS:
        if (time.perf_counter() - start) * 1000.0 > timeout_ms:
            return None, 0.0
        score = 0.0
        for k in keys:
            if " " in k:
                if k in q:
                    score += 1.0
            else:
                if k in tokens or k in q:
                    score += 1.0
        if score <= 0:
            continue
        norm = score / max(1.0, len(keys) * 0.35)
        if norm > best_score:
            best_intent = intent
            best_score = min(1.0, norm)

    if best_score > 0:
        return best_intent, best_score
    return "general", 0.5


def _score_rule(skill: SkillMeta, query: str) -> float:
    q = (query or "").lower()
    hits = sum(1 for k in skill.keywords if k in q)
    if hits == 0:
        return 0.0
    return min(1.0, hits / max(2, len(skill.keywords) // 2))


def _success_rate(skill_name: str) -> float:
    st = _skill_perf.get(skill_name, {})
    attempts = float(st.get("attempts", 0.0))
    successes = float(st.get("successes", 0.0))
    if attempts <= 0:
        return 0.5
    return max(0.0, min(1.0, successes / attempts))


def _latency_bonus(skill_name: str) -> float:
    st = _skill_perf.get(skill_name, {})
    avg_ms = float(st.get("avg_latency_ms", 0.0))
    if avg_ms <= 0:
        return 0.5
    if avg_ms <= 900:
        return 1.0
    if avg_ms <= 1800:
        return 0.7
    if avg_ms <= 3000:
        return 0.4
    return 0.2


def _failure_penalty(skill_name: str) -> float:
    st = _skill_perf.get(skill_name, {})
    attempts = float(st.get("attempts", 0.0))
    failures = float(st.get("failures", 0.0))
    if attempts < MIN_FAILURE_ATTEMPTS:
        return 1.0
    fail_rate = failures / max(1.0, attempts)
    return FAIL_RATE_PENALTY if fail_rate > FAIL_RATE_THRESHOLD else 1.0


def _allow_skill(skill: SkillMeta, mode: str, execution_path: str, intent: str, query: str) -> bool:
    m = (mode or "normal").lower()
    q = (query or "").lower()

    if skill.type == "meta":
        # Meta skills are developer-only: explicit asks in agent mode.
        return m == "agent" and any(x in q for x in ["skill", "agent capability", "prompt", "documentation"])

    if skill.type == "advanced_runtime":
        return m == "agent" and execution_path == "dag"

    if skill.name == "digital-brain":
        return intent in {"memory", "personalization", "general"}

    return True


def _log_trace(query: str, selection: SkillSelection) -> None:
    _ensure_dirs()
    _rotate_trace_if_needed()
    payload = {
        "ts": int(time.time()),
        "query": (query or "")[:200],
        "intent": selection.intent,
        "intent_confidence": round(selection.intent_confidence, 4),
        "domain": selection.domain,
        "selected_skills": selection.selected_skills,
        "confidence": round(selection.confidence, 4),
        "router_stage": selection.router_stage,
        "stage_priority": selection.stage_priority,
        "fallback_level": selection.fallback_level,
        "execution_path": selection.execution_path,
    }
    try:
        with _TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def select_skills(query: str, mode: str, session_id: Optional[str] = None) -> SkillSelection:
    preload_skill_vectors()
    _load_perf()

    selection = SkillSelection()
    selection.execution_path = _complexity_path(query)

    intent, intent_conf = _detect_intent(query, timeout_ms=INTENT_TIMEOUT_MS)
    if intent is None:
        intent = "general"
        intent_conf = 0.0
    selection.intent = intent
    selection.intent_confidence = float(intent_conf)

    domain = INTENT_TO_DOMAIN.get(intent, "general")
    if intent_conf < INTENT_CONFIDENCE_FALLBACK:
        domain = "general"
    selection.domain = domain

    qvec = _vectorize(query or "")
    recent = _session_recent_skills.get(session_id or "", [])

    candidate_skills = _SKILLS
    if domain != "general":
        candidate_skills = [s for s in _SKILLS if s.domain == domain]
        if not candidate_skills:
            candidate_skills = _SKILLS

    ranked: List[Tuple[str, float, str]] = []
    for skill in candidate_skills:
        if not _allow_skill(skill, mode, selection.execution_path, intent, query):
            continue

        rule_score = _score_rule(skill, query)
        emb_score = _jaccard(qvec, _skill_vectors.get(skill.name, {}))

        if rule_score >= 0.62:
            stage = "rules"
            router_score = rule_score
        elif emb_score >= 0.50:
            stage = "embedding"
            router_score = emb_score
        else:
            stage = "fallback"
            router_score = max(rule_score, emb_score)

        perf_score = _success_rate(skill.name)
        latency_score = _latency_bonus(skill.name)
        final_score = (router_score * 0.6) + (perf_score * 0.3) + (latency_score * 0.1)

        if skill.name in DOMAIN_PRIORITY.get(domain, []):
            final_score *= DOMAIN_PRIORITY_BOOST
        if intent in {"research", "analysis"} and skill.name == "research-agent":
            final_score *= RESEARCH_BOOST

        if skill.name in recent:
            final_score *= SESSION_COOLDOWN_PENALTY
        final_score *= _failure_penalty(skill.name)

        ranked.append((skill.name, min(1.0, final_score), stage))

    ranked.sort(key=lambda x: x[1], reverse=True)
    if not ranked:
        _session_last_selection[session_id or ""] = selection
        _log_trace(query, selection)
        return selection

    best_stage = ranked[0][2]
    if best_stage == "fallback":
        selection.fallback_level = 1

    top_conf = ranked[0][1]
    selection.router_stage = best_stage
    selection.stage_priority = ROUTER_STAGE_PRIORITY.get(best_stage, ROUTER_STAGE_PRIORITY["fallback"])
    selection.confidence = top_conf

    if top_conf < CONFIDENCE_THRESHOLD:
        selection.router_stage = "fallback"
        selection.stage_priority = ROUTER_STAGE_PRIORITY["fallback"]
        selection.fallback_level = max(selection.fallback_level, 1)
        _session_last_selection[session_id or ""] = selection
        _log_trace(query, selection)
        return selection

    cap = _mode_caps(mode, selection.execution_path)
    chosen = [name for name, score, _ in ranked if score >= CONFIDENCE_THRESHOLD][:cap]
    selection.selected_skills = chosen
    selection.capsules = [(name, _load_capsule(name)) for name in chosen]

    if session_id and chosen:
        _session_recent_skills[session_id] = (recent + chosen)[-8:]

    _session_last_selection[session_id or ""] = selection
    _log_trace(query, selection)
    _debug(f"intent={intent} domain={domain} chosen={chosen} score={top_conf:.2f}")
    return selection


def inject_skill_capsules(base_prompt: str, selection: SkillSelection, token_budget: int = 4500) -> str:
    if not selection.capsules:
        return base_prompt

    primary = selection.capsules[0]
    additional = selection.capsules[1:]

    sections = [
        "\n\n**ACTIVE SKILLS (RUNTIME-SELECTED):**",
        f"- Primary: {primary[0]}",
        f"\n{primary[1]}",
    ]

    # Protect base prompt + primary capsule. Drop only additional capsules under budget pressure.
    prompt_so_far = base_prompt + "\n".join(sections)
    for name, capsule in additional:
        part = f"\n- Additional: {name}\n{capsule}"
        est = estimate_tokens([{"role": "system", "content": prompt_so_far + part}])
        if est > token_budget:
            break
        sections.append(part)
        prompt_so_far += part

    return base_prompt + "\n".join(sections)


def record_skill_outcome(session_id: Optional[str], success: bool, latency_ms: float, fallback_level: int = 0) -> None:
    _load_perf()
    if not session_id:
        return
    selection = _session_last_selection.get(session_id)
    if not selection or not selection.selected_skills:
        return

    for skill in selection.selected_skills:
        st = _skill_perf.setdefault(skill, {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "avg_latency_ms": 0.0,
        })
        attempts = float(st.get("attempts", 0.0)) + 1.0
        prev_avg = float(st.get("avg_latency_ms", 0.0))
        st["attempts"] = attempts
        if success and fallback_level == 0:
            st["successes"] = float(st.get("successes", 0.0)) + 1.0
        else:
            st["failures"] = float(st.get("failures", 0.0)) + 1.0

        st["avg_latency_ms"] = ((prev_avg * (attempts - 1.0)) + float(max(1.0, latency_ms))) / attempts

    _save_perf()


def get_last_skill_trace(session_id: Optional[str]) -> Dict[str, object]:
    selection = _session_last_selection.get(session_id or "", SkillSelection())
    return {
        "selected_skills": selection.selected_skills,
        "confidence": selection.confidence,
        "router_stage": selection.router_stage,
        "stage_priority": selection.stage_priority,
        "fallback_level": selection.fallback_level,
        "execution_path": selection.execution_path,
        "intent": selection.intent,
        "intent_confidence": selection.intent_confidence,
        "domain": selection.domain,
    }
