"""
Learning Config Store

Versioned config artifacts for planner/routing/tool priors.
No direct prompt mutation is allowed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent / "learning_configs"
ACTIVE_PTR = BASE_DIR / "active_version.json"


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dirs() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)


def default_config() -> Dict[str, Any]:
    return {
        "version": "v1_default",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "planner_patterns": {
            "comparison": ["search", "search", "extract_facts"],
            "numeric_fact": ["search"],
            "research": ["search", "search", "extract_facts"],
        },
        "routing_priors": {
            "simple_fact_shortcut": True,
            "verification_scope": "factual_research_only",
        },
        "tool_priors": {},
    }


def load_active_config() -> Dict[str, Any]:
    ensure_dirs()
    if not ACTIVE_PTR.exists():
        cfg = default_config()
        version = cfg["version"]
        (BASE_DIR / f"{version}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        ACTIVE_PTR.write_text(json.dumps({"active_version": version}, indent=2), encoding="utf-8")
        return cfg

    ptr = json.loads(ACTIVE_PTR.read_text(encoding="utf-8"))
    version = ptr.get("active_version")
    if not version:
        return default_config()

    file = BASE_DIR / f"{version}.json"
    if not file.exists():
        return default_config()

    return json.loads(file.read_text(encoding="utf-8"))


def write_new_version(payload: Dict[str, Any], note: str = "") -> str:
    ensure_dirs()
    tag = _now_tag()
    version = f"cfg_{tag}"
    content = dict(payload or {})
    content["version"] = version
    content["updated_at"] = datetime.now(timezone.utc).isoformat()
    content["note"] = note

    file = BASE_DIR / f"{version}.json"
    file.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return version


def activate_version(version: str) -> bool:
    ensure_dirs()
    file = BASE_DIR / f"{version}.json"
    if not file.exists():
        return False
    ACTIVE_PTR.write_text(json.dumps({"active_version": version}, indent=2), encoding="utf-8")
    return True


def rollback_to(version: str) -> bool:
    return activate_version(version)
