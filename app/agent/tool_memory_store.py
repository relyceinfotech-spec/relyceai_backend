from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.agent.reliability_runtime import semantic_intent_hash, freshness_decay
from app.auth import get_firestore_db
from app.config import (
    TOOL_MEMORY_TTL_SECONDS,
    TOOL_MEMORY_FRESHNESS_MIN,
    TOOL_MEMORY_MAX_ITEMS_PER_USER,
    TOOL_MEMORY_MAX_ITEMS_PER_SESSION,
    TOOL_SCHEMA_VERSION,
)


@dataclass
class ToolMemoryEntry:
    user_id: str
    session_id: str
    intent_hash: str
    tool_name: str
    tool_schema_version: str
    fingerprint: str
    key_facts: List[str]
    source_links: List[str]
    evidence_ids: List[str]
    confidence: float
    source_type: str
    timestamp: float
    freshness_score: float
    last_access_ts: float


class ToolMemoryStore:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str, str, str], ToolMemoryEntry] = {}

    @staticmethod
    def _key(user_id: str, intent_hash: str, tool_name: str, schema_version: str) -> Tuple[str, str, str, str]:
        return (str(user_id or ""), str(intent_hash or ""), str(tool_name or ""), str(schema_version or ""))

    def _prune_bounds(self, user_id: str, session_id: str) -> None:
        now = time.time()
        ttl = max(1, int(TOOL_MEMORY_TTL_SECONDS))
        # TTL prune first
        expired_keys = [k for k, v in self._cache.items() if now - v.timestamp > ttl]
        for k in expired_keys:
            self._cache.pop(k, None)

        # LRU per-user bound
        user_items = [(k, v) for k, v in self._cache.items() if v.user_id == user_id]
        if len(user_items) > TOOL_MEMORY_MAX_ITEMS_PER_USER:
            user_items.sort(key=lambda kv: kv[1].last_access_ts)
            for k, _ in user_items[: len(user_items) - TOOL_MEMORY_MAX_ITEMS_PER_USER]:
                self._cache.pop(k, None)

        # LRU per-session bound
        session_items = [(k, v) for k, v in self._cache.items() if v.user_id == user_id and v.session_id == session_id]
        if len(session_items) > TOOL_MEMORY_MAX_ITEMS_PER_SESSION:
            session_items.sort(key=lambda kv: kv[1].last_access_ts)
            for k, _ in session_items[: len(session_items) - TOOL_MEMORY_MAX_ITEMS_PER_SESSION]:
                self._cache.pop(k, None)

    def put(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        tool_name: str,
        fingerprint: str,
        key_facts: List[str],
        source_links: List[str],
        evidence_ids: Optional[List[str]] = None,
        confidence: float,
        source_type: str = "tool",
        schema_version: str = TOOL_SCHEMA_VERSION,
    ) -> None:
        if not user_id or not tool_name:
            return
        now = time.time()
        intent_hash = semantic_intent_hash(query)
        entry = ToolMemoryEntry(
            user_id=user_id,
            session_id=session_id or "",
            intent_hash=intent_hash,
            tool_name=tool_name,
            tool_schema_version=schema_version,
            fingerprint=str(fingerprint or "")[:200],
            key_facts=[str(x)[:240] for x in (key_facts or [])][:8],
            source_links=[str(x)[:300] for x in (source_links or [])][:8],
            evidence_ids=[str(x)[:120] for x in (evidence_ids or [])][:8],
            confidence=float(confidence or 0.0),
            source_type=str(source_type or "tool")[:32],
            timestamp=now,
            freshness_score=1.0,
            last_access_ts=now,
        )
        self._cache[self._key(user_id, intent_hash, tool_name, schema_version)] = entry
        self._prune_bounds(user_id, session_id)
        self._persist_firestore(entry)

    def get(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        tool_name: str,
        schema_version: str = TOOL_SCHEMA_VERSION,
        current_info: bool = False,
    ) -> Optional[ToolMemoryEntry]:
        intent_hash = semantic_intent_hash(query)
        key = self._key(user_id, intent_hash, tool_name, schema_version)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._load_firestore(user_id, intent_hash, tool_name, schema_version)
            if entry:
                self._cache[key] = entry
        if entry is None:
            return None

        now = time.time()
        age = now - entry.timestamp
        if age > max(1, int(TOOL_MEMORY_TTL_SECONDS)):
            self._cache.pop(key, None)
            return None

        freshness = freshness_decay(age, TOOL_MEMORY_TTL_SECONDS)
        if freshness < float(TOOL_MEMORY_FRESHNESS_MIN):
            return None
        if current_info and freshness < 0.60:
            return None

        entry.freshness_score = freshness
        entry.last_access_ts = now
        return entry

    @staticmethod
    def _doc_id(entry: ToolMemoryEntry) -> str:
        return f"{entry.user_id}:{entry.intent_hash}:{entry.tool_name}:{entry.tool_schema_version}"

    def _persist_firestore(self, entry: ToolMemoryEntry) -> None:
        try:
            db = get_firestore_db()
            if not db:
                return
            db.collection("toolMemory").document(self._doc_id(entry)).set(
                {
                    "user_id": entry.user_id,
                    "session_id": entry.session_id,
                    "intent_hash": entry.intent_hash,
                    "tool_name": entry.tool_name,
                    "tool_schema_version": entry.tool_schema_version,
                    "fingerprint": entry.fingerprint,
                    "key_facts": entry.key_facts,
                    "source_links": entry.source_links,
                    "evidence_ids": entry.evidence_ids,
                    "confidence": entry.confidence,
                    "source_type": entry.source_type,
                    "timestamp": entry.timestamp,
                    "freshness_score": entry.freshness_score,
                    "last_access_ts": entry.last_access_ts,
                },
                merge=True,
            )
        except Exception:
            # Non-blocking persistence path.
            return

    def _load_firestore(self, user_id: str, intent_hash: str, tool_name: str, schema_version: str) -> Optional[ToolMemoryEntry]:
        try:
            db = get_firestore_db()
            if not db:
                return None
            doc = db.collection("toolMemory").document(f"{user_id}:{intent_hash}:{tool_name}:{schema_version}").get()
            if not doc.exists:
                return None
            data = doc.to_dict() or {}
            return ToolMemoryEntry(
                user_id=str(data.get("user_id") or user_id),
                session_id=str(data.get("session_id") or ""),
                intent_hash=str(data.get("intent_hash") or intent_hash),
                tool_name=str(data.get("tool_name") or tool_name),
                tool_schema_version=str(data.get("tool_schema_version") or schema_version),
                fingerprint=str(data.get("fingerprint") or ""),
                key_facts=list(data.get("key_facts") or []),
                source_links=list(data.get("source_links") or []),
                evidence_ids=list(data.get("evidence_ids") or []),
                confidence=float(data.get("confidence") or 0.0),
                source_type=str(data.get("source_type") or "tool"),
                timestamp=float(data.get("timestamp") or 0.0),
                freshness_score=float(data.get("freshness_score") or 0.0),
                last_access_ts=float(data.get("last_access_ts") or 0.0),
            )
        except Exception:
            return None


_tool_memory_singleton: Optional[ToolMemoryStore] = None


def get_tool_memory_store() -> ToolMemoryStore:
    global _tool_memory_singleton
    if _tool_memory_singleton is None:
        _tool_memory_singleton = ToolMemoryStore()
    return _tool_memory_singleton
