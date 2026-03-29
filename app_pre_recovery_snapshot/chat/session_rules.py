"""
Relyce AI - Session Rules Manager
In-memory per-session constraint store for the Safe Chat Agent.

Rules persist across turns within a session and decay automatically
after 2 turns without re-affirmation or on topic change.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

DEBUG_SAFE_AGENT = os.getenv("DEBUG_SAFE_AGENT", "false").lower() == "true"


# ============================================
# ENUMS & DATA STRUCTURES
# ============================================

class Severity(str, Enum):
    STRICT = "STRICT"
    SOFT = "SOFT"


class OverrideScope(str, Enum):
    CONTENT = "content"    # Clears count, length, depth
    FORMAT = "format"      # Clears format preference only
    GLOBAL = "global"      # Clears everything


@dataclass
class Rule:
    """A single constraint rule with severity and decay tracking."""
    key: str                    # e.g., "count_limit", "format_preference"
    value: object               # e.g., 3, "bullet", "short"
    severity: Severity          # STRICT or SOFT
    set_at_turn: int = 0        # Turn number when set/re-affirmed
    last_affirmed_turn: int = 0 # Last turn user re-affirmed this

    def is_strict(self) -> bool:
        return self.severity == Severity.STRICT

    def is_soft(self) -> bool:
        return self.severity == Severity.SOFT


@dataclass
class SessionState:
    """Full session state for a (user_id, chat_id) pair."""
    rules: Dict[str, Rule] = field(default_factory=dict)
    turn_count: int = 0
    last_query: str = ""
    last_query_embedding: List[float] = field(default_factory=list)
    last_assistant_output: str = ""

    def has_strict_rules(self) -> bool:
        return any(r.is_strict() for r in self.rules.values())

    def has_any_rules(self) -> bool:
        return len(self.rules) > 0

    def get_strict_rules(self) -> Dict[str, Rule]:
        return {k: v for k, v in self.rules.items() if v.is_strict()}

    def get_soft_rules(self) -> Dict[str, Rule]:
        return {k: v for k, v in self.rules.items() if v.is_soft()}


# ============================================
# IN-MEMORY STORE
# ============================================

# Key: (user_id, chat_id) → SessionState
_session_store: Dict[Tuple[str, str], SessionState] = {}

# Content-related rule keys (cleared by content override)
CONTENT_RULE_KEYS = {"count_limit", "length_limit", "depth_limit"}
# Format-related rule keys (cleared by format override)
FORMAT_RULE_KEYS = {"format_preference"}
# All rule keys
ALL_RULE_KEYS = CONTENT_RULE_KEYS | FORMAT_RULE_KEYS | {"tone_preference"}

# Decay: rules expire after this many turns without re-affirmation
DECAY_TURNS = 2
# Topic change threshold for cosine similarity
TOPIC_CHANGE_THRESHOLD = 0.5


# ============================================
# PUBLIC API
# ============================================

def get_session(user_id: str, chat_id: str) -> SessionState:
    """Get or create session state."""
    key = (user_id, chat_id)
    if key not in _session_store:
        _session_store[key] = SessionState()
    return _session_store[key]


def increment_turn(user_id: str, chat_id: str, query: str, assistant_output: str = "") -> None:
    """Advance turn counter and store last query/output for reference resolution."""
    session = get_session(user_id, chat_id)
    session.turn_count += 1
    session.last_query = query
    session.last_assistant_output = assistant_output


def update_rules(user_id: str, chat_id: str, new_rules: Dict[str, Rule]) -> None:
    """
    Merge new detected rules into session state.
    Existing rules are overwritten if same key is detected again.
    """
    session = get_session(user_id, chat_id)
    for key, rule in new_rules.items():
        rule.set_at_turn = session.turn_count
        rule.last_affirmed_turn = session.turn_count
        session.rules[key] = rule

    if DEBUG_SAFE_AGENT:
        active = len(session.rules)
        strict = sum(1 for r in session.rules.values() if r.is_strict())
        print(f"[SessionRules] updated: {active} rules ({strict} strict) at turn {session.turn_count}")


def confirmation_reset(user_id: str, chat_id: str, reaffirmed_keys: List[str]) -> bool:
    """
    If user re-affirms a rule, reset its decay timer.
    Returns True if any confirmations happened (skip embedding).
    """
    session = get_session(user_id, chat_id)
    confirmed = False
    for key in reaffirmed_keys:
        if key in session.rules:
            session.rules[key].last_affirmed_turn = session.turn_count
            confirmed = True

    if confirmed and DEBUG_SAFE_AGENT:
        print(f"[SessionRules] confirmation reset for: {reaffirmed_keys}")

    return confirmed


def apply_decay(user_id: str, chat_id: str, current_query_embedding: Optional[List[float]] = None) -> int:
    """
    Expire rules that are stale (2 turns) or on topic change.
    Skip embedding if no active rules.
    Returns count of decayed rules.
    """
    session = get_session(user_id, chat_id)

    if not session.has_any_rules():
        return 0

    decayed_keys = []

    for key, rule in list(session.rules.items()):
        turns_since = session.turn_count - rule.last_affirmed_turn

        # Time-based decay: 2 turns without re-affirmation
        if turns_since >= DECAY_TURNS:
            decayed_keys.append(key)
            continue

    # Topic-based decay (only if we have embeddings)
    if current_query_embedding and session.last_query_embedding:
        from app.llm.embeddings import cosine_similarity
        similarity = cosine_similarity(current_query_embedding, session.last_query_embedding)

        if similarity < TOPIC_CHANGE_THRESHOLD:
            # Topic changed — decay all non-reaffirmed rules
            for key, rule in list(session.rules.items()):
                if key not in decayed_keys and rule.last_affirmed_turn < session.turn_count:
                    decayed_keys.append(key)

            if DEBUG_SAFE_AGENT:
                print(f"[SessionRules] topic change detected (sim={similarity:.2f}), decaying rules")

    # Remove decayed rules
    for key in decayed_keys:
        del session.rules[key]

    # Store embedding for next turn's comparison
    if current_query_embedding:
        session.last_query_embedding = current_query_embedding

    if decayed_keys and DEBUG_SAFE_AGENT:
        print(f"[SessionRules] decayed {len(decayed_keys)} rules: {decayed_keys}")

    return len(decayed_keys)


def apply_override(user_id: str, chat_id: str, scope: OverrideScope) -> None:
    """Clear rules based on override scope."""
    session = get_session(user_id, chat_id)

    if scope == OverrideScope.GLOBAL:
        session.rules.clear()
    elif scope == OverrideScope.CONTENT:
        for key in CONTENT_RULE_KEYS:
            session.rules.pop(key, None)
    elif scope == OverrideScope.FORMAT:
        for key in FORMAT_RULE_KEYS:
            session.rules.pop(key, None)

    if DEBUG_SAFE_AGENT:
        print(f"[SessionRules] override applied: scope={scope.value}, remaining={len(session.rules)}")


def get_active_rules(user_id: str, chat_id: str) -> Dict[str, Rule]:
    """Get all current active rules."""
    session = get_session(user_id, chat_id)
    return dict(session.rules)


def get_last_assistant_output(user_id: str, chat_id: str) -> str:
    """Get last assistant output for reference resolution."""
    session = get_session(user_id, chat_id)
    return session.last_assistant_output


def clear_session(user_id: str, chat_id: str) -> None:
    """Fully clear a session's rules."""
    key = (user_id, chat_id)
    if key in _session_store:
        del _session_store[key]
