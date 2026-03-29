"""
Multi-Layer Memory
Stores session history, task progress, and knowledge findings.
"""
from typing import List, Dict, Any, Optional

class SessionMemory:
    """Stores query/response pairs for the current chat session."""
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_interaction(self, query: str, response: str):
        self.history.append({"query": query, "response": response})

    def get_context(self, limit: int = 5) -> List[Dict[str, str]]:
        return self.history[-limit:]

class TaskMemory:
    """Stores the state of the current autonomous goal."""
    def __init__(self):
        self.nodes: List[Dict] = []
        self.results: List[Dict] = []
        self.reflections: List[str] = []

    def record_step(self, node_id: str, outcome: Any):
        self.results.append({"node_id": node_id, "outcome": outcome})

class KnowledgeMemory:
    """Lightweight RAG-style storage for tool results traversal."""
    def __init__(self):
        self.facts: List[str] = []

    def add_fact(self, fact: str):
        if fact not in self.facts:
            self.facts.append(fact)

    def retrieve(self, query: str) -> List[str]:
        # Simple keyword matching for now
        q = query.lower()
        return [f for f in self.facts if any(word in f.lower() for word in q.split())][:5]
