"""
Graph Builder
Compiles dynamic LLM strategy strings (or linear inputs) into a deterministic PlanGraph. 
"""
from typing import List, Dict, Any, Optional
import uuid

from app.state.plan_graph import PlanGraph, PlanNode

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
        
        node = PlanNode(
            node_id=node_id,
            action_type="REASONING" if "think" in step.lower() else "TOOL_CALL",
            payload={"instruction": step},
            dependencies=deps
        )
        graph.add_node(node)
        prev_id = node_id
        
    return graph

def compile_plan_graph(session_id: str, task_id: str, query: str, context: Optional[str] = None) -> PlanGraph:
    """
    Intelligently compiles a query into a dependency graph.
    In a full production setting, this would use a fast, focused LLM call (like Haiku) 
    to emit a hard-structured JSON DAG. 
    
    For MVP Phase 5, we parse known heuristic patterns into graphs.
    """
    # 1. Very basic routing heuristically 
    lower_query = query.lower()
    
    if "and" in lower_query and "then" in lower_query:
        # Example pattern: "search X and search Y then summarize"
        # N1: Search X, N2: Search Y, N3: Summarize (depends on N1, N2)
        parts = lower_query.split("then")
        parallels = parts[0].split("and")
        
        graph = PlanGraph(graph_id=task_id, session_id=session_id)
        
        parallel_ids = []
        for idx, p in enumerate(parallels):
            n_id = f"P{idx+1}"
            node = PlanNode(node_id=n_id, action_type="TOOL_CALL", payload={"instruction": p.strip()})
            graph.add_node(node)
            parallel_ids.append(n_id)
            
        n_final = PlanNode(
            node_id="FINAL", 
            action_type="REASONING", 
            payload={"instruction": parts[1].strip()},
            dependencies=parallel_ids
        )
        graph.add_node(n_final)
        return graph

    # Fallback to single monolithic node for simple queries
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    node = PlanNode(node_id="N1", action_type="TOOL_CALL", payload={"instruction": query})
    graph.add_node(node)
    
    return graph
