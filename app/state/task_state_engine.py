import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# -------------------------------------------------------------
# SESSION-SCOPED TASK STORE
# -------------------------------------------------------------
# Structure: { session_id: { task_id: TaskState } }
_TASK_STORE: Dict[str, Dict[str, "TaskState"]] = {}

@dataclass
class TaskState:
    task_id: str
    session_id: str
    planning_mode: str
    plan_graph_snapshot: Dict[str, Any] = field(default_factory=dict)
    total_tool_calls: int = 0  # Phase 4C Explicit Budget Tracker
    status: str = "PENDING"  # PENDING, RUNNING, PAUSED, FAILED, COMPLETED
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

# -------------------------------------------------------------
# TTL EXPIRY CLEANUP
# -------------------------------------------------------------
def cleanup_expired_tasks():
    """
    Remove PAUSED tasks older than 30 minutes.
    Remove FAILED tasks older than 24 hours.
    """
    now = time.time()
    for sid in list(_TASK_STORE.keys()):
        for tid in list(_TASK_STORE[sid].keys()):
            state = _TASK_STORE[sid][tid]
            age = now - state.updated_at
            
            if state.status == "PAUSED" and age > 1800:  # 30 mins
                del _TASK_STORE[sid][tid]
            elif state.status == "FAILED" and age > 86400:  # 24 hours
                del _TASK_STORE[sid][tid]
                
        # cleanup empty session dicts
        if not _TASK_STORE[sid]:
            del _TASK_STORE[sid]

# -------------------------------------------------------------
# CRUD OPERATIONS
# -------------------------------------------------------------

def create_task_state(session_id: str, task_id: str, planning_mode: str, plan_graph_snapshot: Dict[str, Any] = None) -> TaskState:
    """Initialize state for multi-step tasks using a deterministic graph."""
    cleanup_expired_tasks()
    
    if session_id not in _TASK_STORE:
        _TASK_STORE[session_id] = {}
        
    state = TaskState(
        task_id=task_id,
        session_id=session_id,
        planning_mode=planning_mode,
        plan_graph_snapshot=plan_graph_snapshot or {},
        status="RUNNING"
    )
    _TASK_STORE[session_id][task_id] = state
    return state

def get_task_state(session_id: str, task_id: str) -> Optional[TaskState]:
    """Retrieve an existing task state if present."""
    cleanup_expired_tasks()
    return _TASK_STORE.get(session_id, {}).get(task_id)

def update_task_graph(session_id: str, task_id: str, plan_graph_snapshot: Dict[str, Any], new_status: Optional[str] = None):
    """
    Checkpoint the DAG state.
    Replaces linear history array with current structural snapshot.
    """
    state = get_task_state(session_id, task_id)
    if not state:
        return
        
    state.plan_graph_snapshot = plan_graph_snapshot
    
    # Simple explicit budget tracking
    state.total_tool_calls += 1 
    
    if new_status:
        state.status = new_status
        
    state.updated_at = time.time()

def set_task_status(session_id: str, task_id: str, new_status: str):
    """Updates the explicit status (e.g. PAUSED, COMPLETED, FAILED)."""
    state = get_task_state(session_id, task_id)
    if state:
        state.status = new_status
        state.updated_at = time.time()

# -------------------------------------------------------------
# CONTEXT SUMMARIZATION
# -------------------------------------------------------------
def summarize_task_history(session_id: str, task_id: str) -> str:
    """
    Summarize past steps into a compact block mapped for LLM re-injection.
    Reads from the PlanGraph executed nodes.
    """
    state = get_task_state(session_id, task_id)
    if not state or not state.plan_graph_snapshot:
        return "No prior execution history."
        
    summary_lines = [f"[Task {task_id} Resume Context (DAG)]"]
    nodes = state.plan_graph_snapshot.get("nodes", {})
    
    completed = 0
    for n_id, node_data in nodes.items():
        if node_data.get("status") in ["COMPLETED", "FAILED"]:
            completed += 1
            action = node_data.get("action_type", "Unknown")
            status = node_data.get("status")
            out = str(node_data.get("result", {}))[:200]
            summary_lines.append(f"Node {n_id}: Executed {action} -> {status}. Output excerpt: {out}")
            
    if completed == 0:
        return "Graph initialized, but no nodes have executed."
        
    return "\\n".join(summary_lines)

def clear_session_tasks(session_id: str):
    """For testing sandbox isolation."""
    if session_id in _TASK_STORE:
        del _TASK_STORE[session_id]
