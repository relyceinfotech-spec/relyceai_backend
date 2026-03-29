import pytest
import time
from app.state.task_state_engine import (
    create_task_state,
    get_task_state,
    update_task_graph,
    set_task_status,
    summarize_task_history,
    clear_session_tasks
)
from app.state.plan_graph import PlanGraph, PlanNode, NodeStatus

# ============================================
# STATE CREATION & CHECKPOINTING
# ============================================

def test_task_state_creation():
    session_id = "test_session_create"
    task_id = "task_001"
    clear_session_tasks(session_id)
    
    state = create_task_state(session_id, task_id, "ADAPTIVE_CODE_PLAN")
    assert state.task_id == task_id
    assert state.session_id == session_id
    assert state.planning_mode == "ADAPTIVE_CODE_PLAN"
    assert state.status == "RUNNING"
    assert state.total_tool_calls == 0
    
    # Verify retrieval
    retrieved = get_task_state(session_id, task_id)
    assert retrieved is not None
    assert retrieved.task_id == task_id

def test_task_state_checkpointing_size_caps():
    session_id = "test_session_checkpoint"
    task_id = "task_002"
    clear_session_tasks(session_id)
    
    create_task_state(session_id, task_id, "SEQUENTIAL_PLAN")
    
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    n = PlanNode(node_id="N1", action_type="Search", payload={}, status=NodeStatus.COMPLETED, result={"outcome_summary": "Done"})
    graph.add_node(n)

    update_task_graph(
        session_id, 
        task_id, 
        plan_graph_snapshot=graph.serialize(),
        new_status="RUNNING"
    )
    
    state = get_task_state(session_id, task_id)
    
    assert state.total_tool_calls == 1
    assert "N1" in state.plan_graph_snapshot["nodes"]
    
    # Assert result preserved
    saved_outcome = state.plan_graph_snapshot["nodes"]["N1"]["result"]["outcome_summary"]
    assert saved_outcome == "Done"

# ============================================
# RESUME & SUMMARIZATION LOGIC
# ============================================

def test_task_summarization():
    session_id = "test_session_summary"
    task_id = "task_003"
    clear_session_tasks(session_id)
    
    create_task_state(session_id, task_id, "ADAPTIVE_CODE_PLAN")
    
    graph = PlanGraph(graph_id=task_id, session_id=session_id)
    graph.add_node(PlanNode("A", "Search", {}, status=NodeStatus.COMPLETED, result={"output": "Found docs"}))
    graph.add_node(PlanNode("B", "WriteFile", {}, status=NodeStatus.COMPLETED, result={"output": "Created App.js"}))
    graph.add_node(PlanNode("C", "RunCommand", {}, status=NodeStatus.FAILED, result={"output": "Missing script"}))

    update_task_graph(session_id, task_id, graph.serialize())
    
    summary = summarize_task_history(session_id, task_id)
    
    assert "Resume Context" in summary
    assert "Node A: Executed Search -> COMPLETED" in summary
    assert "Node B: Executed WriteFile -> COMPLETED" in summary
    assert "Node C: Executed RunCommand -> FAILED" in summary
    assert "Missing script" in summary

def test_task_isolation():
    session_a = "user_A"
    session_b = "user_B"
    task_id = "task_shared_id" # Same task ID, different sessions
    
    clear_session_tasks(session_a)
    clear_session_tasks(session_b)
    
    create_task_state(session_a, task_id, "ADAPTIVE_CODE_PLAN")
    create_task_state(session_b, task_id, "SEQUENTIAL_PLAN")
    
    graph_a = PlanGraph(graph_id=task_id, session_id=session_a)
    graph_a.add_node(PlanNode("N1", "Act", {}, status=NodeStatus.COMPLETED))
    
    update_task_graph(session_a, task_id, graph_a.serialize())
    
    state_a = get_task_state(session_a, task_id)
    state_b = get_task_state(session_b, task_id)
    
    assert state_a.total_tool_calls == 1
    assert "N1" in state_a.plan_graph_snapshot["nodes"]
    
    assert state_b.total_tool_calls == 0
    assert not state_b.plan_graph_snapshot

def test_task_status_updates():
    session_id = "test_session_status"
    task_id = "task_004"
    clear_session_tasks(session_id)
    
    create_task_state(session_id, task_id, "ADAPTIVE_CODE_PLAN")
    set_task_status(session_id, task_id, "PAUSED")
    
    state = get_task_state(session_id, task_id)
    assert state.status == "PAUSED"
