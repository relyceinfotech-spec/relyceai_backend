import pytest
from app.state.plan_graph import PlanGraph, PlanNode, NodeStatus

def test_add_and_retrieve_ready_nodes():
    graph = PlanGraph(graph_id="g1", session_id="s1")
    
    n1 = PlanNode(node_id="A", action_type="TOOL_CALL", payload={"type": "read"})
    n2 = PlanNode(node_id="B", action_type="TOOL_CALL", payload={"type": "write"}, dependencies=["A"])
    
    graph.add_node(n1)
    graph.add_node(n2)
    
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].node_id == "A"
    
    # Mark A completed
    graph.mark_completed("A", {"status": "ok"})
    
    # B should now be ready
    ready2 = graph.get_ready_nodes()
    assert len(ready2) == 1
    assert ready2[0].node_id == "B"

def test_cycle_detection():
    graph = PlanGraph(graph_id="g_cycle", session_id="s1")
    
    n1 = PlanNode(node_id="A", action_type="TOOL", payload={}, dependencies=["B"])
    n2 = PlanNode(node_id="B", action_type="TOOL", payload={}, dependencies=["A"])
    
    graph.add_node(n1)
    graph.add_node(n2)
    
    assert graph.validate_cycle_free() is False

def test_failure_cascade():
    graph = PlanGraph(graph_id="g_fail", session_id="s1")
    
    n1 = PlanNode(node_id="A", action_type="TOOL", payload={})
    n2 = PlanNode(node_id="B", action_type="TOOL", payload={}, dependencies=["A"])
    n3 = PlanNode(node_id="C", action_type="TOOL", payload={}, dependencies=["A"])
    n4 = PlanNode(node_id="D", action_type="TOOL", payload={}, dependencies=["C"])
    n5 = PlanNode(node_id="X", action_type="TOOL", payload={}) # Independent branch
    
    for n in [n1, n2, n3, n4, n5]:
        graph.add_node(n)
        
    graph.mark_failed("A")
    
    assert graph.nodes["A"].status == NodeStatus.FAILED
    assert graph.nodes["B"].status == NodeStatus.BLOCKED
    assert graph.nodes["C"].status == NodeStatus.BLOCKED
    assert graph.nodes["D"].status == NodeStatus.BLOCKED
    assert graph.nodes["X"].status == NodeStatus.PENDING # Independent branch untouched

def test_serialization_round_trip():
    graph = PlanGraph(graph_id="g_serial", session_id="s_serial")
    n1 = PlanNode(node_id="A", action_type="TOOL", role="planner", payload={"key": "val"})
    n2 = PlanNode(
        node_id="B",
        action_type="TOOL",
        role="synthesizer",
        payload={},
        dependencies=["A"],
        status=NodeStatus.COMPLETED,
        result={"done": True},
        role_fallback_applied=True,
        role_resolution_source="test",
    )
    
    graph.add_node(n1)
    graph.add_node(n2)
    
    dump = graph.serialize()
    
    restored = PlanGraph.deserialize(dump)
    
    assert restored.graph_id == "g_serial"
    assert restored.session_id == "s_serial"
    assert len(restored.nodes) == 2
    assert restored.nodes["B"].status == NodeStatus.COMPLETED
    assert restored.nodes["B"].result == {"done": True}
    assert restored.nodes["A"].payload == {"key": "val"}
    assert restored.nodes["A"].role == "planner"
    assert restored.nodes["B"].role_fallback_applied is True
    assert restored.nodes["B"].role_resolution_source == "test"
