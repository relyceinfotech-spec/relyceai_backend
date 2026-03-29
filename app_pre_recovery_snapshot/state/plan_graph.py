"""
Phase 5: Deterministic Planning Graph Engine
Provides a Directed Acyclic Graph (DAG) for dependency-aware sequential execution.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

class NodeStatus:
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"  # Upstream dependency failed

@dataclass
class PlanNode:
    node_id: str
    action_type: str  # TOOL_CALL | REASONING | VALIDATION | REPAIR
    payload: Dict
    dependencies: List[str] = field(default_factory=list)
    status: str = NodeStatus.PENDING
    result: Optional[Dict] = None

@dataclass
class PlanGraph:
    graph_id: str
    session_id: str
    nodes: Dict[str, PlanNode] = field(default_factory=dict)
    
    def add_node(self, node: PlanNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node {node.node_id} already exists in the graph.")
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[PlanNode]:
        """Returns the node for the given node_id, or None if not found."""
        return self.nodes.get(node_id)

    def get_ready_nodes(self) -> List[PlanNode]:
        """
        Returns a list of nodes that are either explicitly READY or PENDING
        with all dependencies COMPLETED.
        """
        ready = []
        for node in self.nodes.values():
            if node.status == NodeStatus.READY:
                ready.append(node)
            elif node.status == NodeStatus.PENDING:
                if all(self.nodes[dep].status == NodeStatus.COMPLETED for dep in node.dependencies):
                    node.status = NodeStatus.READY
                    ready.append(node)
                    
        # Sort heuristically (could be priority based, for now ID based for determinism)
        ready.sort(key=lambda n: n.node_id)
        return ready

    def mark_completed(self, node_id: str, result: Dict) -> None:
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        node.status = NodeStatus.COMPLETED
        node.result = result

    def mark_failed(self, node_id: str) -> None:
        """
        Marks a node as FAILED.
        Cascades BLOCKED status to all downstream dependent nodes.
        """
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        node.status = NodeStatus.FAILED
        
        # Cascade failure (Block downstream)
        self._cascade_block(node_id)
        
    def _cascade_block(self, failed_node_id: str) -> None:
        """Recursively blocks nodes that depend on the failed node."""
        for n_id, node in self.nodes.items():
            if node.status in [NodeStatus.PENDING, NodeStatus.READY]:
                if failed_node_id in node.dependencies:
                    node.status = NodeStatus.BLOCKED
                    self._cascade_block(n_id)

    def validate_cycle_free(self) -> bool:
        """Returns True if the graph is a valid DAG (no cycles)."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _is_cyclic(n_id: str) -> bool:
            visited.add(n_id)
            rec_stack.add(n_id)
            
            node = self.nodes.get(n_id)
            if node:
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        if _is_cyclic(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
                        
            rec_stack.remove(n_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if _is_cyclic(node_id):
                    return False
        return True

    def is_fully_completed(self) -> bool:
        """Returns True if all reachable nodes are COMPLETED."""
        for node in self.nodes.values():
            if node.status not in [NodeStatus.COMPLETED, NodeStatus.BLOCKED]:
                return False
        return True

    def serialize(self) -> Dict:
        """Dumps graph to a secure dictionary for persistence."""
        return {
            "graph_id": self.graph_id,
            "session_id": self.session_id,
            "nodes": {
                n_id: {
                    "node_id": node.node_id,
                    "action_type": node.action_type,
                    "payload": node.payload,
                    "dependencies": node.dependencies,
                    "status": node.status,
                    "result": node.result
                } for n_id, node in self.nodes.items()
            }
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'PlanGraph':
        """Restores a PlanGraph from a dictionary representation."""
        graph = cls(
            graph_id=data.get("graph_id", "restored_graph"),
            session_id=data.get("session_id", "unknown")
        )
        for n_id, n_data in data.get("nodes", {}).items():
            node = PlanNode(
                node_id=n_data["node_id"],
                action_type=n_data["action_type"],
                payload=n_data.get("payload", {}),
                dependencies=n_data.get("dependencies", []),
                status=n_data.get("status", NodeStatus.PENDING),
                result=n_data.get("result")
            )
            graph.add_node(node)
        return graph
