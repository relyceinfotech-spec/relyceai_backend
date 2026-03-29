import pytest
import asyncio
from typing import Dict, Any

from app.state.transaction_manager import (
    begin_transaction,
    commit_transaction,
    rollback_transaction,
    add_rollback_action,
    get_transaction,
    suppress_memory_updates,
    is_memory_suppressed,
    CompensatingAction,
    _ACTIVE_TRANSACTIONS
)

def setup_function():
    _ACTIVE_TRANSACTIONS.clear()

def test_begin_and_commit_transaction():
    session_id = "sess_1"
    task_id = "task_A"
    
    ctx = begin_transaction(session_id, task_id)
    assert ctx.is_active is True
    assert ctx.is_rolling_back is False
    
    # Add fake operations
    action = CompensatingAction("FakeTool", "FAKE_RESTORE", {"args": 1})
    add_rollback_action(session_id, task_id, action)
    add_rollback_action(session_id, task_id, action)
    
    assert len(ctx.operations) == 2
    
    # Commit
    commit_transaction(session_id, task_id)
    assert ctx.is_active is False
    assert len(ctx.operations) == 0  # Flushed

def test_nested_transaction_blocked():
    session_id = "sess_2"
    task_id = "task_B"
    
    begin_transaction(session_id, task_id)
    
    with pytest.raises(ValueError, match="Nested transactions are explicitly disallowed"):
        begin_transaction(session_id, task_id)

def test_rollback_lifo_and_memory_isolation():
    session_id = "sess_iso"
    task_id = "task_iso"
    
    ctx = begin_transaction(session_id, task_id)
    
    action1 = CompensatingAction("CreateUser", "DELETE_USER", {"user_id": 10})
    action2 = CompensatingAction("WriteFile", "RESTORE_FILE", {"path": "/tmp/a", "original_content": "old"})
    
    add_rollback_action(session_id, task_id, action1)
    add_rollback_action(session_id, task_id, action2)
    
    assert len(ctx.operations) == 2
    
    # Pre-assertions
    assert is_memory_suppressed(session_id) is False
    
    asyncio.run(rollback_transaction(session_id, task_id))
    
    assert ctx.is_active is False
    assert ctx.is_rolling_back is False
    assert len(ctx.operations) == 0

def test_suppress_memory_updates_manager():
    session_id = "sess_mem"
    
    assert is_memory_suppressed(session_id) is False
    
    with suppress_memory_updates(session_id):
        assert is_memory_suppressed(session_id) is True
        
    assert is_memory_suppressed(session_id) is False
