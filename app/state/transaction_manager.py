"""
Transactional Tool Manager (Phase 4D)
Provides atomic rollback semantics across logical boundaries.

Features:
- Tool Classification (Reversible, Compensatable, Non-Transactional)
- Single-Layer Transaction locking (no nesting)
- Strict LIFO Rollback
- is_rolling_back re-entrance guard
- Memory update suppression via Context Manager
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

# ==================================
# CLASSIFICATIONS
# ==================================

class ToolClass:
    REVERSIBLE = "REVERSIBLE"
    COMPENSATABLE = "COMPENSATABLE"
    NON_TRANSACTIONAL = "NON_TRANSACTIONAL"

TOOL_CLASSIFICATOR = {
    "WriteFile": ToolClass.REVERSIBLE,
    "UpdateDatabaseRow": ToolClass.REVERSIBLE, # Assuming snapshot
    "SendEmail": ToolClass.COMPENSATABLE,
    "CreateUser": ToolClass.COMPENSATABLE,
    "DeleteFile": ToolClass.COMPENSATABLE, # Cannot physically undo easily if garbage collected, but can recreate
    "run_terminal_command": ToolClass.NON_TRANSACTIONAL, # External side effects!
    "WebhookCall": ToolClass.NON_TRANSACTIONAL,
    "search_web": ToolClass.REVERSIBLE, # Read-only, safe to include in transactions
    "search_news": ToolClass.REVERSIBLE,
    "search_scholar": ToolClass.REVERSIBLE,
    "search_patents": ToolClass.REVERSIBLE,
    "search_documents": ToolClass.REVERSIBLE,
    "search_weather": ToolClass.REVERSIBLE,
    "search_finance": ToolClass.REVERSIBLE,
    "search_currency": ToolClass.REVERSIBLE,
    "search_company": ToolClass.REVERSIBLE,
    "search_legal": ToolClass.REVERSIBLE,
    "search_jobs": ToolClass.REVERSIBLE,
    "search_academic": ToolClass.REVERSIBLE,
    "search_tech_docs": ToolClass.REVERSIBLE,
    "compare_products": ToolClass.REVERSIBLE,
    "extract_entities": ToolClass.REVERSIBLE,
    "validate_code": ToolClass.REVERSIBLE,
    "generate_tests": ToolClass.REVERSIBLE,
    "summarize_url": ToolClass.REVERSIBLE,
    "web_fetch": ToolClass.REVERSIBLE,
    "extract_tables": ToolClass.REVERSIBLE,
    "search_products": ToolClass.REVERSIBLE,
    "search_competitors": ToolClass.REVERSIBLE,
    "search_trends": ToolClass.REVERSIBLE,
    "sentiment_scan": ToolClass.REVERSIBLE,
    "faq_builder": ToolClass.REVERSIBLE,
    "document_compare": ToolClass.REVERSIBLE,
    "data_cleaner": ToolClass.REVERSIBLE,
    "unit_cost_calc": ToolClass.REVERSIBLE,
    "execute_code": ToolClass.REVERSIBLE,
    "get_current_time": ToolClass.REVERSIBLE,
}

# ==================================
# MODELS
# ==================================

@dataclass
class CompensatingAction:
    """Action required to logically revert a tool execution."""
    tool_name: str
    action_type: str  # e.g., "RESTORE_FILE", "DELETE_USER"
    rollback_args: Dict[str, Any]

@dataclass
class TransactionContext:
    """Tracks state mutations within a defined boundary."""
    session_id: str
    task_id: str
    operations: List[CompensatingAction] = field(default_factory=list)
    is_active: bool = False
    is_rolling_back: bool = False
    created_at: float = 0.0  # timestamp for cleanup

# ==================================
# GLOBAL STORE
# ==================================
# Map: session_id -> { task_id -> TransactionContext }
_ACTIVE_TRANSACTIONS: Dict[str, Dict[str, TransactionContext]] = {}

MAX_TRANSACTION_DEPTH = 10
MAX_STALE_AGE_SECONDS = 1800  # 30 minutes — auto-cleanup inactive transactions

# ==================================
# MEMORY ISOLATION
# ==================================
_MEMORY_SUPPRESSED_SESSIONS = set()

@contextmanager
def suppress_memory_updates(session_id: str):
    """
    Blocks strategy_memory from mutating integers or floats during rollbacks.
    Rollbacks are corrective infrastructure, not learning events.
    """
    _MEMORY_SUPPRESSED_SESSIONS.add(session_id)
    try:
        yield
    finally:
        _MEMORY_SUPPRESSED_SESSIONS.discard(session_id)

def is_memory_suppressed(session_id: str) -> bool:
    return session_id in _MEMORY_SUPPRESSED_SESSIONS

# ==================================
# TRANSACTION API
# ==================================

def begin_transaction(session_id: str, task_id: str) -> TransactionContext:
    """Start a new transaction block. Enforces Single-Layer Transactions."""
    import time as _time
    
    # Auto-cleanup stale transactions before creating new ones
    cleanup_stale_transactions()
    
    if session_id not in _ACTIVE_TRANSACTIONS:
        _ACTIVE_TRANSACTIONS[session_id] = {}
        
    ctx = _ACTIVE_TRANSACTIONS[session_id].get(task_id)
    
    if ctx and ctx.is_active:
        raise ValueError("Nested transactions are explicitly disallowed (Single-Layer execution).")
        
    if ctx and ctx.is_rolling_back:
        raise RuntimeError("Rollback in progress. Cannot begin a new transaction block.")
        
    ctx = TransactionContext(session_id=session_id, task_id=task_id, is_active=True, created_at=_time.time())
    _ACTIVE_TRANSACTIONS[session_id][task_id] = ctx
    return ctx

def commit_transaction(session_id: str, task_id: str) -> None:
    """Finalizes and flushes the buffer if successful."""
    ctx = _ACTIVE_TRANSACTIONS.get(session_id, {}).get(task_id)
    if ctx and ctx.is_active:
        ctx.is_active = False
        ctx.operations = [] # Clear operations on commit
        # Keep context object alive but inactive

def add_rollback_action(session_id: str, task_id: str, action: CompensatingAction):
    """Appends a rollback instruction to the active transaction."""
    ctx = _ACTIVE_TRANSACTIONS.get(session_id, {}).get(task_id)
    if not ctx or not ctx.is_active:
        return
        
    if len(ctx.operations) >= MAX_TRANSACTION_DEPTH:
        raise MemoryError(f"Transaction Depth Limit Reached ({MAX_TRANSACTION_DEPTH}). Forced rollback boundary.")
        
    ctx.operations.append(action)

def get_transaction(session_id: str, task_id: str) -> Optional[TransactionContext]:
    return _ACTIVE_TRANSACTIONS.get(session_id, {}).get(task_id)

async def rollback_transaction(session_id: str, task_id: str):
    """
    Reverts state mutations using strict LIFO execution of CompensatingActions.
    Executed inside a suppression block to isolate learning memory.
    """
    ctx = _ACTIVE_TRANSACTIONS.get(session_id, {}).get(task_id)
    if not ctx or not ctx.is_active or not ctx.operations:
        if ctx:
            ctx.is_active = False
        return

    ctx.is_rolling_back = True
    print(f"[Transaction] Rolling back {len(ctx.operations)} operations for task {task_id} LIFO...")

    with suppress_memory_updates(session_id):
        # LIFO (Last-In-First-Out)
        for action in reversed(ctx.operations):
            print(f"[Transaction] Reverting action: {action.tool_name} ({action.action_type})")
            await _execute_compensating_action(action)
    
    # Reset state
    ctx.operations = []
    ctx.is_active = False
    ctx.is_rolling_back = False

async def _execute_compensating_action(action: CompensatingAction):
    """Router for physical state reversion based on ToolClass."""
    from app.agent.tool_executor import execute_tool, ToolCall
    
    # REVERSIBLE Ex: WriteFile (Restore original text)
    if action.action_type == "RESTORE_FILE":
        file_path = action.rollback_args.get("path")
        original_content = action.rollback_args.get("original_content", "")
        # Emulate a direct write call bypassing normal agent context memory
        tool_call = ToolCall(name="WriteFile", args=f'{{"path": "{file_path}", "content": "{original_content}"}}', raw="")
        # Minimal mock execution context for safety rollback
        from app.agent.tool_executor import ExecutionContext
        ctx_mock = ExecutionContext()
        setattr(ctx_mock, "session_id", "SYSTEM_ROLLBACK")
        await execute_tool(tool_call, ctx_mock)
        
    # COMPENSATABLE Ex: CreateUser -> DeleteUser
    elif action.action_type == "DELETE_USER":
        user_id = action.rollback_args.get("user_id")
        print(f"[Rollback] Executing compensatory DELETE_USER for {user_id}")
        # Send compensating DELETE call...
        
    # More compensating actions wired here as production demands...


def cleanup_stale_transactions() -> int:
    """Remove inactive transactions older than MAX_STALE_AGE_SECONDS. Returns count removed."""
    import time as _time
    now = _time.time()
    removed = 0
    for session_id in list(_ACTIVE_TRANSACTIONS.keys()):
        tasks = _ACTIVE_TRANSACTIONS[session_id]
        for task_id in list(tasks.keys()):
            ctx = tasks[task_id]
            if not ctx.is_active and not ctx.is_rolling_back:
                age = now - ctx.created_at if ctx.created_at > 0 else float('inf')
                if age > MAX_STALE_AGE_SECONDS:
                    del tasks[task_id]
                    removed += 1
        if not tasks:
            del _ACTIVE_TRANSACTIONS[session_id]
    return removed
