"""
Trace Logger
Structured observability for agent execution.
"""
import json
import time
import os
from typing import Any, Dict

TRACE_DIR = "logs/traces"

def log_trace(trace_id: str, stage: str, detail: Any):
    """
    Logs a structured trace entry.
    """
    os.makedirs(TRACE_DIR, exist_ok=True)
    
    entry = {
        "timestamp": time.time(),
        "trace_id": trace_id,
        "stage": stage,
        "detail": detail
    }
    
    file_path = os.path.join(TRACE_DIR, f"{trace_id}.jsonl")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    
    print(f"[TRACE:{trace_id}] {stage} | {str(detail)[:200]}")
