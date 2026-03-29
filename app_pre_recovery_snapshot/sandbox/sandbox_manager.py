"""
Phase 6: Sandbox Manager
Spawn and manage isolated tool execution environments.

Architecture:
    Processor → SandboxManager → Isolated Subprocess → Tool Execution

The SandboxManager:
  - Spawns a Python subprocess running tool_runner.py
  - Passes tool payload via stdin (JSON)
  - Captures stdout/stderr
  - Enforces timeout at the process level (kills on expiry)
  - Returns a structured ToolResult

Security guarantees:
  - Tool code runs in a SEPARATE process
  - If the tool crashes, hangs, or leaks memory, the core engine survives
  - No shared memory between orchestrator and tool runtime
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import SANDBOX_TIMEOUT, SANDBOX_MEMORY_LIMIT_MB, SANDBOX_CPU_SECONDS
from app.observability.event_types import EventType
from app.observability.event_logger import get_event_logger
from app.observability.metrics_collector import get_metrics_collector

# Path to the isolated tool runner script
_TOOL_RUNNER_PATH = os.path.join(os.path.dirname(__file__), "tool_runner.py")


@dataclass
class SandboxResult:
    """Structured result from a sandboxed tool execution."""
    success: bool = False
    data: Any = None
    source: str = "sandbox"
    confidence: str = "low"
    trust: str = "verified"
    error: str = ""
    timed_out: bool = False
    crashed: bool = False


class SandboxManager:
    """
    Manages subprocess-isolated tool execution.

    Usage:
        manager = SandboxManager()
        result = await manager.execute_tool_sandboxed("search_web", "python DAG")
    """

    def __init__(
        self,
        timeout: int = SANDBOX_TIMEOUT,
        memory_limit_mb: int = SANDBOX_MEMORY_LIMIT_MB,
        cpu_seconds: int = SANDBOX_CPU_SECONDS,
    ):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.cpu_seconds = cpu_seconds

    async def execute_tool_sandboxed(
        self,
        tool_name: str,
        tool_args: str,
        session_id: str = "",
    ) -> SandboxResult:
        """
        Execute a tool inside an isolated subprocess.

        1. Serialize payload to JSON
        2. Spawn subprocess running tool_runner.py
        3. Feed payload via stdin
        4. Wait for result (enforce timeout)
        5. Parse JSON output
        6. Return SandboxResult
        """
        payload = json.dumps({
            "tool": tool_name,
            "args": tool_args,
            "session_id": session_id,
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_limit_mb,
        })

        result = SandboxResult()
        _logger = get_event_logger()
        _metrics = get_metrics_collector()
        _metrics.record_sandbox_execution()
        _logger.emit(EventType.SANDBOX_STARTED, {"tool": tool_name})

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, _TOOL_RUNNER_PATH,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=payload.encode("utf-8")),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                # Kill the subprocess if it exceeds the timeout
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                result.timed_out = True
                result.error = f"Sandbox execution timed out after {self.timeout}s"
                result.source = "sandbox_timeout"
                _metrics.record_sandbox_kill()
                _logger.emit(EventType.SANDBOX_KILLED, {"tool": tool_name, "timeout": self.timeout})
                print(f"[Sandbox] TIMEOUT: {tool_name} killed after {self.timeout}s")
                return result

            # Parse subprocess output
            if proc.returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                result.crashed = True
                result.error = f"Sandbox process exited with code {proc.returncode}: {stderr_text[:500]}"
                result.source = "sandbox_crash"
                _metrics.record_sandbox_crash()
                _logger.emit(EventType.SANDBOX_CRASHED, {"tool": tool_name, "exit_code": proc.returncode})
                print(f"[Sandbox] CRASH: {tool_name} exit={proc.returncode}")
                return result

            # Decode and parse JSON output
            stdout_text = stdout.decode("utf-8", errors="replace").strip()

            # HARDENING: Cap output size to prevent memory bomb from rogue subprocess
            MAX_SANDBOX_OUTPUT = 65536  # 64KB
            if len(stdout_text) > MAX_SANDBOX_OUTPUT:
                result.error = f"Sandbox output exceeded {MAX_SANDBOX_OUTPUT} bytes — rejected"
                result.source = "sandbox_overflow"
                return result

            if not stdout_text:
                result.error = "Sandbox returned empty output"
                result.source = "sandbox_empty"
                return result

            try:
                output = json.loads(stdout_text)
            except json.JSONDecodeError as e:
                result.error = f"Sandbox returned invalid JSON: {str(e)}"
                result.source = "sandbox_parse"
                return result

            # HARDENING: Strict output schema validation
            # Treat subprocess output as UNTRUSTED — validate structure
            if not isinstance(output, dict):
                result.error = "Sandbox output is not a JSON object"
                result.source = "sandbox_schema"
                return result

            _ALLOWED_KEYS = {"status", "data", "source", "confidence", "trust", "error"}
            extra_keys = set(output.keys()) - _ALLOWED_KEYS
            if extra_keys:
                result.error = f"Sandbox output contains unexpected keys: {extra_keys}"
                result.source = "sandbox_schema"
                return result

            # Validate field types
            status_val = output.get("status")
            if status_val not in ("success", "failure", None):
                result.error = f"Sandbox returned invalid status: {status_val}"
                result.source = "sandbox_schema"
                return result

            confidence_val = output.get("confidence", "low")
            if confidence_val not in ("high", "medium", "low"):
                confidence_val = "low"  # Clamp to safe default

            # Map validated output to SandboxResult
            result.success = status_val == "success"
            result.data = output.get("data")
            result.source = output.get("source", "sandbox")
            result.confidence = confidence_val
            result.trust = output.get("trust", "verified")
            if not result.success:
                result.error = output.get("error", output.get("data", "Unknown failure"))

            # --- TELEMETRY: Sandbox Completed ---
            _logger.emit(EventType.SANDBOX_COMPLETED, {
                "tool": tool_name,
                "success": result.success,
                "source": result.source,
            })

            return result

        except Exception as e:
            result.crashed = True
            result.error = f"Sandbox manager error: {str(e)}"
            result.source = "sandbox_manager"
            print(f"[Sandbox] ERROR: {tool_name} → {str(e)}")
            return result


# Module-level singleton for convenience
_default_manager: Optional[SandboxManager] = None


def get_sandbox_manager() -> SandboxManager:
    """Returns the module-level SandboxManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SandboxManager()
    return _default_manager
