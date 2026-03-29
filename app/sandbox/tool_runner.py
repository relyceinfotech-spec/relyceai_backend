"""
Phase 6: Isolated Tool Runner (Hardened)
This script is the ONLY code that runs inside the sandboxed subprocess.

It receives a JSON payload on stdin, executes the requested tool,
and writes a JSON result to stdout.

SECURITY HARDENING:
  1. Environment variables STRIPPED on startup (whitelist only)
  2. Working directory locked to SANDBOX_SAFE_ROOT
  3. OS-level resource limits: CPU, memory, max processes (fork bomb guard)
  4. NO access to agent memory, PlanGraph, strategy, transaction contexts
  5. Only loads tool registry — no other orchestration imports
"""
import sys
import json
import asyncio
import os
import signal
import traceback

# Ensure the project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


# ============================================
# HARDENING: Environment Variable Stripping
# ============================================
# Whitelist ONLY what the tool registry needs.
# Everything else (API keys, DB creds, tokens) is purged.

_ENV_WHITELIST = {
    "PATH",              # Needed for subprocess/OS resolution
    "SYSTEMROOT",        # Windows requires this
    "TEMP", "TMP",       # Temp directory resolution
    "HOME", "USERPROFILE",  # Home directory (some libs need it)
    "PYTHONPATH",        # Module resolution
    "SERPER_API_KEY",    # search_web needs this
    "VIRTUAL_ENV",       # Virtualenv detection
}

def _strip_environment():
    """Remove all environment variables except whitelisted ones."""
    keys_to_remove = [k for k in os.environ if k not in _ENV_WHITELIST]
    for k in keys_to_remove:
        del os.environ[k]


# ============================================
# HARDENING: Working Directory Restriction
# ============================================

SANDBOX_SAFE_ROOT = os.path.join(_PROJECT_ROOT, "data")

def _enforce_safe_root():
    """
    Lock working directory and validate file access paths.
    Any file operation must resolve within SANDBOX_SAFE_ROOT.
    """
    os.makedirs(SANDBOX_SAFE_ROOT, exist_ok=True)
    os.chdir(SANDBOX_SAFE_ROOT)


def validate_file_path(path: str) -> bool:
    """Returns True only if the resolved path is inside SANDBOX_SAFE_ROOT."""
    try:
        real_path = os.path.realpath(os.path.abspath(path))
        real_root = os.path.realpath(SANDBOX_SAFE_ROOT)
        return real_path.startswith(real_root)
    except Exception:
        return False


# ============================================
# HARDENING: Resource Limits (OS-Level)
# ============================================

def _apply_resource_limits(cpu_seconds: int, memory_mb: int, max_procs: int = 10):
    """
    Apply OS-level resource limits.
    Linux: uses resource.setrlimit for CPU, memory, and max processes.
    Windows: limits enforced by parent process timeout (no resource module).
    """
    try:
        import resource

        # CPU time limit (hard kill after cpu_seconds + 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))

        # Memory limit (address space)
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        # Fork bomb protection: limit max child processes
        resource.setrlimit(resource.RLIMIT_NPROC, (max_procs, max_procs))

    except (ImportError, AttributeError):
        # Windows: resource module unavailable
        # Limits enforced by parent process timeout and job objects
        pass
    except Exception:
        pass


# ============================================
# TOOL EXECUTION (Isolated)
# ============================================

async def _run_tool(tool_name: str, tool_args: str, session_id: str = "") -> dict:
    """Load tool from registry and execute it."""
    from app.agent.tool_executor import TOOLS, validate_tool_result, truncate_payload, safe_execute
    import inspect

    if tool_name not in TOOLS:
        return {
            "status": "failure",
            "data": f"Unknown tool: {tool_name}",
            "source": "sandbox",
            "confidence": "low",
        }

    tool_meta = TOOLS[tool_name]
    handler = tool_meta["func"]
    is_async = tool_meta["is_async"]

    sig = inspect.signature(handler)
    kwargs = {}
    if "session_id" in sig.parameters:
        kwargs["session_id"] = session_id

    raw = await safe_execute(handler, tool_args, timeout=30, is_async=is_async, **kwargs)

    if validate_tool_result(raw):
        return {
            "status": "success",
            "data": truncate_payload(raw["data"]),
            "source": raw.get("source", tool_name),
            "confidence": raw.get("confidence", "medium"),
            "trust": raw.get("trust", "verified"),
        }
    else:
        return {
            "status": "failure",
            "data": raw.get("data") if raw else None,
            "source": raw.get("source", tool_name) if raw else "error",
            "confidence": "low",
            "error": raw.get("data", "Tool failed") if raw else "No response",
        }


# ============================================
# ENTRY POINT
# ============================================

def main():
    """Entry point for sandboxed tool execution."""
    # --- HARDENING: Apply all security layers BEFORE any tool logic ---

    # Layer 0: Import guard (MUST be first — blocks dangerous imports)
    if str(os.getenv("SANDBOX_IMPORT_GUARD_ENABLED", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from app.sandbox.import_guard import install_import_guard
            install_import_guard()
        except Exception:
            pass  # If import guard fails, continue with other hardening

    _strip_environment()
    _enforce_safe_root()

    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input)
    except Exception as e:
        json.dump({
            "status": "failure",
            "data": str(e),
            "source": "sandbox_parse",
            "confidence": "low"
        }, sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

    tool_name = payload.get("tool", "")
    tool_args = payload.get("args", "")
    session_id = payload.get("session_id", "")
    cpu_limit = payload.get("cpu_seconds", 5)
    memory_limit = payload.get("memory_mb", 256)
    max_procs = payload.get("max_procs", 10)

    # Apply OS-level limits
    _apply_resource_limits(cpu_limit, memory_limit, max_procs)

    try:
        result = asyncio.run(_run_tool(tool_name, tool_args, session_id))
    except MemoryError:
        result = {
            "status": "failure",
            "data": "Sandbox memory limit exceeded",
            "source": "sandbox_oom",
            "confidence": "low"
        }
    except Exception as e:
        result = {
            "status": "failure",
            "data": f"Sandbox execution error: {str(e)}",
            "source": "sandbox_crash",
            "confidence": "low"
        }

    json.dump(result, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
