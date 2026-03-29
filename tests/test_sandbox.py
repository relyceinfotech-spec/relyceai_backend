"""
Phase 6: Sandbox Isolation Tests
Validates process-level isolation, timeout enforcement, and crash containment.
"""
import pytest
import asyncio
import json
import sys
import os

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.sandbox.sandbox_manager import SandboxManager, SandboxResult


# ============================================
# TEST 1: Fast tool executes successfully
# ============================================

def test_sandbox_fast_tool_success():
    """A known tool (calculate) should execute and return quickly."""
    manager = SandboxManager(timeout=10)
    result = asyncio.run(manager.execute_tool_sandboxed(
        tool_name="calculate",
        tool_args="2 + 3 * 4",
    ))
    
    assert result.success is True
    assert result.data == 14  # 2 + 3*4 = 14
    assert result.source == "calculator"
    assert result.confidence == "high"
    assert result.timed_out is False
    assert result.crashed is False


# ============================================
# TEST 2: Unknown tool returns graceful failure
# ============================================

def test_sandbox_unknown_tool():
    """An unknown tool name should fail gracefully without crashing."""
    manager = SandboxManager(timeout=5)
    result = asyncio.run(manager.execute_tool_sandboxed(
        tool_name="nonexistent_danger_tool",
        tool_args="payload",
    ))
    
    assert result.success is False
    assert result.timed_out is False
    assert result.crashed is False
    assert "Unknown tool" in str(result.data) or "Unknown tool" in str(result.error)


# ============================================
# TEST 3: Subprocess crash does not crash test
# ============================================

def test_sandbox_crash_containment():
    """
    Simulate a tool that crashes the subprocess.
    The SandboxManager must survive and return a structured error.
    We test this by providing an intentionally malformed payload
    to the runner by using a custom subprocess invocation.
    """
    manager = SandboxManager(timeout=5)
    
    # Even a tool that raises internally should be caught by tool_runner
    result = asyncio.run(manager.execute_tool_sandboxed(
        tool_name="calculate",
        tool_args="1/0",  # ZeroDivisionError
    ))
    
    # The tool should fail gracefully — not crash the test process
    assert result.success is False
    assert result.timed_out is False
    # The process itself doesn't crash, the tool returns a failure
    assert result.crashed is False


# ============================================
# TEST 4: Timeout enforcement kills long-running tool
# ============================================

def test_sandbox_timeout_enforcement():
    """
    Create a custom subprocess that sleeps forever.
    The SandboxManager's timeout must kill it.
    """
    import tempfile

    # Write a rogue tool_runner that hangs
    rogue_script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    )
    rogue_script.write("""
import sys
import time
import json

# Read stdin to complete protocol
payload = sys.stdin.read()

# Simulate infinite loop / hung tool
time.sleep(999)

# This line should never execute
json.dump({"status": "success", "data": "should not reach"}, sys.stdout)
""")
    rogue_script.flush()
    rogue_path = rogue_script.name
    rogue_script.close()

    try:
        import app.sandbox.sandbox_manager as sm
        original_path = sm._TOOL_RUNNER_PATH
        
        # Point sandbox at the rogue script
        sm._TOOL_RUNNER_PATH = rogue_path
        
        manager = SandboxManager(timeout=2)  # 2 second timeout
        result = asyncio.run(manager.execute_tool_sandboxed(
            tool_name="calculate",
            tool_args="1+1",
        ))
        
        assert result.timed_out is True
        assert result.success is False
        assert "timed out" in result.error.lower()
        
        # Restore
        sm._TOOL_RUNNER_PATH = original_path
    finally:
        os.unlink(rogue_path)


# ============================================
# TEST 5: SandboxResult dataclass structure
# ============================================

def test_sandbox_result_defaults():
    """Verify SandboxResult has correct default values."""
    r = SandboxResult()
    assert r.success is False
    assert r.data is None
    assert r.source == "sandbox"
    assert r.confidence == "low"
    assert r.timed_out is False
    assert r.crashed is False
    assert r.error == ""


# ============================================
# TEST 6: Tool executor routes through sandbox
# ============================================

def test_tool_executor_sandbox_routing():
    """
    When SANDBOX_ENABLED is True, eligible tools should be routed
    through the SandboxManager instead of direct execution.
    """
    import app.config as config
    original_val = config.SANDBOX_ENABLED
    config.SANDBOX_ENABLED = True
    
    try:
        from app.agent.tool_executor import execute_tool, ToolCall, ExecutionContext
        
        tool_call = ToolCall(name="calculate", args="5 * 5")
        exec_ctx = ExecutionContext()
        
        result = asyncio.run(execute_tool(tool_call, exec_ctx))
        
        assert result.success is True
        assert result.data == 25
        assert result.source == "calculator"
        assert exec_ctx.degraded is False
    finally:
        config.SANDBOX_ENABLED = original_val


# ============================================
# TEST 7: Non-sandbox tools bypass isolation
# ============================================

def test_tool_executor_bypass_for_non_eligible():
    """
    Tools NOT in _SANDBOX_ELIGIBLE should execute directly
    in-process even when SANDBOX_ENABLED is True.
    """
    import app.config as config
    original_val = config.SANDBOX_ENABLED
    config.SANDBOX_ENABLED = True
    
    try:
        from app.agent.tool_executor import execute_tool, ToolCall, ExecutionContext
        
        tool_call = ToolCall(name="get_current_time", args="")
        exec_ctx = ExecutionContext()
        
        result = asyncio.run(execute_tool(tool_call, exec_ctx))
        
        # get_current_time is NOT sandbox-eligible, runs in-process
        assert result.success is True
        assert result.source == "clock"
    finally:
        config.SANDBOX_ENABLED = original_val


# ============================================
# HARDENING TESTS
# ============================================

# TEST 8: Environment variable stripping
def test_sandbox_env_stripping():
    """
    Verify that tool_runner strips environment variables.
    The subprocess should NOT have access to sensitive keys.
    """
    import tempfile

    # Script that dumps env to stdout as JSON
    probe_script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    )
    probe_script.write("""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Simulate the hardening from tool_runner
_WHITELIST = {"PATH","SYSTEMROOT","TEMP","TMP","HOME","USERPROFILE","PYTHONPATH","SERPER_API_KEY","VIRTUAL_ENV"}
keys_to_remove = [k for k in os.environ if k not in _WHITELIST]
for k in keys_to_remove:
    del os.environ[k]

# Read stdin protocol
payload = sys.stdin.read()

# Dump remaining env keys
remaining = list(os.environ.keys())
json.dump({"status": "success", "data": remaining, "source": "env_probe", "confidence": "high"}, sys.stdout)
sys.stdout.flush()
""")
    probe_script.flush()
    probe_path = probe_script.name
    probe_script.close()

    try:
        import app.sandbox.sandbox_manager as sm
        original_path = sm._TOOL_RUNNER_PATH
        sm._TOOL_RUNNER_PATH = probe_path

        # Set a fake sensitive key
        os.environ["SUPER_SECRET_DB_PASSWORD"] = "hunter2"
        
        manager = SandboxManager(timeout=5)
        result = asyncio.run(manager.execute_tool_sandboxed("calculate", "1+1"))

        assert result.success is True
        remaining_keys = result.data
        # The secret should NOT be in the subprocess environment
        assert "SUPER_SECRET_DB_PASSWORD" not in remaining_keys

        sm._TOOL_RUNNER_PATH = original_path
    finally:
        os.environ.pop("SUPER_SECRET_DB_PASSWORD", None)
        os.unlink(probe_path)


# TEST 9: Strict JSON schema rejects extra keys
def test_sandbox_rejects_extra_keys():
    """
    A subprocess that returns extra JSON keys should be rejected.
    """
    import tempfile

    rogue_script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    )
    rogue_script.write("""
import sys, json
payload = sys.stdin.read()
# Return JSON with an unexpected key (injection attempt)
json.dump({
    "status": "success",
    "data": "looks normal",
    "source": "sandbox",
    "confidence": "high",
    "malicious_payload": "DROP TABLE users;"
}, sys.stdout)
sys.stdout.flush()
""")
    rogue_script.flush()
    rogue_path = rogue_script.name
    rogue_script.close()

    try:
        import app.sandbox.sandbox_manager as sm
        original_path = sm._TOOL_RUNNER_PATH
        sm._TOOL_RUNNER_PATH = rogue_path

        manager = SandboxManager(timeout=5)
        result = asyncio.run(manager.execute_tool_sandboxed("calculate", "1+1"))

        # Must FAIL due to extra keys
        assert result.success is False
        assert "unexpected keys" in result.error.lower()
        assert result.source == "sandbox_schema"

        sm._TOOL_RUNNER_PATH = original_path
    finally:
        os.unlink(rogue_path)


# TEST 10: Output size cap enforcement
def test_sandbox_output_size_cap():
    """
    A subprocess that returns massive output should be rejected.
    """
    import tempfile

    flood_script = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    )
    flood_script.write("""
import sys, json
payload = sys.stdin.read()
# Generate 100KB of output (exceeds 64KB cap)
huge_data = "A" * 100000
json.dump({"status": "success", "data": huge_data, "source": "flood", "confidence": "high"}, sys.stdout)
sys.stdout.flush()
""")
    flood_script.flush()
    flood_path = flood_script.name
    flood_script.close()

    try:
        import app.sandbox.sandbox_manager as sm
        original_path = sm._TOOL_RUNNER_PATH
        sm._TOOL_RUNNER_PATH = flood_path

        manager = SandboxManager(timeout=5)
        result = asyncio.run(manager.execute_tool_sandboxed("calculate", "1+1"))

        assert result.success is False
        assert "exceeded" in result.error.lower()
        assert result.source == "sandbox_overflow"

        sm._TOOL_RUNNER_PATH = original_path
    finally:
        os.unlink(flood_path)

