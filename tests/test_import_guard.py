"""
Advanced Hardening: Import Guard Tests
Verifies both layers of import protection work correctly.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from app.sandbox.import_guard import (
    install_import_guard,
    uninstall_import_guard,
    SandboxImportError,
    _is_blocked,
    ALLOWED_MODULES,
    BLOCKED_MODULES,
)


# ============================================
# UNIT TESTS — _is_blocked logic
# ============================================

def test_blocked_modules_are_blocked():
    """All explicitly blocked modules should be detected."""
    for mod in ["subprocess", "socket", "ctypes", "multiprocessing",
                 "http", "urllib", "asyncio", "threading", "importlib",
                 "os", "shutil", "webbrowser"]:
        assert _is_blocked(mod) is True, f"{mod} should be blocked"


def test_blocked_submodules_are_blocked():
    """Sub-modules of blocked parents should be blocked."""
    for mod in ["subprocess.Popen", "http.client", "urllib.request",
                 "os.path", "ctypes.util", "socket.socket"]:
        assert _is_blocked(mod) is True, f"{mod} should be blocked"


def test_allowed_modules_pass():
    """Whitelisted modules should not be blocked."""
    for mod in ["json", "math", "datetime", "re", "pathlib",
                 "hashlib", "base64", "collections", "typing",
                 "dataclasses", "enum", "functools", "itertools",
                 "time", "copy", "string", "decimal"]:
        assert _is_blocked(mod) is False, f"{mod} should be allowed"


def test_unknown_modules_are_blocked():
    """Unknown modules should be blocked by default (whitelist approach)."""
    assert _is_blocked("some_random_module") is True
    assert _is_blocked("evil_package") is True
    assert _is_blocked("foo.bar.baz") is True


# ============================================
# INTEGRATION TESTS — Full guard install
# ============================================

def test_guard_install_and_uninstall():
    """Guard should install and uninstall cleanly."""
    install_import_guard()
    try:
        # json should still work
        import json
        assert json.dumps({"a": 1}) == '{"a": 1}'
    finally:
        uninstall_import_guard()


def test_guard_blocks_subprocess_import():
    """Guard should block import subprocess."""
    install_import_guard()
    try:
        with pytest.raises(SandboxImportError, match="subprocess"):
            __import__("subprocess")
    finally:
        uninstall_import_guard()


def test_guard_blocks_socket_import():
    """Guard should block import socket."""
    install_import_guard()
    try:
        with pytest.raises(SandboxImportError, match="socket"):
            __import__("socket")
    finally:
        uninstall_import_guard()


def test_guard_blocks_os_import():
    """Guard should block import os."""
    install_import_guard()
    try:
        with pytest.raises(SandboxImportError, match="os"):
            __import__("os")
    finally:
        uninstall_import_guard()


def test_guard_allows_json():
    """Guard should allow json import through builtins.__import__."""
    install_import_guard()
    try:
        j = __import__("json")
        assert hasattr(j, "dumps")
    finally:
        uninstall_import_guard()


def test_guard_allows_pathlib():
    """Guard should allow pathlib (replacement for os.path)."""
    install_import_guard()
    try:
        p = __import__("pathlib")
        assert hasattr(p, "PurePath")
    finally:
        uninstall_import_guard()


def test_guard_blocks_dunder_import_bypass():
    """
    builtins.__import__("subprocess") bypass should be caught.
    This is the critical test.
    """
    import builtins
    install_import_guard()
    try:
        with pytest.raises(SandboxImportError, match="subprocess"):
            builtins.__import__("subprocess")
    finally:
        uninstall_import_guard()
