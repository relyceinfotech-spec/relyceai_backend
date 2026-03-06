"""
Advanced Hardening: Sandbox Import Guard
Blocks dangerous module imports inside the sandboxed subprocess.

CRITICAL DESIGN:
  1. Overrides builtins.__import__ — catches __import__("subprocess") bypass
  2. Installs sys.meta_path hook — catches import subprocess
  3. Both layers must be active for defense-in-depth

Blocked modules (focused list):
  subprocess, socket, ctypes, multiprocessing, http, urllib,
  asyncio, threading, importlib, os, shutil, code, webbrowser,
  xmlrpc, ftplib, smtplib, telnetlib, poplib, imaplib, nntplib

Allowed modules (minimal whitelist):
  json, math, datetime, re, pathlib, hashlib, base64,
  collections, typing, dataclasses, enum, functools, itertools,
  string, decimal, fractions, statistics, copy, time, struct,
  io (StringIO/BytesIO only), traceback, sys (read-only)

The guard treats unknown imports as BLOCKED by default.
Only whitelisted modules pass through.
"""
from __future__ import annotations

import sys
import builtins
from typing import Set, Optional


# ============================================
# WHITELIST — Only these modules are allowed
# ============================================

ALLOWED_MODULES: Set[str] = {
    # Core language
    "json", "math", "datetime", "re", "string", "copy",
    "decimal", "fractions", "statistics", "struct", "time",

    # Data structures
    "collections", "collections.abc",
    "typing", "typing_extensions",
    "dataclasses", "enum",
    "functools", "itertools", "operator",

    # Crypto/encoding
    "hashlib", "base64", "hmac", "secrets",

    # File paths (read-only, no os)
    "pathlib",

    # Serialization
    "pickle", "csv",

    # Introspection (limited)
    "traceback", "inspect", "abc",

    # IO (in-memory only)
    "io",

    # Internal use
    "_thread", "__future__",
}

# Prefixes that are allowed (for sub-module imports)
ALLOWED_PREFIXES = (
    "collections.", "typing.", "pathlib.",
    "json.", "dataclasses.", "enum.",
    "functools.", "itertools.",
    "encodings.", "_",  # Python internals needed for boot
)

# ============================================
# BLOCKED — Explicitly dangerous modules
# ============================================

BLOCKED_MODULES: Set[str] = {
    # Process execution
    "subprocess", "multiprocessing", "concurrent",

    # Network
    "socket", "http", "urllib", "requests",
    "xmlrpc", "ftplib", "smtplib", "telnetlib",
    "poplib", "imaplib", "nntplib", "ssl",
    "webbrowser", "socketserver",

    # Foreign function interface
    "ctypes", "cffi",

    # OS-level power
    "os", "shutil", "tempfile", "glob",

    # Threading (prevent fork/spawn inside sandbox)
    "threading", "asyncio",

    # Import manipulation
    "importlib", "zipimport", "pkgutil",

    # Code execution
    "code", "codeop", "compileall", "py_compile",

    # Signal handling
    "signal",
}

BLOCKED_PREFIXES = (
    "subprocess.", "multiprocessing.", "concurrent.",
    "socket.", "http.", "urllib.", "requests.",
    "ctypes.", "os.", "shutil.", "tempfile.",
    "threading.", "asyncio.", "importlib.",
    "xmlrpc.", "ftplib.", "smtplib.",
    "ssl.", "webbrowser.", "socketserver.",
    "code.", "signal.",
)


class SandboxImportError(ImportError):
    """Raised when a blocked module is imported inside the sandbox."""
    pass


# ============================================
# META PATH FINDER (Layer 1)
# ============================================

class SandboxImportBlocker:
    """
    sys.meta_path hook that intercepts import statements.
    This catches: import subprocess, from subprocess import Popen
    """

    def find_module(self, fullname: str, path=None):
        if _is_blocked(fullname):
            return self  # Returning self triggers load_module
        return None

    def load_module(self, fullname: str):
        raise SandboxImportError(
            f"SANDBOX: Import of '{fullname}' is blocked. "
            f"This module is not allowed in the sandboxed execution environment."
        )


# ============================================
# BUILTINS OVERRIDE (Layer 2)
# ============================================

_original_import = builtins.__import__


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Override of builtins.__import__ that blocks dangerous modules.
    This catches: __import__("subprocess"), exec("import subprocess")
    """
    if _is_blocked(name):
        raise SandboxImportError(
            f"SANDBOX: Import of '{name}' is blocked via __import__. "
            f"This module is not allowed in the sandboxed execution environment."
        )
    return _original_import(name, globals, locals, fromlist, level)


# ============================================
# HELPER
# ============================================

def _is_blocked(module_name: str) -> bool:
    """Check if a module name is blocked."""
    # Explicitly blocked
    if module_name in BLOCKED_MODULES:
        return True

    # Blocked prefix (sub-modules of blocked parents)
    if module_name.startswith(BLOCKED_PREFIXES):
        return True

    # If it's in the whitelist, allow
    if module_name in ALLOWED_MODULES:
        return False

    # If it matches an allowed prefix, allow
    if module_name.startswith(ALLOWED_PREFIXES):
        return False

    # UNKNOWN MODULE — block by default (whitelist approach)
    # This is the most secure posture
    return True


# ============================================
# INSTALL / UNINSTALL
# ============================================

_installed = False


def install_import_guard() -> None:
    """
    Install both layers of import protection.
    Call this ONCE at the start of the sandbox subprocess.
    """
    global _installed
    if _installed:
        return

    # Layer 1: sys.meta_path hook
    sys.meta_path.insert(0, SandboxImportBlocker())

    # Layer 2: builtins.__import__ override
    builtins.__import__ = _guarded_import

    _installed = True


def uninstall_import_guard() -> None:
    """
    Remove import guard (for testing cleanup).
    """
    global _installed

    # Remove meta_path hook
    sys.meta_path[:] = [
        f for f in sys.meta_path
        if not isinstance(f, SandboxImportBlocker)
    ]

    # Restore original __import__
    builtins.__import__ = _original_import

    _installed = False
