"""
Relyce AI - Static Validator (Production Hardened)
Pre-execution code safety gate.

Called ONLY from processor.py — never from hybrid controller.
Validates generated code for dangerous operations using AST parsing.
Prevents dynamic execution, arbitrary file writes, and internal networking.
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional


class SecurityASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: List[Dict] = []
        # Dangerous built-ins
        self.banned_globals = {"eval", "exec", "open", "compile", "__import__"}
        # Dangerous module imports
        self.banned_modules = {"subprocess", "os", "socket", "sys", "shutil"}
        # Dangerous method calls (e.g. Path.write_text, os.system)
        self.banned_methods = {
            "system", "popen", "remove", "unlink", "rmdir", "rmtree", 
            "write_text", "write_bytes", "call", "run", "Popen",
            "check_output", "check_call", "getoutput"
        }

    def add_violation(self, name: str, description: str, lineno: int):
        self.violations.append({
            "name": name,
            "description": f"{description} at line {lineno}",
            "count": 1
        })

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module in self.banned_modules:
                self.add_violation(f"import_{base_module}", f"Imported banned module: {alias.name}", node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split('.')[0]
            if base_module in self.banned_modules:
                self.add_violation(f"import_{base_module}", f"Imported from banned module: {node.module}", node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check global function calls (e.g. eval(), open())
        if isinstance(node.func, ast.Name):
            if node.func.id in self.banned_globals:
                self.add_violation(f"call_{node.func.id}", f"Called banned global function: {node.func.id}", node.lineno)
            if node.func.id == "__import__":
                self.add_violation("call___import__", "Called banned __import__ function", node.lineno)
            if node.func.id == "getattr":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    if node.args[1].value in self.banned_globals:
                         self.add_violation("call_getattr_banned", f"Called getattr with banned global: {node.args[1].value}", node.lineno)
        
        # Check method calls (e.g. os.system(), x.write_text())
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.banned_methods:
                self.add_violation(f"call_method_{node.func.attr}", f"Called banned method: {node.func.attr}", node.lineno)

        # Check for private IP strings in arguments (basic heuristic within AST)
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                val = arg.value
                if any(ip in val for ip in ["127.0.0.1", "localhost", "0.0.0.0", "192.168.", "10."]):
                    self.add_violation("network_internal", f"Detected internal IP/host in arguments: {val}", node.lineno)

        self.generic_visit(node)


def extract_python_blocks(text: str) -> str:
    """Extracts Python code from Markdown code blocks if present."""
    if "```python" in text or "```" in text:
        import re
        blocks = re.findall(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
        if blocks:
            return "\n".join(blocks)
    return text


def get_dangerous_ops(code: str) -> List[Dict]:
    """
    Parse code into an AST and walk it to identify dangerous operations.
    Returns list of detected dangerous patterns with details.
    """
    if not code or not code.strip():
        return []

    # Try to extract code blocks, otherwise treat entire string as code
    clean_code = extract_python_blocks(code)

    try:
        tree = ast.parse(clean_code)
        visitor = SecurityASTVisitor()
        visitor.visit(tree)
        return visitor.violations
    except SyntaxError:
        # If it doesn't parse, we can't reliably AST-validate it, 
        # but execution will fail natively anyway.
        # We flag it as unsafe to let the repair engine handle the SyntaxError safely.
        return [{"name": "syntax_error", "description": "Code could not be parsed into AST for validation", "count": 1}]
    except Exception as e:
        return [{"name": "validation_error", "description": f"AST validation failed: {str(e)}", "count": 1}]


def contains_dangerous_ops(code: str) -> bool:
    """
    Check if code contains any dangerous operations via AST parsing.
    Returns True if any dangerous pattern is found.
    """
    return len(get_dangerous_ops(code)) > 0


def static_validation(context: Dict) -> str:
    """
    Validate generated code before execution securely via AST.

    Called ONLY from processor.py.

    Args:
        context: Must contain "generated_code" key.

    Returns:
        "validation_passed"  — code is safe to execute
        "validation_failed"  — no code found
        "unsafe_code"        — dangerous operations detected (sets context["risk_flag"])
    """
    code = context.get("generated_code", "")

    if not code or not code.strip():
        return "validation_failed"

    ops = get_dangerous_ops(code)
    if ops:
        context["risk_flag"] = True
        context["dangerous_ops"] = ops
        return "unsafe_code"

    return "validation_passed"
