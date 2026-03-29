from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"

PATTERNS = [
    (re.compile(r"detail\s*=\s*str\(e\)"), "Raw exception returned in HTTP response"),
    (re.compile(r"content\s*[:=]\s*str\(e\)"), "Raw exception returned in stream payload"),
    (re.compile(r"authorization\s*[:=]\s*['\"].+['\"]", re.IGNORECASE), "Possible hardcoded authorization"),
    (re.compile(r"api[_-]?key\s*=\s*['\"][^'\"]{8,}['\"]", re.IGNORECASE), "Possible hardcoded API key"),
]

SKIP_DIRS = {"__pycache__", ".venv", "venv", "tests"}

def iter_py_files(base: pathlib.Path):
    for p in base.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p

def main() -> int:
    failures = []
    for path in iter_py_files(APP):
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            for rgx, msg in PATTERNS:
                if rgx.search(line):
                    failures.append(f"{path}:{i}: {msg}: {line.strip()[:160]}")
    if failures:
        print("[security-lint] FAIL")
        for f in failures:
            print(f)
        return 1
    print("[security-lint] PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
