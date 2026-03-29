from __future__ import annotations

import re


def format_normal_mode_answer(text: str) -> str:
    lines = []
    skip_followups = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"(?i)^direct answer\s*:\s*$", line):
            continue
        if re.match(r"(?i)^follow-?up questions\s*:\s*$", line):
            skip_followups = True
            continue
        if skip_followups and re.match(r"^\d+\.", line):
            continue
        if re.match(r"(?i)^sources?\s*:", line):
            continue
        if re.match(r"(?i)^confidence\s*:", line):
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()

