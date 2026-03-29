import json
from pathlib import Path

from app.security.safety_filter import detect_injection


def test_redteam_prompt_regression() -> None:
    dataset_path = Path(__file__).parent / "security" / "redteam_prompts.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8-sig"))

    mismatches = []
    for item in data:
        got = detect_injection(item["prompt"])
        expected = bool(item["expected_blocked"])
        if got != expected:
            mismatches.append({"id": item["id"], "expected": expected, "got": got})

    assert not mismatches, f"Red-team regression mismatches: {mismatches}"