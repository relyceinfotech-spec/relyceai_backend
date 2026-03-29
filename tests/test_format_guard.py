import asyncio
import re
import sys
import os

# Mock the parts of processor needed for testing
class MockProcessor:
    def is_structured_text(self, resp: str) -> bool:
        if not resp: return False
        # Phase 11 Logic: Check for headings AND (bullets OR tables)
        has_headings = bool(re.search(r"^##\s", resp, re.MULTILINE))
        has_structure = "-" in resp or "|" in resp or "*" in resp
        return has_headings and has_structure

    async def test_guard(self):
        cases = [
            {
                "name": "Good Response",
                "text": "## Overview\n- Item 1\n- Item 2",
                "expected": True
            },
            {
                "name": "No Headings",
                "text": "Item 1\nItem 2",
                "expected": False
            },
            {
                "name": "Headings but No Bullets",
                "text": "## Overview\nThis is just a paragraph without bullets.",
                "expected": False
            },
            {
                "name": "Empty",
                "text": "",
                "expected": False
            },
            {
                "name": "Table Response",
                "text": "## Comparison\n| A | B |\n|---|---|",
                "expected": True
            }
        ]

        print("--- FORMAT GUARD TEST ---")
        for case in cases:
            result = self.is_structured_text(case["text"])
            status = "PASS" if result == case["expected"] else "FAIL"
            print(f"[{status}] {case['name']}: Result={result}, Expected={case['expected']}")

if __name__ == "__main__":
    p = MockProcessor()
    asyncio.run(p.test_guard())
