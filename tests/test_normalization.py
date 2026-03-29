import asyncio
from app.llm.processor import LLMProcessor

async def test_normalization():
    processor = LLMProcessor()
    
    messy_input = """
### Overview
This is a test.

## Overview
Duplicate overview should be skipped.

#### Detailed Architectural Insights and Analysis for Modern Systems
This heading is too long and should be trimmed.

# Technical Comparison
This single-hash heading should become double-hash.

## Technical Comparison
Duplicate comparison should be skipped.
"""
    
    print("--- TESTING PHASE 14 NORMALIZATION ---")
    normalized = processor.normalize_headings(messy_input)
    print("\nNORMALIZED OUTPUT:\n")
    print(normalized)
    
    # Validation
    lines = normalized.split("\n")
    headings = [l for l in lines if l.startswith("##")]
    
    print("\nVALIDATION:")
    print(f"- Total headings found: {len(headings)}")
    for h in headings:
        print(f"  - {h}")
        if len(h.split()) > 8: # ## + 7 words
             print(f"    [FAIL] Heading too long: {h}")
             
    # Check for expected count (Overview, Detailed Architectural Insights, Technical Comparison)
    if len(headings) == 3:
        print("\n[PASS] Heading normalization and de-duplication successful.")
    else:
        print(f"\n[FAIL] Expected 3 headings, found {len(headings)}.")

if __name__ == "__main__":
    asyncio.run(test_normalization())
