
import asyncio
import sys
import os
import re

# Add backend to path
sys.path.append(os.path.join(os.getcwd()))

from app.llm.router import analyze_and_route_query, get_system_prompt_for_mode

async def test_adaptive_prompts():
    print("--- ADAPTIVE PROMPT GENERATION TEST ---")
    q = "compare GPU vs CPU"
    prompt = get_system_prompt_for_mode("normal", user_query=q)
    
    # Check for keywords in the prompt to ensure injection
    keywords = ["ADAPTIVE RESPONSE STYLE — NORMAL MODE", "ADAPTIVE STRUCTURE PLANNING"]
    for k in keywords:
        found = k in prompt
        print(f"  Contains '{k}': {found}")
    
    if "ADAPTIVE RESPONSE STYLE" in prompt and "ADAPTIVE STRUCTURE PLANNING" in prompt:
        print("  SUCCESS: Prompt contains adaptive style and planning blocks.")
    else:
        print("  FAILURE: Prompt missing critical adaptive blocks.")

async def test_validator():
    print("\n--- VALIDATOR TEST ---")
    from app.llm.processor import llm_processor
    
    good_adaptive = "## Overview\nTest\n## Key Differences\nTest"
    bad_adaptive = "No headers here"
    
    is_good = llm_processor.is_structured_text(good_adaptive)
    is_bad = llm_processor.is_structured_text(bad_adaptive)
    
    print(f"  Good text (2 headers) valid: {is_good}")
    print(f"  Bad text (0 headers) valid: {is_bad}")
    
    if is_good and not is_bad:
        print("  SUCCESS: Validator correctly handles adaptive structure.")
    else:
        print("  FAILURE: Validator logic incorrect.")

if __name__ == "__main__":
    asyncio.run(test_adaptive_prompts())
    asyncio.run(test_validator())
