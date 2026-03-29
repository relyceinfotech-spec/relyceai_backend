import os
import sys

# Mock path to import prompts
sys.path.append(os.getcwd() + "/app/llm")
from app.llm import prompts

def check_tokens():
    print("--- PROMPT TOKEN EFFICIENCY CHECK ---")
    
    cases = [
        ("NORMAL_SYSTEM_PROMPT", prompts.NORMAL_SYSTEM_PROMPT),
        ("BUSINESS_SYSTEM_PROMPT", prompts.BUSINESS_SYSTEM_PROMPT),
        ("AGENT_FORMAT_PROMPT", prompts.AGENT_FORMAT_PROMPT)
    ]
    
    for name, content in cases:
        char_count = len(content)
        # Rough estimate: 4 chars per token
        est_tokens = char_count // 4
        print(f"\n{name}:")
        print(f"  - Characters: {char_count}")
        print(f"  - Est. Tokens: ~{est_tokens}")
        if est_tokens > 600:
            print(f"  [WARNING] Above target (600 tokens)")
        else:
            print(f"  [PASS] Within efficient range")

if __name__ == "__main__":
    check_tokens()
