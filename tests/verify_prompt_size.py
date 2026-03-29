
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tiktoken
from app.llm.prompts import NORMAL_SYSTEM_PROMPT, INTERNAL_SYSTEM_PROMPT

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

if __name__ == "__main__":
    normal_tokens = count_tokens(NORMAL_SYSTEM_PROMPT)
    internal_tokens = count_tokens(INTERNAL_SYSTEM_PROMPT)
    
    print("-" * 30)
    print(f"NORMAL_SYSTEM_PROMPT: {normal_tokens} tokens")
    print(f"INTERNAL_SYSTEM_PROMPT: {internal_tokens} tokens")
    print("-" * 30)
    
    if 350 <= normal_tokens <= 600:
        print("?? SUCCESS: NORMAL_SYSTEM_PROMPT is within target range (350-600).")
    else:
        print("?? WARNING: NORMAL_SYSTEM_PROMPT is outside target range.")
