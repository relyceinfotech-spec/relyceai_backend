"""Quick test for SmartMemory classifier"""
from app.chat.smart_memory import classify_memory

tests = [
    ("My name is Tamizh", True),
    ("lol", False),
    ("I am building a RAG chatbot", True),
    ("I prefer concise answers", True),
    ("I live in Chennai", True),
    ("I am a full stack developer", True),
    ("ok", False),
    ("hi", False),
    ("I use React and Python daily", True),
    ("respond in tanglish", True),
    ("I want to build an AI startup", True),
]

passed = 0
for msg, should_match in tests:
    results = classify_memory(msg)
    matched = len(results) > 0
    ok = matched == should_match
    status = "PASS" if ok else "FAIL"
    content = results[0].content if results else "NOISE"
    cat = results[0].category if results else "-"
    print(f"[{status}] '{msg}' => [{cat}] {content}")
    if ok:
        passed += 1

print(f"\n{passed}/{len(tests)} tests passed")
