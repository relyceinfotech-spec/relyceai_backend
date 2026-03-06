import re

with open('app/llm/processor.py', 'r', encoding='utf-8') as f:
    text = f.read()

p_start = text.find('async def _process_agent_mode')
p_end = text.find('async def process_message_stream') 

part1 = text[:p_start]
part2 = text[p_start:p_end]
part3 = text[p_end:]

injection = """
        _event_seq = 0
        import json as _json
        def _emit_info(payload: dict) -> str:
            nonlocal _event_seq
            payload["seq"] = _event_seq
            _event_seq += 1
            return f"[INFO]{_json.dumps(payload)}"
        
        def _emit_intel(payload: dict) -> str:
            nonlocal _event_seq
            payload["seq"] = _event_seq
            _event_seq += 1
            return f"[INFO]INTEL:{_json.dumps(payload)}"
"""

part2 = part2.replace('execution_id = kwargs.get("task_id") or f"exec_{int(_time.time())}"', 
                      'execution_id = kwargs.get("task_id") or f"exec_{int(_time.time())}"' + injection)

part2 = re.sub(r'yield\s+f?[\'"]\[INFO\]INTEL:\{_json\.dumps\((.*?)\)\}[\'"]', r'yield _emit_intel(\1)', part2)
part2 = re.sub(r'yield\s+f?[\'"]\[INFO\]\{_json\.dumps\((.*?)\)\}[\'"]', r'yield _emit_info(\1)', part2)

with open('app/llm/processor.py', 'w', encoding='utf-8') as f:
    f.write(part1 + part2 + part3)
print('Refactored.')
