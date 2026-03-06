import sys
path = r'd:\finalai\testing\backend\app\llm\processor.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

count = 0
for i, line in enumerate(lines):
    if 'delta = chunk.choices[0].delta' in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines[i] = indent + 'if not hasattr(chunk, "choices") or not chunk.choices: continue\n' + line
        count += 1

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Patched {count} occurrences of delta = chunk.choices[0].delta")
