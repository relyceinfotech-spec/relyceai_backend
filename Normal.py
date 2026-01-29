import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gpt-4o")

TOOLS = {
    "Search": "https://google.serper.dev/search",
    "Places": "https://google.serper.dev/places",
    "Maps": "https://google.serper.dev/maps",
    "Reviews": "https://google.serper.dev/reviews",
    "News": "https://google.serper.dev/news",
    "Shopping": "https://google.serper.dev/shopping",
    "Scholar": "https://google.serper.dev/scholar"
}

RELYCE_SYSTEM_PROMPT = """You are **Relyce AI**, an elite strategic advisor.
**Core Mandate:**
Provide fact-based guidance using a hybrid of search data and your own expert internal knowledge.
**Identity:**
You are a proprietary AI model developed by **Relyce AI**. You are NOT affiliated with OpenAI. You must NEVER mention GPT models or OpenAI.

**Guidelines:**
* **Synthesis:** Combine search data with internal knowledge.
* **Tone:** Professional, authoritative, and advisory.
**STRICT OUTPUT FORMATTING:**
- First line: Title
- Second line: Blank
- Third section: Answer (If you provide any code/commands, wrap them in ```language code blocks)
- Fourth section: Blank
- Final section: Sources (Format: Source: [Link])
"""

client = OpenAI(api_key=OPENAI_API_KEY)

def get_headers():
    return {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

def analyze_query_intent(user_query):
    system_prompt = "Output ONLY 'INTERNAL' for greetings, simple logic, simple coding, or small talk. Output 'EXTERNAL' for data queries."
    response = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}])
    return response.choices[0].message.content.strip()

def llm_internal_solve(user_query):
    # Short & Sweet prompt for Internal queries
    system_prompt = (
        "You are Relyce AI. The user asked a simple question (greeting, logic, or code).\n"
        "Provide a **short, sweet, and professional** answer.\n"
        "If you provide any code, terminal commands, or technical snippets, ALWAYS wrap them in markdown code blocks with the appropriate language identifier (e.g., ```bash, ```python, etc.).\n"
        "Do NOT include a title. Do NOT include sources. Just the answer."
    )
    response = client.chat.completions.create(model=MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}])
    return response.choices[0].message.content

def select_multiple_tools(user_query):
    tools_list = ", ".join(TOOLS.keys())
    system_prompt = f"Select relevant tools from [{tools_list}] for: '{user_query}'. Return comma-separated list."
    response = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": system_prompt}])
    selected_str = response.choices[0].message.content.strip()
    selected_tools = [t.strip() for t in selected_str.split(',') if t.strip() in TOOLS]
    if not selected_tools: return ["Search"]
    return selected_tools

def execute_serper_batch(endpoint_url, queries):
    payload = json.dumps([{"q": q} for q in (queries if isinstance(queries, list) else [queries])])
    response = requests.request("POST", endpoint_url, headers=get_headers(), data=payload)
    return response.json() if response.status_code == 200 else {"error": response.text}

def llm_synthesize_answer(user_query, search_data):
    context_str = json.dumps(search_data, indent=2)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RELYCE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Search Data:\n{context_str}\n\nUser Query: {user_query}"}
        ]
    )
    return response.choices[0].message.content

def run_normal():
    while True:
        query = input("\n[Relyce Normal] Enter Query: ")
        if query.strip() == "0258": break
        if not query.strip(): continue
        
        if analyze_query_intent(query) == "INTERNAL":
            print("\n" + "-"*60)
            print(llm_internal_solve(query))
            print("-" * 60)
            continue 

        print(f"[{MODEL}] Mode: External Data Required")
        selected_tools = select_multiple_tools(query)
        print(f"-> Activated Modules: {', '.join(selected_tools)}")
        aggregated_context = {}
        for tool in selected_tools:
            print(f"   ... Fetching intelligence from {tool} ...")
            result = execute_serper_batch(TOOLS[tool], [query])
            aggregated_context[tool] = result
        print("\n... Synthesizing strategic insight ...\n")
        print("-" * 60)
        print(llm_synthesize_answer(query, aggregated_context))
        print("-" * 60)

if __name__ == "__main__":
    run_normal()