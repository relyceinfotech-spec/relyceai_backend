import os
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# CONFIGURATION
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gpt-5-mini")

# FULL TOOL SUITE
TOOLS = {
    "Search": "https://google.serper.dev/search",
    "Images": "https://google.serper.dev/images",
    "Videos": "https://google.serper.dev/videos",
    "Places": "https://google.serper.dev/places",
    "Maps": "https://google.serper.dev/maps",
    "Reviews": "https://google.serper.dev/reviews",
    "News": "https://google.serper.dev/news",
    "Shopping": "https://google.serper.dev/shopping",
    "Lens": "https://google.serper.dev/lens",
    "Scholar": "https://google.serper.dev/scholar",
    "Patents": "https://google.serper.dev/patents",
    "Webpage": "https://scrape.serper.dev"
}

# RELYCE AI SYSTEM PROMPT
RELYCE_SYSTEM_PROMPT = """You are **Relyce AI**, an elite strategic advisor.
You are a highly accomplished and multi-faceted AI assistant, functioning as an **elite consultant and strategic advisor** for businesses and startups. Your persona embodies the collective expertise of a Chief Operating Officer, a Head of Legal, a Chief Technology Officer, and a Chief Ethics Officer.

**Core Mandate:**
You must provide zero-hallucination, fact-based guidance operating with:
1. **Technical Proficiency:** Ability to discuss technology stacks, software development, data analytics, and cybersecurity with precision.
2. **Ethical Integrity:** A commitment to responsible AI usage, data privacy, and understanding the societal impact of business decisions.
3. **Legal Prudence:** Awareness of legal frameworks, IP, and compliance.
4. **Corporate Identity (Relyce AI):** You are the proprietary AI engine of **Relyce AI**. You are NOT affiliated with OpenAI.

**Strict Guidelines for Response Generation:**
* **Internal Logic:** If the query is math, coding, or logic, solve it with high precision using your internal knowledge.
* **Context-Bound (External):** If the query requires external data, use ONLY the provided context.
* **Zero Hallucination:** If information is missing, state it clearly.
* **Tone:** Professional, authoritative, and advisory.

**STRICT OUTPUT FORMATTING:**
You must strictly follow this visual structure. Do NOT use numbered lists (1, 2, 3) for the headers.

- First line: A short, descriptive **Title** (No Markdown bolding, just plain text).
- Second line: A blank line.
- Third section: The **Answer** (The detailed response).
- Fourth section: A blank line.
- Final section: List **all Sources** used. 
  * For Internal Knowledge: Source: [Internal Knowledge Base]
  * For Web: Source: [Link]
"""

client = OpenAI(api_key=OPENAI_API_KEY)

def get_headers():
    return {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

def analyze_query_intent(user_query):
    """
    Decides if the query is INTERNAL (Math/Code/Logic) or EXTERNAL (Needs Data).
    """
    system_prompt = (
        "You are a router. Analyze the user's query.\n"
        "1. If the query is simple math (e.g. '50*3'), coding, basic logic, or a greeting, return 'INTERNAL'.\n"
        "2. If the query requires real-world data, news, places, specific facts, or deep research, return 'EXTERNAL'.\n"
        "Output ONLY 'INTERNAL' or 'EXTERNAL'."
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    return response.choices[0].message.content.strip()

def llm_internal_solve(user_query):
    """
    Solves logic/math/code questions strictly with the LLM using Relyce Persona.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RELYCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )
    return response.choices[0].message.content

def select_multiple_tools(user_query):
    """
    Asks the LLM to select ALL relevant tools for a deep dive.
    """
    tools_list = ", ".join(TOOLS.keys())
    system_prompt = (
        f"You are a Senior Research Architect. The user wants a 'Deep Search' on: '{user_query}'.\n"
        f"Available Tools: [{tools_list}]\n"
        "Select the top 3-5 tools that will provide the most comprehensive, detailed, and varied data.\n"
        "Rules:\n"
        "- ALWAYS include 'Search'.\n"
        "- Include 'News' for current events.\n"
        "- Include 'Scholar' or 'Patents' ONLY for technical/academic topics.\n"
        "- Include 'Places'/'Maps' for locations.\n"
        "Return the tool names as a comma-separated list (e.g., 'Search, News, Videos')."
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": system_prompt}]
    )
    
    selected_str = response.choices[0].message.content.strip()
    selected_tools = [t.strip() for t in selected_str.split(',') if t.strip() in TOOLS]
    
    if not selected_tools:
        return ["Search", "News"]
        
    return selected_tools

def execute_serper_batch(endpoint_url, queries, param_key="q"):
    payload_queries = [{param_key: q} for q in queries] if isinstance(queries, list) else [{param_key: queries}]
    payload = json.dumps(payload_queries)
    
    try:
        response = requests.request("POST", endpoint_url, headers=get_headers(), data=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def llm_deep_synthesize(user_query, aggregated_data):
    """
    Synthesizes data using the strict Relyce AI persona.
    """
    context_str = json.dumps(aggregated_data, indent=2)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RELYCE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context Documents:\n{context_str}\n\nUser Query: {user_query}"}
        ]
    )
    return response.choices[0].message.content

def run_deepsearch():
    print("=========================================================")
    print("        RELYCE AI: DEEPSEARCH STRATEGIC ADVISOR          ")
    print("              (Type '0258' to Exit)                      ")
    print("=========================================================")
    
    while True:
        query = input("\n[Relyce Deepsearch] Enter Query: ")
        
        if query.strip() == "0258":
            print("\nRelyce AI Session Terminated.")
            break
            
        if not query.strip():
            continue

        # 1. Check Intent
        intent = analyze_query_intent(query)
        
        if intent == "INTERNAL":
            print(f"\n[Relyce] Mode: Internal Knowledge Base")
            final_answer = llm_internal_solve(query)
        
        else:
            # 2. External Deep Search
            print(f"\n... Relyce AI is analyzing query complexity and selecting tools ...")
            selected_tools = select_multiple_tools(query)
            print(f"-> Activated Modules: {', '.join(selected_tools)}")
            
            aggregated_context = {}
            
            for tool in selected_tools:
                print(f"   ... Retrieving intelligence from {tool} ...")
                endpoint = TOOLS[tool]
                param_key = "url" if tool == "Webpage" else "q"
                
                result = execute_serper_batch(endpoint, [query], param_key=param_key)
                aggregated_context[tool] = result
                
            print("\n... Synthesizing strategic insight (Zero Hallucination Mode) ...\n")
            final_answer = llm_deep_synthesize(query, aggregated_context)
        
        # Output Result
        print("-" * 60)
        print(final_answer)
        print("-" * 60)

if __name__ == "__main__":
    run_deepsearch()