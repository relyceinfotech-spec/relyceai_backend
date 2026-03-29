
import asyncio
import sys
import os

# Add backend to path dynamically based on script location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm.processor import llm_processor

async def run_test(name, query, mode="normal"):
    print(f"\n--- TEST: {name} ---")
    print(f"Query: {query}")
    
    full_text = await llm_processor.process_agent_query(
        query, 
        user_id="test_user", 
        mode=mode
    )
            
    print("\n[GENERATED RESPONSE]")
    print(full_text)
    print("\n" + "="*50)
    return full_text

async def run_web_test(name, query, mock_search_data):
    print(f"\n--- TEST: {name} ---")
    print(f"Query: {query}")
    
    # We mock the process_deep_search_query logic but with injected data
    context_str = mock_search_data
    
    system_prompt = await llm_processor.process_agent_query(
        "dummy", mode="normal" 
    ) # This is just to get a prompt context, but we'll use processor internals
    
    # We'll use a hidden trick: process_deep_search_query uses get_system_prompt_for_mode
    from app.platform.llm_gateway import get_system_prompt_for_mode
    system_prompt = get_system_prompt_for_mode("normal", {}, "test_user", query)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Search Data:\n{context_str}\n\nUser Query: {query}"}
    ]
    
    from app.platform.llm_gateway import get_openrouter_client, LLM_MODEL
    client = get_openrouter_client()
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3
    )
    
    full_text = llm_processor._sanitize_output_text(resp.choices[0].message.content)
    
    print("\n[GENERATED WEB RESPONSE]")
    print(full_text)
    print("\n" + "="*50)
    return full_text

async def run_deep_dive_test():
    print("\n--- TEST: Deep Dive (Calvin Cycle) ---")
    # Context: previous query about Photosynthesis
    context = [
        {"role": "user", "content": "How does photosynthesis work?"},
        {"role": "assistant", "content": "Photosynthesis is a process... (summary of photosynthesis)"}
    ]
    query = "Explain the Calvin cycle in more detail. I want to understand the exact chemical steps."
    
    full_text = await llm_processor.process_agent_query(
        query, 
        user_id="test_user", 
        mode="normal" # This will trigger the educational prompt
    )
    # Note: In a real scenario, process_message would pass the context. 
    # For this unit test of the prompt style, we see how it handles a specific follow-up.
            
    print("\n[GENERATED RESPONSE]")
    print(full_text)
    print("\n" + "="*50)
    return full_text

async def main():
    test_cases = [
        ("Simple (Sky Blue)", "Why is the sky blue?"),
        ("Guidance (Entropy)", "What is entropy?"),
        ("Enrichment (Photosystems)", "Describe the photosystems PSI and PSII."),
        ("Technical (Expansion Base)", "Explain photosynthesis."),
        ("Expansion (Light Reactions)", "Tell me more about the light reactions in detail."),
        ("Long AI Answer", "Write a long answer about AI in 10 points")
    ]
    
    results = []
    for name, query in test_cases:
        res = await run_test(name, query)
        results.append(f"### {name}\n{res}\n\n---\n")

    # Web Normalization Test
    mock_web_data = """
    [Source 1: TechBlog] Kubernetes is great. It has pods and services. Pods are smallest units.
    [Source 2: Official Docs] Kubernetes (K8s) is an open-source system for automating deployment, scaling, and management of containerized applications. Key features: Auto-scaling, Self-healing.
    [Source 3: Random Forum] K8s is hard to learn but good for scale. It uses nodes.
    [Noise] Click here for cheap GPUs! Subscribe to our newsletter.
    """
    web_res = await run_web_test("Web Normalization (K8s)", "What is Kubernetes and its benefits?", mock_web_data)
    results.append(f"### Web Normalization\n{web_res}\n\n---\n")

    # Web Deduplication & Synthesis Test
    mock_multi_source = """
    SOURCE 1 (MIT):
    Photosynthesis consists of two stages:
    1. Light reactions in thylakoid membranes
    2. Calvin cycle in stroma

    SOURCE 2 (Britannica):
    Calvin cycle is the second phase of photosynthesis.
    Calvin cycle steps:
    - CO2 fixation via RuBisCO
    - Reduction to G3P
    - RuBP regeneration

    SOURCE 3 (Nature Education):
    Energy cost for photosynthesis:
    3 CO2 + 9 ATP + 6 NADPH → 1 G3P
    The Calvin cycle occurs in the stroma of the chloroplast.
    """
    dedup_res = await run_web_test("Web Deduplication (Photosynthesis)", "How does the Calvin cycle work?", mock_multi_source)
    results.append(f"### Web Deduplication & synthesis\n{dedup_res}\n\n---\n")

    # FACT Mode Test
    fact_res = await run_test("FACT Mode (NVIDIA)", "Who founded NVIDIA and when?")
    results.append(f"### FACT Mode Check\n{fact_res}\n\n---\n")

    # Hierarchy & Readability Check (Generic Headings)
    hierarchy_res = await run_test("Hierarchy & Breaks (Quantum)", "Explain quantum superposition in detail.")
    results.append(f"### Hierarchy & Readability Check\n{hierarchy_res}\n\n---\n")

    # Pedagogical Layering Test (Idea -> Intuition -> Mechanism)
    layering_res = await run_test("Pedagogical Layering (Entropy)", "Explain entropy to a beginner.")
    results.append(f"### Pedagogical Layering Check\n{layering_res}\n\n---\n")
    
    output_path = os.path.join(os.path.dirname(__file__), "test_results.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(results)
    print(f"\n[DONE] All tests completed. Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
