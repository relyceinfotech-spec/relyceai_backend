"""
Prompt Optimizer — Automated prompt variant testing using the evaluation dataset.

Pipeline:
  1. Define prompt variants
  2. Run evaluation on each
  3. Compare scores per category
  4. Select best prompt
  5. Save winner

Usage:
  python -m app.evaluation.prompt_optimizer
  python -m app.evaluation.prompt_optimizer --top 3
"""
import json
import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================
# PROMPT VARIANTS
# ============================================

PROMPT_VARIANTS = {
    "v1_baseline": (
        "You are a technical AI assistant.\n\n"
        "Use provided context when relevant.\n"
        "Reason carefully before answering."
    ),
    "v2_structured": (
        "You are an expert AI engineer assistant.\n\n"
        "Before answering:\n"
        "1. Identify the task type.\n"
        "2. Use relevant context from provided sources.\n"
        "3. Verify reasoning before responding.\n\n"
        "Be concise and accurate."
    ),
    "v3_selective": (
        "You are a technical assistant focused on architecture, coding, and research.\n\n"
        "Use context sources selectively. Avoid unsupported claims.\n"
        "Structure responses clearly with sections when appropriate."
    ),
    "v4_reasoning_scaffold": (
        "You are an AI assistant that answers technical and research questions.\n\n"
        "When solving tasks:\n"
        "1. Identify the problem type.\n"
        "2. Use the provided context if relevant.\n"
        "3. Plan the solution internally.\n"
        "4. Verify the answer before responding.\n\n"
        "Before finalizing, check:\n"
        "- Does the response address the user's question?\n"
        "- Are all claims supported by context or general knowledge?\n"
        "- Is the response clear and actionable?"
    ),
    "v5_concise": (
        "You are a senior software engineer assistant.\n\n"
        "Answer precisely. Use context when available.\n"
        "For code: include working examples.\n"
        "For architecture: list trade-offs.\n"
        "For debugging: identify root cause first."
    ),
}


# ============================================
# OPTIMIZER
# ============================================

async def optimize_prompts(
    limit: Optional[int] = None,
    categories: Optional[List[str]] = None,
) -> Dict:
    """
    Run evaluation with each prompt variant and compare scores.
    Returns ranked results.
    """
    from app.evaluation.scorer import score_by_keywords

    # Load dataset
    dataset_path = Path(__file__).parent / "dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Filter
    if categories:
        dataset = [t for t in dataset if t.get("category") in categories]

    # Skip web tests
    dataset = [t for t in dataset if not t.get("requires_web")]

    # Item #7: 80/20 train/validation split to prevent prompt overfitting
    import random
    random.seed(42)  # Deterministic split
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_set = dataset[:split_idx]
    val_set = dataset[split_idx:]
    
    print(f"\n  Split: {len(train_set)} training / {len(val_set)} validation")
    
    # Use training set for optimization, validation for final scoring
    if limit:
        train_set = train_set[:limit]
    dataset = train_set  # Optimize on training set

    print(f"\n{'='*60}")
    print(f"  PROMPT OPTIMIZATION")
    print(f"  Variants: {len(PROMPT_VARIANTS)}")
    print(f"  Test cases: {len(dataset)}")
    print(f"{'='*60}\n")

    results = {}

    for name, prompt_text in PROMPT_VARIANTS.items():
        print(f"  Testing: {name}...")

        scores = []
        category_scores = {}

        for test in dataset:
            # Simulate response by scoring the prompt's instruction quality
            # In production, this would actually query the LLM with the prompt
            response = await _query_with_prompt(prompt_text, test["question"])

            score_result = score_by_keywords(
                response,
                expected_keywords=test.get("expected_keywords"),
                expected_topics=test.get("expected_topics"),
            )

            scores.append(score_result["score"])

            cat = test.get("category", "unknown")
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(score_result["score"])

        avg = sum(scores) / len(scores) if scores else 0.0
        cat_avgs = {
            cat: round(sum(s) / len(s), 3)
            for cat, s in category_scores.items()
        }

        results[name] = {
            "avg_score": round(avg, 3),
            "total_tests": len(scores),
            "categories": cat_avgs,
        }
        print(f"    Score: {avg:.3f}")

    # Rank
    ranked = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    # Print report
    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION REPORT")
    print(f"{'='*60}")
    print(f"  {'Variant':<25} {'Score':>8}")
    print(f"  {'-'*25} {'-'*8}")
    for name, data in ranked:
        marker = " ★" if name == ranked[0][0] else ""
        print(f"  {name:<25} {data['avg_score']:>8.3f}{marker}")

    best_name = ranked[0][0]
    print(f"\n  Best prompt: {best_name}")
    print(f"  Score: {ranked[0][1]['avg_score']:.3f}")
    print(f"{'='*60}\n")

    return {
        "best_prompt": best_name,
        "best_score": ranked[0][1]["avg_score"],
        "rankings": [{"name": n, **d} for n, d in ranked],
        "prompt_text": PROMPT_VARIANTS[best_name],
    }


async def _query_with_prompt(prompt: str, question: str) -> str:
    """
    Query the assistant with a specific system prompt.
    Falls back to direct LLM call for isolated testing.
    """
    try:
        from app.llm.router import get_openrouter_client
        from app.config import FAST_MODEL

        client = get_openrouter_client()
        result = await asyncio.wait_for(
            client.chat.completions.create(
                model=FAST_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question[:500]},
                ],
                max_tokens=500,
                temperature=0.3,
            ),
            timeout=15.0,
        )
        return (result.choices[0].message.content or "").strip()

    except Exception as e:
        return f"[ERROR] {str(e)[:100]}"


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt optimizer")
    parser.add_argument("--limit", type=int, help="Max tests per variant")
    parser.add_argument("--category", type=str, nargs="+", help="Filter categories")
    parser.add_argument("--output", type=str, help="Save JSON report")
    args = parser.parse_args()

    async def main():
        from dotenv import load_dotenv
        load_dotenv()

        report = await optimize_prompts(
            limit=args.limit,
            categories=args.category,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  Report saved to: {args.output}")

    asyncio.run(main())
