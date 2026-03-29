"""
Evaluation Runner — Sends test cases to the assistant and collects scored results.

Usage:
  python -m app.evaluation.run_eval                    # keyword scoring only
  python -m app.evaluation.run_eval --llm-judge        # + LLM judge scoring
  python -m app.evaluation.run_eval --category coding  # filter by category

Output: JSON report to stdout + summary table.
"""
import json
import asyncio
import time
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


DATASET_PATH = Path(__file__).parent / "dataset.json"


# ============================================
# ASSISTANT QUERY
# ============================================

async def query_assistant(question: str, timeout: float = 30.0) -> str:
    """Send a question to the assistant and get the response."""
    try:
        from app.llm.processor import llm_processor

        full_response = ""
        async for chunk in llm_processor.stream_response(
            user_query=question,
            context_messages=[],
            analysis={"sub_intent": "general", "mode": "chat"},
            chat_id="eval_test",
            user_id="eval_user",
            session_id="eval_session",
        ):
            if isinstance(chunk, str):
                full_response += chunk

        return full_response.strip()

    except Exception as e:
        return f"[ERROR] {str(e)[:200]}"


# ============================================
# EVALUATION RUN
# ============================================

async def run_evaluation(
    category: Optional[str] = None,
    use_llm_judge: bool = False,
    limit: Optional[int] = None,
) -> Dict:
    """
    Run the full evaluation pipeline.
    Returns structured results report.
    """
    from app.evaluation.scorer import evaluate_response

    # Load dataset
    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    # Filter by category
    if category:
        dataset = [t for t in dataset if t.get("category") == category]

    if limit:
        dataset = dataset[:limit]

    print(f"\n{'='*60}")
    print(f"  EVALUATION RUN: {len(dataset)} test cases")
    print(f"  LLM Judge: {'ON' if use_llm_judge else 'OFF'}")
    if category:
        print(f"  Category filter: {category}")
    print(f"{'='*60}\n")

    results = []
    start_time = time.time()

    for i, test in enumerate(dataset):
        test_id = test.get("id", f"test_{i}")
        question = test["question"]

        # Skip web tests if not explicitly enabled
        if test.get("requires_web") and not os.getenv("EVAL_ENABLE_WEB"):
            print(f"  [{test_id}] SKIP (requires web)")
            continue

        print(f"  [{test_id}] Querying: {question[:60]}...")

        # Query assistant
        q_start = time.time()
        response = await query_assistant(question)
        q_latency = time.time() - q_start

        # Score response
        score_result = await evaluate_response(
            test_case=test,
            response=response,
            use_llm_judge=use_llm_judge,
        )
        score_result["latency_s"] = round(q_latency, 2)
        score_result["response_preview"] = response[:150]

        status = "PASS" if score_result["passed"] else "FAIL"
        print(f"  [{test_id}] {status}  score={score_result['final_score']:.2f}  "
              f"latency={q_latency:.1f}s  "
              f"matched={len(score_result.get('matched', []))}/{len(score_result.get('matched', []))+len(score_result.get('missed', []))}")

        results.append(score_result)

    total_time = time.time() - start_time

    # Build report
    report = _build_report(results, total_time, use_llm_judge)
    return report


# ============================================
# REPORT
# ============================================

def _build_report(results: List[Dict], total_time: float, llm_judge: bool) -> Dict:
    """Build structured evaluation report."""
    if not results:
        return {"error": "No results", "total": 0}

    scores = [r["final_score"] for r in results]
    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"scores": [], "passed": 0, "total": 0}
        categories[cat]["scores"].append(r["final_score"])
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1

    cat_summary = {}
    for cat, data in categories.items():
        cat_summary[cat] = {
            "avg_score": round(sum(data["scores"]) / len(data["scores"]), 3),
            "pass_rate": round(data["passed"] / data["total"], 3),
            "total": data["total"],
        }

    report = {
        "summary": {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(results), 3),
            "avg_score": round(sum(scores) / len(scores), 3),
            "avg_latency_s": round(sum(r.get("latency_s", 0) for r in results) / len(results), 2),
            "total_time_s": round(total_time, 1),
            "llm_judge": llm_judge,
        },
        "categories": cat_summary,
        "results": results,
    }

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"  Pass Rate: {report['summary']['pass_rate']*100:.1f}%")
    print(f"  Avg Score: {report['summary']['avg_score']:.3f}")
    print(f"  Avg Latency: {report['summary']['avg_latency_s']}s")
    print(f"  Total Time: {report['summary']['total_time_s']}s")
    print()

    print(f"  {'Category':<20} {'Score':>8} {'Pass Rate':>10} {'Tests':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*6}")
    for cat, data in sorted(cat_summary.items()):
        print(f"  {cat:<20} {data['avg_score']:>8.3f} "
              f"{data['pass_rate']*100:>9.1f}% {data['total']:>6}")

    print()

    # Show failures
    failures = [r for r in results if not r.get("passed")]
    if failures:
        print(f"  FAILURES:")
        for r in failures:
            print(f"    [{r['id']}] score={r['final_score']:.2f} "
                  f"missed={r.get('missed', [])}")
    print(f"{'='*60}\n")

    return report


# ============================================
# CLI ENTRY
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM judge scoring")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--limit", type=int, help="Max test cases")
    parser.add_argument("--output", type=str, help="Save JSON report to file")
    args = parser.parse_args()

    async def main():
        from dotenv import load_dotenv
        load_dotenv()

        report = await run_evaluation(
            category=args.category,
            use_llm_judge=args.llm_judge,
            limit=args.limit,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  Report saved to: {args.output}")

    asyncio.run(main())
