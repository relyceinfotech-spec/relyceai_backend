"""
Evaluation Scorer — LLM-based and keyword-based response scoring.

Two scoring methods:
  1. Keyword/topic coverage (fast, no LLM call)
  2. LLM judge (deep quality assessment, optional)

Outputs score 0.0–1.0 per test case.
"""
import re
from typing import List, Dict, Optional


# ============================================
# KEYWORD SCORING (Fast, No LLM)
# ============================================

def score_by_keywords(
    response: str,
    expected_keywords: Optional[List[str]] = None,
    expected_topics: Optional[List[str]] = None,
) -> Dict:
    """
    Score response by keyword/topic coverage.
    Returns { score: float, matched: list, missed: list }
    """
    terms = expected_keywords or expected_topics or []
    if not terms:
        return {"score": 1.0, "matched": [], "missed": []}

    response_lower = response.lower()
    matched = []
    missed = []

    for term in terms:
        # Check if term or its variants appear
        term_lower = term.lower()
        if term_lower in response_lower:
            matched.append(term)
        # Also check individual words of multi-word terms
        elif len(term_lower.split()) > 1:
            words = term_lower.split()
            if all(w in response_lower for w in words):
                matched.append(term)
            else:
                missed.append(term)
        else:
            missed.append(term)

    coverage = len(matched) / len(terms) if terms else 1.0

    return {
        "score": round(coverage, 3),
        "matched": matched,
        "missed": missed,
    }


# ============================================
# LLM JUDGE SCORING (Deep, Requires API)
# ============================================

JUDGE_PROMPT = """Evaluate the following AI assistant response.

Question: {question}

Expected coverage: {expected}

Response:
{response}

Score from 0.0 to 1.0 based on:
• Topic coverage (are expected topics addressed?)
• Correctness (is the information accurate?)
• Clarity (is the response well-structured?)
• Completeness (are there missing important details?)

Reply with ONLY a JSON object:
{{"score": 0.XX, "reasoning": "brief explanation"}}"""


async def score_by_llm_judge(
    question: str,
    response: str,
    expected: List[str],
    timeout: float = 5.0,
) -> Dict:
    """
    Use a fast LLM to evaluate response quality.
    Returns { score: float, reasoning: str }
    """
    import asyncio
    import json

    try:
        from app.llm.router import get_openrouter_client
        from app.config import MEMORY_EXTRACTION_MODEL

        client = get_openrouter_client()
        prompt = JUDGE_PROMPT.format(
            question=question[:500],
            expected=", ".join(expected),
            response=response[:2000],
        )

        result = await asyncio.wait_for(
            client.chat.completions.create(
                model=MEMORY_EXTRACTION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,
            ),
            timeout=timeout,
        )

        raw = (result.choices[0].message.content or "").strip()

        # Parse JSON response
        # Handle cases where model wraps in ```json
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()

        parsed = json.loads(raw)
        return {
            "score": float(parsed.get("score", 0.0)),
            "reasoning": parsed.get("reasoning", ""),
        }

    except asyncio.TimeoutError:
        return {"score": -1.0, "reasoning": "LLM judge timed out"}
    except Exception as e:
        return {"score": -1.0, "reasoning": f"LLM judge error: {str(e)[:100]}"}


# ============================================
# COMBINED SCORING
# ============================================

async def evaluate_response(
    test_case: Dict,
    response: str,
    use_llm_judge: bool = False,
) -> Dict:
    """
    Full evaluation of a single test response.
    Combines keyword coverage + optional LLM judge.
    """
    # Keyword/topic coverage
    keyword_result = score_by_keywords(
        response,
        expected_keywords=test_case.get("expected_keywords"),
        expected_topics=test_case.get("expected_topics"),
    )

    result = {
        "id": test_case.get("id", ""),
        "category": test_case.get("category", ""),
        "keyword_score": keyword_result["score"],
        "matched": keyword_result["matched"],
        "missed": keyword_result["missed"],
        "response_length": len(response),
    }

    # Optional LLM judge
    if use_llm_judge:
        expected = test_case.get("expected_topics") or test_case.get("expected_keywords") or []
        judge_result = await score_by_llm_judge(
            question=test_case["question"],
            response=response,
            expected=expected,
        )
        result["llm_score"] = judge_result["score"]
        result["llm_reasoning"] = judge_result.get("reasoning", "")

        # Combined score: 40% keyword + 60% LLM judge (if available)
        if judge_result["score"] >= 0:
            result["final_score"] = round(
                keyword_result["score"] * 0.4 + judge_result["score"] * 0.6,
                3,
            )
        else:
            result["final_score"] = keyword_result["score"]
    else:
        result["final_score"] = keyword_result["score"]

    # Pass/fail
    min_score = test_case.get("min_score", 0.7)
    result["passed"] = result["final_score"] >= min_score
    result["min_score"] = min_score

    return result
