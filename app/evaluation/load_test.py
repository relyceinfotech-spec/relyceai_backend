"""
Load Test Script — Simulates concurrent users hitting the assistant.

Usage:
  python -m app.evaluation.load_test                     # 10 users
  python -m app.evaluation.load_test --users 50           # 50 users
  python -m app.evaluation.load_test --users 100 --rps 5  # 100 users, 5 req/sec

Reports: TTFT, avg latency, error rate, throughput.
"""
import asyncio
import time
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional


# Test queries (mix of simple and complex)
TEST_QUERIES = [
    "Hello",
    "What is Python?",
    "Explain async/await in Python",
    "Write a function to reverse a string",
    "How does a load balancer work?",
    "What is the difference between REST and GraphQL?",
    "Explain the CAP theorem",
    "Write a binary search implementation",
    "What are microservices?",
    "How do database indexes work?",
    "Explain WebSocket vs HTTP polling",
    "What is Docker and why use it?",
    "Write a rate limiter in Python",
    "Explain OAuth 2.0 flow",
    "What is eventual consistency?",
]

BASE_URL = "http://localhost:8080"


async def send_request(
    query: str,
    session_id: str,
    timeout: float = 30.0,
) -> Dict:
    """Send a single HTTP request to the chat stream endpoint."""
    import httpx

    start = time.time()
    first_byte_time = 0.0
    total_bytes = 0
    status = "success"
    error = ""

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{BASE_URL}/chat/stream",
                json={
                    "messages": [{"role": "user", "content": query}],
                    "mode": "chat",
                    "session_id": session_id,
                },
                timeout=timeout,
            )

            if response.status_code == 200:
                total_bytes = len(response.content)
                first_byte_time = time.time()
            elif response.status_code == 503:
                status = "rejected"
                error = "Server busy (503)"
            else:
                status = "error"
                error = f"HTTP {response.status_code}"

    except httpx.TimeoutException:
        status = "timeout"
        error = f"Timeout after {timeout}s"
    except httpx.ConnectError:
        status = "connection_error"
        error = "Connection refused"
    except Exception as e:
        status = "error"
        error = str(e)[:100]

    end = time.time()
    latency = end - start
    ttft = (first_byte_time - start) if first_byte_time else latency

    return {
        "query": query[:50],
        "status": status,
        "latency_s": round(latency, 3),
        "ttft_s": round(ttft, 3),
        "bytes": total_bytes,
        "error": error,
    }


async def run_load_test(
    num_users: int = 10,
    requests_per_second: float = 2.0,
    duration_seconds: float = 0.0,
) -> Dict:
    """
    Run concurrent load test.
    Spawns num_users concurrent tasks with throttled request rate.
    """
    print(f"\n{'='*60}")
    print(f"  LOAD TEST")
    print(f"  Users: {num_users}")
    print(f"  Request rate: {requests_per_second}/sec")
    print(f"{'='*60}\n")

    results: List[Dict] = []
    start = time.time()
    delay = 1.0 / requests_per_second if requests_per_second > 0 else 0

    async def user_task(user_id: int):
        query = random.choice(TEST_QUERIES)
        session_id = f"loadtest_user_{user_id}"
        result = await send_request(query, session_id)
        result["user_id"] = user_id
        results.append(result)

        status_char = "." if result["status"] == "success" else "X"
        print(status_char, end="", flush=True)

    # Spawn users with throttling
    tasks = []
    for i in range(num_users):
        tasks.append(asyncio.create_task(user_task(i)))
        if delay and i < num_users - 1:
            await asyncio.sleep(delay)

    await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start
    print()

    # Build report
    report = _build_load_report(results, total_time)
    return report


def _build_load_report(results: List[Dict], total_time: float) -> Dict:
    """Build load test summary."""
    if not results:
        return {"error": "No results"}

    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] != "success"]

    latencies = [r["latency_s"] for r in success]
    ttfts = [r["ttft_s"] for r in success]

    report = {
        "total_requests": len(results),
        "successful": len(success),
        "failed": len(errors),
        "error_rate": round(len(errors) / len(results), 4),
        "throughput_rps": round(len(results) / total_time, 2),
        "total_time_s": round(total_time, 1),
    }

    if latencies:
        latencies.sort()
        report["latency"] = {
            "avg_s": round(sum(latencies) / len(latencies), 3),
            "min_s": round(latencies[0], 3),
            "max_s": round(latencies[-1], 3),
            "p50_s": round(latencies[len(latencies)//2], 3),
            "p95_s": round(latencies[int(len(latencies)*0.95)], 3),
        }

    if ttfts:
        ttfts.sort()
        report["ttft"] = {
            "avg_s": round(sum(ttfts) / len(ttfts), 3),
            "p50_s": round(ttfts[len(ttfts)//2], 3),
            "p95_s": round(ttfts[int(len(ttfts)*0.95)], 3),
        }

    if errors:
        from collections import Counter
        error_types = Counter(r["status"] for r in errors)
        report["error_breakdown"] = dict(error_types)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  LOAD TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Requests: {report['total_requests']}")
    print(f"  Success:  {report['successful']}")
    print(f"  Failed:   {report['failed']}")
    print(f"  Error Rate: {report['error_rate']*100:.1f}%")
    print(f"  Throughput: {report['throughput_rps']} req/s")

    if "latency" in report:
        lat = report["latency"]
        print(f"\n  Latency:")
        print(f"    avg={lat['avg_s']}s  p50={lat['p50_s']}s  p95={lat['p95_s']}s  max={lat['max_s']}s")

    if "ttft" in report:
        ttft = report["ttft"]
        print(f"\n  TTFT:")
        print(f"    avg={ttft['avg_s']}s  p50={ttft['p50_s']}s  p95={ttft['p95_s']}s")

    if "error_breakdown" in report:
        print(f"\n  Errors: {report['error_breakdown']}")

    print(f"{'='*60}\n")

    return report


# CLI entry
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load test")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--rps", type=float, default=2.0, help="Requests per second")
    parser.add_argument("--output", type=str, help="Save JSON report")
    args = parser.parse_args()

    async def main():
        report = await run_load_test(
            num_users=args.users,
            requests_per_second=args.rps,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  Report saved to: {args.output}")

    asyncio.run(main())
