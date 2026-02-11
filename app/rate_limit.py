from collections import defaultdict, deque
from time import time
from typing import Deque, Dict

RATE_LIMIT_PER_MINUTE = 30
WINDOW_SECONDS = 60

_requests: Dict[str, Deque[float]] = defaultdict(deque)


def check_rate_limit(user_id: str) -> bool:
    """
    Simple in-memory sliding window rate limit.
    Returns True if allowed, False if rate limit exceeded.
    """
    if not user_id:
        user_id = "anonymous"
    now = time()
    queue = _requests[user_id]

    # Drop old timestamps
    while queue and now - queue[0] > WINDOW_SECONDS:
        queue.popleft()

    if len(queue) >= RATE_LIMIT_PER_MINUTE:
        return False

    queue.append(now)
    return True
