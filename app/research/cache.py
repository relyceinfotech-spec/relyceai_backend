"""
Research Cache.
TTL-based in-memory cache for research queries.
"""
import time
from typing import Optional, Dict, Any

# In-memory store
_CACHE: Dict[str, Dict[str, Any]] = {}
TTL_SECONDS = 600  # 10 minutes
MAX_CACHE_SIZE = 100  # Prevent unbounded memory growth

def get_cached(query: str) -> Optional[Any]:
    """Retrieve cached research data if it exists and is fresh."""
    entry = _CACHE.get(query)
    if not entry:
        return None

    if time.time() - entry["timestamp"] > TTL_SECONDS:
        del _CACHE[query]
        return None

    return entry["data"]

def set_cache(query: str, data: Any) -> None:
    """Store research data in cache with current timestamp."""
    # Evict oldest if at capacity
    if len(_CACHE) >= MAX_CACHE_SIZE and query not in _CACHE:
        oldest_key = min(_CACHE, key=lambda k: _CACHE[k]["timestamp"])
        del _CACHE[oldest_key]

    _CACHE[query] = {
        "data": data,
        "timestamp": time.time()
    }

def clear_cache() -> None:
    """Clear all entries (useful for testing)."""
    _CACHE.clear()
