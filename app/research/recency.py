"""
Recency Scoring for Research Sources.
Weighting based on published dates.
"""
import datetime
from typing import Optional

def compute_recency_score(published_date: Optional[datetime.date] = None) -> float:
    """
    Computes a recency score from 0.4 to 1.0.
    1.0 = < 30 days
    0.8 = < 180 days
    0.6 = < 365 days
    0.4 = > 365 days
    0.5 = No date
    """
    if not published_date:
        return 0.5

    days_old = (datetime.date.today() - published_date).days

    if days_old < 30:
        return 1.0
    elif days_old < 180:
        return 0.8
    elif days_old < 365:
        return 0.6
    
    return 0.4

def parse_date_heuristic(date_str: str) -> Optional[datetime.date]:
    """Basic heuristic to parse common search API date strings if needed."""
    if not date_str:
        return None
    try:
        # e.g. '2023-10-01' or '2023-10-01T...Z'
        if len(date_str) >= 10:
            return datetime.date.fromisoformat(date_str[:10])
    except ValueError:
        pass
    return None

def final_source_score(similarity: float, trust: float, recency: float) -> float:
    """Combine signals into a final source ranking weight."""
    return (similarity * 0.6) + (trust * 0.25) + (recency * 0.15)
