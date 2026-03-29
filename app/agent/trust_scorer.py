"""
Source Trust Scorer
Ranks domains based on reliability (e.g., official docs > blogs > social media).
"""
import re
from typing import Dict

# Domain trust lookup
TRUST_MAP: Dict[str, float] = {
    # High Trust
    ".gov": 0.95,
    ".edu": 0.90,
    "wikipedia.org": 0.85,
    "github.com": 0.85,
    "docs.python.org": 0.95,
    "developer.mozilla.org": 0.95,
    "microsoft.com": 0.90,
    "apple.com": 0.90,
    "aws.amazon.com": 0.90,
    "google.com": 0.85,
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "bloomberg.com": 0.85,
    
    # Medium Trust
    "stackoverflow.com": 0.75,
    "medium.com": 0.60,
    "reddit.com": 0.50,
    "youtube.com": 0.50,
    
    # Low Trust (often SEO spam or unverified)
    "blogspot.com": 0.40,
    "wordpress.com": 0.40,
    "quora.com": 0.30,
    "facebook.com": 0.20,
    "twitter.com": 0.20,
    "x.com": 0.20,
}

def score_source(url: str) -> float:
    """
    Returns a trust score between 0.0 and 1.0 based on the URL domain.
    Default score for unknown domains is 0.50.
    """
    if not url:
        return 0.0
    
    url_lower = url.lower()
    
    # Check specific mappings first
    for domain, score in TRUST_MAP.items():
        if domain in url_lower:
            return score
            
    # Check TLD fallbacks
    if ".gov" in url_lower: return 0.95
    if ".edu" in url_lower: return 0.90
    if ".org" in url_lower: return 0.70
    
    return 0.50
