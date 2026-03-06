"""
Trust Scoring for Research Sources.
Assigns trust score based on domain.
Penalizes low-quality domains, boosts official documentation.
"""
from urllib.parse import urlparse

TRUSTED_DOMAINS = {
    "react.dev": 1.0,
    "nextjs.org": 1.0,
    "docs.python.org": 1.0,
    "developer.mozilla.org": 0.95,
    "github.com": 0.9,
    "stackoverflow.com": 0.7,
}

LOW_TRUST_PATTERNS = [
    "medium.com",
    "dev.to",
    "blogspot",
]

def extract_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""

def compute_trust_score(url: str) -> float:
    domain = extract_domain(url)
    
    # Precise match or subdomain match for trusted
    for t_domain, score in TRUSTED_DOMAINS.items():
        if domain == t_domain or domain.endswith("." + t_domain):
            return score

    # Pattern match for low trust
    for pattern in LOW_TRUST_PATTERNS:
        if pattern in domain:
            return 0.4

    return 0.6  # Default neutral trust

def enforce_domain_diversity(results: list[dict], max_results: int = 5, max_per_domain: int = 2) -> list[dict]:
    """Ensure top K sources come from different domains with a max cap per domain."""
    domain_counts = {}
    unique = []

    for r in results:
        url = r.get("url", "")
        domain = extract_domain(url)
        
        count = domain_counts.get(domain, 0)
        if count < max_per_domain:
            if domain: # only track if we could parse the domain
                domain_counts[domain] = count + 1
            unique.append(r)
            if len(unique) >= max_results:
                break

    return unique
