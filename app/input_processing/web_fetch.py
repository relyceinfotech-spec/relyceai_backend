"""
Safe Web Fetch — robots.txt-compliant page scraper.
Fetches URL content, extracts clean text, blocks SSRF.

Pipeline:
  1. Validate URL (block internal IPs, file://, ftp://)
  2. Check robots.txt
  3. Fetch page with httpx (10s timeout)
  4. Extract clean text with BeautifulSoup
  5. Cap at 12000 chars (~3000 tokens)

Returns standard tool contract for tool_executor.py integration.
"""
import re
import asyncio
import urllib.robotparser
from typing import Dict
from urllib.parse import urlparse


# ============================================
# SECURITY
# ============================================

_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
_BLOCKED_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                     "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                     "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                     "172.30.", "172.31.", "192.168.", "169.254.")

MAX_WEB_TOKENS = 2500       # token cap for injected content
MAX_CONTENT_CHARS = MAX_WEB_TOKENS * 4  # ~4 chars per token estimate
FETCH_TIMEOUT = 10

URL_PATTERN = re.compile(r'https?://\S+')


def detect_urls(text: str) -> list:
    """Extract URLs from user message text."""
    return URL_PATTERN.findall(text)


def _is_safe_url(url: str) -> bool:
    """Block SSRF: reject internal IPs, file://, ftp://."""
    if not url.startswith("http"):
        return False

    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False

    if host in _BLOCKED_HOSTS:
        return False
    if any(host.startswith(p) for p in _BLOCKED_PREFIXES):
        return False

    return True


# ============================================
# ROBOTS.TXT
# ============================================

async def _check_robots(url: str) -> bool:
    """Check robots.txt compliance. Returns True if allowed."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)

        # Fetch robots.txt with timeout
        import httpx
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(robots_url, follow_redirects=True)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
                return rp.can_fetch("*", url)

        # If robots.txt not found, assume allowed
        return True
    except Exception:
        # If we can't check, assume allowed (most sites allow)
        return True


# ============================================
# FETCH + EXTRACT
# ============================================

async def fetch_and_extract(url: str) -> Dict:
    """
    Fetch a URL and extract clean text content.
    Returns standard tool contract.
    """
    # Security check
    if not _is_safe_url(url):
        return {
            "status": "failure",
            "data": "URL blocked: internal/private addresses not allowed.",
            "source": "web_fetch",
            "confidence": "low",
        }

    # Robots.txt check
    allowed = await _check_robots(url)
    if not allowed:
        return {
            "status": "failure",
            "data": "Website does not allow automated scraping (robots.txt).",
            "source": "web_fetch",
            "confidence": "low",
        }

    try:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RelyceAI/1.0; +https://relyce.ai)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)

        if resp.status_code != 200:
            return {
                "status": "failure",
                "data": f"HTTP {resp.status_code} fetching {url}",
                "source": "web_fetch",
                "confidence": "low",
            }

        html = resp.text

        # Extract text with BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "iframe", "noscript", "form"]):
            tag.decompose()

        # Try article/main content first
        content_el = soup.find("article") or soup.find("main") or soup.find("body")
        if content_el:
            text = content_el.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # Cap content
        if len(clean_text) > MAX_CONTENT_CHARS:
            clean_text = clean_text[:MAX_CONTENT_CHARS] + "\n\n[Content truncated at ~3000 tokens]"

        if not clean_text or len(clean_text) < 50:
            return {
                "status": "failure",
                "data": "Could not extract meaningful content from page.",
                "source": "web_fetch",
                "confidence": "low",
            }

        # Get title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        return {
            "status": "success",
            "data": {
                "url": url,
                "title": title,
                "content": clean_text,
                "length": len(clean_text),
            },
            "source": "web_fetch",
            "confidence": "high",
        }

    except asyncio.TimeoutError:
        return {
            "status": "failure",
            "data": f"Timeout fetching {url} ({FETCH_TIMEOUT}s)",
            "source": "web_fetch",
            "confidence": "low",
        }
    except Exception as e:
        return {
            "status": "failure",
            "data": f"Error fetching {url}: {str(e)[:200]}",
            "source": "web_fetch",
            "confidence": "low",
        }


# ============================================
# TOOL INTERFACE (for tool_executor.py)
# ============================================

async def _tool_web_fetch(args: str = "", **kwargs) -> Dict:
    """Tool executor interface: fetches a URL and returns clean text."""
    url = args.strip().strip('"').strip("'")
    if not url:
        return {
            "status": "failure",
            "data": "No URL provided.",
            "source": "web_fetch",
            "confidence": "low",
        }
    return await fetch_and_extract(url)
