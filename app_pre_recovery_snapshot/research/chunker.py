"""
Structural Chunking for Research Results.
Replaces arbitrary token splitting with header-aware splitting.
preserves semantic boundaries.
"""
import re

# Content safety limits
MAX_CONTENT_BYTES = 200 * 1024  # 200KB
BINARY_SIGNATURES = [b'\x89PNG', b'GIF8', b'\xff\xd8\xff', b'%PDF', b'PK\x03\x04']


class ContentTooLargeError(ValueError):
    """Raised when fetched content exceeds the size cap."""
    pass


class BinaryContentError(ValueError):
    """Raised when fetched content appears to be binary."""
    pass


def validate_content(raw: str | bytes) -> str:
    """
    Validate fetched content before chunking.
    Rejects: >200KB HTML, binary content.
    Returns cleaned text if valid.
    """
    if isinstance(raw, bytes):
        # Binary detection
        for sig in BINARY_SIGNATURES:
            if raw[:8].startswith(sig):
                raise BinaryContentError(f"Binary content detected (signature: {sig[:4]})")
        raw = raw.decode("utf-8", errors="replace")

    if len(raw.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise ContentTooLargeError(
            f"Content size ({len(raw.encode('utf-8'))} bytes) exceeds {MAX_CONTENT_BYTES} byte limit."
        )

    return raw

def split_by_headers(text: str) -> list[str]:
    """
    Splits text by markdown headers (# to ######).
    Each section starts with the header (if present).
    """
    if not text:
        return []
        
    # Split, keeping the delimiter
    # (\n#{1,6}\s+) creates a capturing group so re.split returns the headers too
    parts = re.split(r'(\n#{1,6}\s+)', text)
    
    sections = []
    current_section = parts[0] if parts else ""
    
    for i in range(1, len(parts), 2):
        header = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        
        if current_section.strip():
            sections.append(current_section.strip())
            
        current_section = header.strip() + " " + content
        
    if current_section.strip():
        sections.append(current_section.strip())
        
    return sections

def cap_chunk_length(section: str, max_chars: int = 2000) -> str:
    """Caps a chunk length without breaking mid-word if possible."""
    if len(section) <= max_chars:
        return section
        
    capped = section[:max_chars]
    # Try to trim to the last space to avoid cutting words in half
    last_space = capped.rfind(' ')
    if last_space > max_chars * 0.8: # Only if it's reasonably close to the end
        capped = capped[:last_space]
        
    return capped + "..."

def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Splits into semantic sections and caps them."""
    sections = split_by_headers(text)
    return [cap_chunk_length(sec, max_chars) for sec in sections]
