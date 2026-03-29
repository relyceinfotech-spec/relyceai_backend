"""
Response Synthesizer
Synthesizes aggregated observations and tool results into a final high-quality response.
"""
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.config import FAST_MODEL
from app.llm.prompts import RESPONSE_SYNTHESIZER_SYSTEM_PROMPT


class ResponseSynthesizer:
    def __init__(self, client, model_to_use: str = FAST_MODEL):
        self.client = client
        self.model_to_use = model_to_use

    @staticmethod
    def _sanitize_final_text(text: str) -> str:
        out = str(text or "").strip()
        if not out:
            return out
        out = out.replace("\u2014", " - ").replace("\u2013", " - ")
        out = re.sub(
            r"^\s*As\s+Grok,\s*built\s+by\s+xAI,\s*",
            "As Relyce AI, ",
            out,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        out = re.sub(r"\bGrok\b", "Relyce AI", out, flags=re.IGNORECASE)
        out = re.sub(r"\bxAI\b", "Relyce", out, flags=re.IGNORECASE)

        # Remove leaked prompt artifacts like:
        # "Goal: <query>### Relyce AI Response"
        out = re.sub(
            r"(?is)^\s*goal\s*:\s*.*?relyce\s*ai\s*response\s*[:\-#]*\s*",
            "",
            out,
        )
        out = re.sub(r"(?im)^\s*#{0,6}\s*goal\s*:\s*.*$", "", out)
        out = re.sub(r"(?im)^\s*#{0,6}\s*relyce\s*ai\s*response\s*:?\s*$", "", out)

        lines = [ln for ln in out.splitlines()]
        while lines and lines[0].strip().lower().startswith("goal:"):
            lines.pop(0)
        out = "\n".join(lines).strip()

        # Collapse accidental full-answer duplication when output is repeated twice.
        core = out.strip()
        if len(core) >= 60:
            half = len(core) // 2
            left = core[:half].strip()
            right = core[half:].strip()
            if left and left == right:
                out = left
        return out.strip()

    @staticmethod
    def _normalize_dedupe_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @staticmethod
    def _dedupe_paragraph_blocks(text: str) -> str:
        blocks = [b.strip() for b in re.split(r"\n{2,}", str(text or "").strip()) if b.strip()]
        out: List[str] = []
        seen = set()
        for block in blocks:
            key = ResponseSynthesizer._normalize_dedupe_key(block)
            # Ignore tiny blocks for dedupe key collisions.
            if len(key) >= 40 and key in seen:
                continue
            if len(key) >= 40:
                seen.add(key)
            out.append(block)
        return "\n\n".join(out).strip()

    @staticmethod
    def _split_sentences(text: str, max_sentences: int = 6) -> List[str]:
        raw = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
        out: List[str] = []
        for sentence in raw:
            s = sentence.strip(" -\t")
            if not s:
                continue
            if len(s) > 260:
                s = s[:257].rstrip() + "..."
            out.append(s)
            if len(out) >= max_sentences:
                break
        return out

    @staticmethod
    def _postprocess_research_layout(text: str) -> str:
        lines = [ln.rstrip() for ln in str(text or "").splitlines()]
        output: List[str] = []
        buffer_para: List[str] = []

        def flush_para() -> None:
            nonlocal buffer_para
            if not buffer_para:
                return
            para = " ".join(x.strip() for x in buffer_para if x.strip()).strip()
            buffer_para = []
            if not para:
                return
            # Convert long prose paragraphs into bullets for readability.
            if len(para) > 170:
                for s in ResponseSynthesizer._split_sentences(para, max_sentences=5):
                    output.append(f"- {s}")
            else:
                output.append(f"- {para}")

        for raw in lines:
            line = raw.strip()
            if not line:
                flush_para()
                if output and output[-1] != "":
                    output.append("")
                continue

            is_heading = bool(re.match(r"^\s{0,3}(#{1,6}\s+.+|[A-Z][^:\n]{2,80}:)\s*$", line))
            is_bullet = bool(re.match(r"^\s*([-*]|[0-9]+\.)\s+", line))

            if is_heading:
                flush_para()
                # Normalize heading style for clean UI rendering.
                heading = re.sub(r"^#{1,6}\s*", "", line).strip()
                output.append(f"### {heading}")
                continue

            if is_bullet:
                flush_para()
                bullet = re.sub(r"^\s*([-*]|[0-9]+\.)\s+", "", line).strip()
                if bullet:
                    output.append(f"- {bullet}")
                continue

            buffer_para.append(line)

        flush_para()

        # Remove duplicate bullets/headings after transformation.
        deduped: List[str] = []
        seen = set()
        for ln in output:
            if not ln.strip():
                if deduped and deduped[-1] != "":
                    deduped.append("")
                continue
            key = ResponseSynthesizer._normalize_dedupe_key(ln)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ln)
        return "\n".join(deduped).strip()

    @staticmethod
    def _is_research_like_goal(goal: str) -> bool:
        q = str(goal or "").lower()
        markers = {
            "research", "reserch", "recent", "latest", "today", "current",
            "war", "fight", "conflict", "attack", "timeline", "news",
            "iran", "israel", "pakistan", "ukraine", "russia", "gaza",
        }
        return any(m in q for m in markers)

    @staticmethod
    def _coerce_trust(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        txt = str(value or "").strip().lower()
        if txt in {"high", "verified"}:
            return 0.9
        if txt in {"medium", "moderate"}:
            return 0.7
        if txt in {"low", "unverified"}:
            return 0.3
        try:
            return float(txt)
        except Exception:
            return 0.5

    @staticmethod
    def _compact_observations(observations: List[str], max_items: int = 10) -> List[str]:
        out: List[str] = []
        seen = set()
        for obs in observations or []:
            line = str(obs or "").strip()
            if not line:
                continue
            line = re.sub(r"^Node\s+[A-Za-z0-9_]+\s+\([^)]+\):\s*", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            key = line.lower()[:220]
            if key in seen:
                continue
            seen.add(key)
            out.append(line[:520])
            if len(out) >= max_items:
                break
        return out

    @classmethod
    def _top_sources(cls, sources: Optional[List[Dict[str, Any]]], max_items: int = 8) -> List[Dict[str, Any]]:
        items = []
        for s in sources or []:
            if not isinstance(s, dict):
                continue
            url = str(s.get("url") or s.get("link") or "").strip()
            if not url:
                continue
            host = (urlparse(url).netloc or "").lower()
            if not host:
                continue
            items.append(
                {
                    "url": url,
                    "host": host,
                    "trust": cls._coerce_trust(s.get("trust_score", s.get("trust", 0.5))),
                }
            )
        items.sort(key=lambda x: x["trust"], reverse=True)
        out: List[Dict[str, Any]] = []
        seen_url = set()
        for item in items:
            if item["url"] in seen_url:
                continue
            seen_url.add(item["url"])
            out.append(item)
            if len(out) >= max_items:
                break
        return out

    async def synthesize(
        self,
        goal: str,
        observations: List[str],
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Synthesize the final response.
        """
        if not observations:
            return "I was unable to gather enough information to fulfill your request."
        system_prompt = RESPONSE_SYNTHESIZER_SYSTEM_PROMPT
        research_like = self._is_research_like_goal(goal)
        compact_findings = self._compact_observations(observations, max_items=10)
        top_sources = self._top_sources(sources, max_items=8)

        sources_text = ""
        if top_sources:
            sources_text = "\nSOURCES & TRUST SCORES:\n"
            for s in top_sources:
                sources_text += f"- {s.get('url')} (Trust: {float(s.get('trust', 0.5)):.2f})\n"

        format_hint = ""
        if research_like:
            format_hint = (
                "\nOUTPUT STYLE (ADAPTIVE):\n"
                "Choose section headings dynamically based on the actual findings.\n"
                "Do NOT force a fixed template.\n"
                "Use 2-5 relevant sections only, such as:\n"
                "- Latest Situation\n"
                "- What Happened\n"
                "- Timeline\n"
                "- Military/Political Impact\n"
                "- Regional Reactions\n"
                "- What Is Still Unclear\n"
                "- Sources\n"
                "Rules:\n"
                "- Keep headings content-specific (not generic copy-paste).\n"
                "- Do not add greeting/opening chat lines (for example 'Hey macha').\n"
                "- If timeline data is weak, skip timeline section.\n"
                "- If source quality is mixed, include a short uncertainty note.\n"
                "- Prefer short bullets and short paragraphs; avoid repetitive blocks.\n"
                "- Emoji usage: optional and contextual only (0-3 max), for example 📅 timeline, ⚠ uncertainty, 🔗 sources.\n"
                "- Never use random or decorative emojis.\n"
            )

        user_content = (
            f"INITIAL GOAL: {goal}\n\n"
            f"RESEARCH FINDINGS:\n{chr(10).join(compact_findings or observations)}\n"
            f"{sources_text}\n"
            f"{format_hint}\n"
            "Synthesize the final response now:"
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
            final_text = self._sanitize_final_text(response.choices[0].message.content)
            final_text = self._dedupe_paragraph_blocks(final_text)
            if research_like:
                final_text = self._postprocess_research_layout(final_text)
            return final_text
        except Exception as e:
            print(f"[Synthesizer] LLM call failed: {e}")
            return "I hit an internal issue while preparing the final answer. Please retry in a moment."
