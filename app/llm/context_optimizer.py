"""
Relyce AI - Context Window Optimizer
Smart context compression that maximizes relevant information density.

Architecture:
- Importance scoring: code blocks > errors > questions > greetings
- Progressive compression: old messages get summarized, recent stay verbatim
- Signal preservation: always keep code, errors, and user corrections
- Token budgeting: estimate token count, compress to fit model window
"""
import re
from typing import List, Dict, Optional, Tuple


# ==========================================
# Importance Weights
# ==========================================
IMPORTANCE_WEIGHTS = {
    "code_block": 0.9,      # Contains code
    "error_message": 0.9,   # Contains error/traceback
    "user_correction": 0.8, # "no", "that's wrong", "not what I meant"
    "user_goal": 0.85,      # User's stated goal/intent — critical context
    "emotional_signal": 0.75, # Emotional content — preserves tone awareness
    "question": 0.7,        # Contains "?"
    "technical": 0.6,       # Technical terms
    "greeting": 0.1,        # "hi", "hello", "thanks"
    "acknowledgment": 0.2,  # "ok", "got it", "sure"
    "default": 0.4,
}

# Signal patterns
CODE_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`|def |class |import |from |function |const |let |var ')
ERROR_PATTERN = re.compile(r'error|traceback|exception|failed|crash|bug|broken|TypeError|ValueError|SyntaxError', re.IGNORECASE)
CORRECTION_PATTERN = re.compile(r"no[,.]?\s|that'?s wrong|not what i|incorrect|actually|i meant", re.IGNORECASE)
GREETING_PATTERN = re.compile(r'^(hi|hello|hey|thanks|thank you|bye|ok|sure|got it|alright)\s*[.!]?\s*$', re.IGNORECASE)
QUESTION_PATTERN = re.compile(r'\?')
TECHNICAL_PATTERN = re.compile(r'api|database|server|deploy|docker|kubernetes|webpack|nginx|redis|firebase|auth', re.IGNORECASE)

# New: Emotional signal patterns (Priority #4 + #6)
EMOTIONAL_PATTERN = re.compile(r'frustrat|confus|stuck|help|struggling|annoyed|lost|don\'t understand|can\'t figure', re.IGNORECASE)
# New: User goal patterns — these express what the user is trying to achieve
GOAL_PATTERN = re.compile(r'(?:i\'?m trying|i need|my goal|i want to|i\'?m building|i\'?m working on|the idea is)', re.IGNORECASE)



class ContextOptimizer:
    def __init__(self, max_tokens: int = 6000, chars_per_token: float = 4.0):
        """
        max_tokens: Target token budget for context
        chars_per_token: Approximate characters per token
        """
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token
        self.max_chars = int(max_tokens * chars_per_token)

    # ==========================================
    # Importance Scoring
    # ==========================================
    def score_message(self, message: Dict) -> float:
        """Score a single message by importance (0.0 - 1.0)."""
        content = message.get("content", "")
        role = message.get("role", "user")
        
        # Start with default
        score = IMPORTANCE_WEIGHTS["default"]
        
        # Code blocks are always important
        if CODE_PATTERN.search(content):
            score = max(score, IMPORTANCE_WEIGHTS["code_block"])
        
        # Errors are critical
        if ERROR_PATTERN.search(content):
            score = max(score, IMPORTANCE_WEIGHTS["error_message"])
        
        # User corrections are important context
        if role == "user" and CORRECTION_PATTERN.search(content):
            score = max(score, IMPORTANCE_WEIGHTS["user_correction"])

        # User goals — critical for maintaining context about what user is building
        if role == "user" and GOAL_PATTERN.search(content):
            score = max(score, IMPORTANCE_WEIGHTS["user_goal"])

        # Emotional signals — preserve for tone-aware responses
        if EMOTIONAL_PATTERN.search(content):
            score = max(score, IMPORTANCE_WEIGHTS["emotional_signal"])
        
        # Questions carry context
        if QUESTION_PATTERN.search(content):
            score = max(score, IMPORTANCE_WEIGHTS["question"])
        
        # Technical content
        if TECHNICAL_PATTERN.search(content):
            score = max(score, IMPORTANCE_WEIGHTS["technical"])
        
        # Greetings/acknowledgments are low value
        if GREETING_PATTERN.match(content.strip()):
            score = IMPORTANCE_WEIGHTS["greeting"]
        elif len(content.strip()) < 15 and not CODE_PATTERN.search(content):
            score = IMPORTANCE_WEIGHTS["acknowledgment"]
        
        return score

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate."""
        return max(1, int(len(text) / self.chars_per_token))

    # ==========================================
    # Compression Strategies
    # ==========================================
    def compress_message(self, message: Dict) -> Dict:
        """Compress a single message while preserving key signals."""
        content = message.get("content", "")
        role = message.get("role", "user")
        
        # If message is already short, don't compress
        if len(content) < 200:
            return message
        
        compressed_parts = []
        
        # Always preserve code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        if code_blocks:
            # Keep code blocks, summarize surrounding text
            non_code = re.sub(r'```[\s\S]*?```', '[CODE]', content)
            # Truncate non-code text
            if len(non_code) > 150:
                non_code = non_code[:150] + "..."
            compressed_parts.append(non_code)
            for block in code_blocks[:2]:  # Keep max 2 code blocks
                compressed_parts.append(block)
        else:
            # No code: truncate to key sentences
            sentences = re.split(r'[.!?]\s+', content)
            kept = []
            char_count = 0
            for sentence in sentences:
                if char_count + len(sentence) > 300:
                    break
                kept.append(sentence)
                char_count += len(sentence)
            compressed_parts.append(". ".join(kept) + "..." if len(sentences) > len(kept) else ". ".join(kept))
        
        compressed_content = "\n".join(compressed_parts)
        
        return {
            **message,
            "content": compressed_content,
            "_compressed": True
        }

    # ==========================================
    # Main Optimization
    # ==========================================
    def optimize(
        self,
        messages: List[Dict],
        keep_last_n: int = 4,
        summary: str = ""
    ) -> List[Dict]:
        """
        Optimize context messages to fit within token budget.
        
        Strategy:
        1. Always keep last N messages verbatim (most relevant)
        2. Score older messages by importance
        3. Keep high-importance old messages, compress medium ones, drop low ones
        4. Inject summary for dropped context
        """
        if not messages:
            return []

        total_chars = sum(len(m.get("content", "")) for m in messages)
        
        # If already within budget, return as-is
        if total_chars <= self.max_chars:
            return messages

        # Split into recent (keep verbatim) and older (optimize)
        if len(messages) <= keep_last_n:
            return messages
        
        recent = messages[-keep_last_n:]
        older = messages[:-keep_last_n]

        # Score older messages
        scored = [(msg, self.score_message(msg)) for msg in older]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Budget for older messages
        recent_chars = sum(len(m.get("content", "")) for m in recent)
        summary_chars = len(summary) if summary else 0
        remaining_budget = self.max_chars - recent_chars - summary_chars

        # Build optimized older context
        optimized_older = []
        chars_used = 0

        for msg, score in scored:
            msg_chars = len(msg.get("content", ""))
            
            if score >= 0.7:
                # High importance: keep verbatim if budget allows
                if chars_used + msg_chars <= remaining_budget:
                    optimized_older.append(msg)
                    chars_used += msg_chars
                else:
                    # Compress high-importance messages that don't fit
                    compressed = self.compress_message(msg)
                    comp_chars = len(compressed.get("content", ""))
                    if chars_used + comp_chars <= remaining_budget:
                        optimized_older.append(compressed)
                        chars_used += comp_chars
            
            elif score >= 0.4:
                # Medium importance: compress
                compressed = self.compress_message(msg)
                comp_chars = len(compressed.get("content", ""))
                if chars_used + comp_chars <= remaining_budget:
                    optimized_older.append(compressed)
                    chars_used += comp_chars
            
            # Score < 0.4: drop (greetings, acknowledgments)
        
        # Re-sort by original order
        original_order = {id(msg): i for i, msg in enumerate(older)}
        optimized_older_with_index = []
        for msg in optimized_older:
            # Find original index
            for orig_msg in older:
                if orig_msg.get("content") == msg.get("content") or \
                   (msg.get("_compressed") and orig_msg.get("content", "")[:50] in msg.get("content", "")):
                    optimized_older_with_index.append((older.index(orig_msg), msg))
                    break
            else:
                optimized_older_with_index.append((0, msg))
        
        optimized_older_with_index.sort(key=lambda x: x[0])
        optimized_older = [msg for _, msg in optimized_older_with_index]

        # Combine: optimized older + recent
        result = optimized_older + recent
        
        return result

    def get_stats(self, messages: List[Dict]) -> Dict:
        """Get optimization statistics."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        scores = [self.score_message(m) for m in messages]
        return {
            "total_messages": len(messages),
            "total_chars": total_chars,
            "estimated_tokens": self.estimate_tokens(str(total_chars)),
            "avg_importance": round(sum(scores) / max(len(scores), 1), 3),
            "high_importance": sum(1 for s in scores if s >= 0.7),
            "low_importance": sum(1 for s in scores if s < 0.3),
        }


# Singleton
context_optimizer = ContextOptimizer(max_tokens=6000)
