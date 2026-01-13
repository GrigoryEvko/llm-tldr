"""Unified tokenization utilities.

Provides consistent token counting across the codebase using Qwen3 tokenizer
with character-estimate fallback when the tokenizer is unavailable.

Thread-safe lazy initialization ensures the tokenizer is loaded once and
shared across all callers.
"""

import logging
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)

_tokenizer: Optional[Any] = None
_tokenizer_lock = threading.Lock()


def estimate_tokens_fallback(text: str) -> int:
    """Estimate tokens when tokenizer unavailable.

    Uses character-type heuristics for better accuracy than simple len//4.
    Different character types have different token densities:
    - ASCII/code: ~4 chars per token
    - CJK characters: ~1.5 chars per token (each ideograph often = 1 token)
    - Emoji: ~3 tokens per emoji
    - Other Unicode: ~2 chars per token

    Args:
        text: Text to estimate token count for.

    Returns:
        Estimated number of tokens (minimum 1 for non-empty text).
    """
    if not text:
        return 0

    ascii_chars = 0
    cjk_chars = 0
    emoji_chars = 0
    other_chars = 0

    for char in text:
        code = ord(char)
        if code < 128:
            # ASCII (includes code, English text)
            ascii_chars += 1
        elif (0x4E00 <= code <= 0x9FFF or      # CJK Unified Ideographs
              0x3400 <= code <= 0x4DBF or      # CJK Extension A
              0x20000 <= code <= 0x2A6DF or    # CJK Extension B
              0x2A700 <= code <= 0x2B73F or    # CJK Extension C
              0x2B740 <= code <= 0x2B81F or    # CJK Extension D
              0xF900 <= code <= 0xFAFF or      # CJK Compatibility
              0x3000 <= code <= 0x303F or      # CJK Punctuation
              0x3040 <= code <= 0x309F or      # Hiragana
              0x30A0 <= code <= 0x30FF or      # Katakana
              0xAC00 <= code <= 0xD7AF):       # Korean Hangul
            cjk_chars += 1
        elif (0x1F300 <= code <= 0x1F9FF or    # Misc Symbols and Pictographs, Emoticons
              0x1FA00 <= code <= 0x1FA6F or    # Chess Symbols, Extended-A
              0x1F600 <= code <= 0x1F64F or    # Emoticons
              0x2600 <= code <= 0x26FF or      # Misc Symbols
              0x2700 <= code <= 0x27BF or      # Dingbats
              0xFE00 <= code <= 0xFE0F or      # Variation Selectors
              0x1F000 <= code <= 0x1F02F):     # Mahjong, Dominos
            emoji_chars += 1
        else:
            # Other Unicode (extended Latin, Cyrillic, Arabic, etc.)
            other_chars += 1

    # Estimate tokens by character type density
    estimated = (
        ascii_chars / 4.0 +      # ~4 ASCII chars per token
        cjk_chars * 1.5 +        # Each CJK char often = 1-2 tokens
        emoji_chars * 3.0 +      # Emoji = 2-4 tokens each
        other_chars / 2.0        # Other Unicode ~2 chars per token
    )

    return max(1, int(estimated))


def get_tokenizer() -> Optional[Any]:
    """Get or create the Qwen3 tokenizer.

    Uses double-check locking pattern for thread-safe lazy initialization.
    Returns None if the tokenizer cannot be loaded.

    Returns:
        The Qwen3 tokenizer instance, or None if unavailable.
    """
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                try:
                    from tokenizers import Tokenizer

                    _tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
                except Exception as e:
                    logger.warning(f"Failed to load Qwen3 tokenizer: {e}")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in text using Qwen3 tokenizer.

    Falls back to character-type aware estimation when the tokenizer
    is unavailable (handles CJK, emoji, and other Unicode correctly).

    Args:
        text: Text to count tokens for.

    Returns:
        Number of tokens.

    Raises:
        ValueError: If text is None.
    """
    if text is None:
        raise ValueError("text cannot be None")
    if not text:
        return 0

    tokenizer = get_tokenizer()
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        except Exception:
            pass

    # Fallback: character-type aware estimation
    return estimate_tokens_fallback(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget.

    Uses the tokenizer for precise truncation when available,
    falls back to character estimation otherwise.

    Args:
        text: Text to truncate.
        max_tokens: Maximum number of tokens allowed.

    Returns:
        Truncated text that fits within the token budget.

    Raises:
        ValueError: If max_tokens is negative.
    """
    if not text:
        return text
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")

    tokenizer = get_tokenizer()
    if tokenizer is None:
        # Fallback: ~4 chars per token
        return text[: max_tokens * 4]

    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded.ids) <= max_tokens:
            return text

        truncated_ids = encoded.ids[:max_tokens]
        return tokenizer.decode(truncated_ids)
    except Exception:
        # Fallback on any tokenizer error
        return text[: max_tokens * 4]
