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

    Falls back to character-based estimation (~4 chars per token) when
    the tokenizer is unavailable.

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

    # Fallback: ~4 chars per token for code
    return len(text) // 4


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
