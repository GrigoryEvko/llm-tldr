"""
Semantic search data structures and extraction utilities.

This module provides:
- EmbeddingUnit: Dataclass for code units with 5-layer TLDR analysis
- build_embedding_text: Convert units to embedding-ready text
- extract_units_from_project: Extract all indexable code units from a project
- Smart chunking for oversized units (>8K tokens)
- Semantic pattern detection and code tagging

Token-based approach:
- All code extraction uses token counting, not line counting
- Oversized classes are split at method boundaries
- Oversized functions are split at logical block boundaries
- Each chunk maintains context with parent reference

DEPRECATED (2024-01): The following functions were removed in favor of ml_engine.py:
- build_semantic_index() -> use ml_engine.build_index()
- semantic_search() -> use ml_engine.search()
- get_model() -> use ml_engine.ModelManager
- compute_embedding() -> use ml_engine.ModelManager.encode()

The CLI (tldr semantic index/search) now uses ml_engine.py which provides:
- Qwen3-Embedding models (better than BGE)
- TEI backend for high-throughput inference
- usearch for faster vector search
- Matryoshka Representation Learning (MRL) support
"""

import logging
import multiprocessing as mp
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

from tldr.tokenizer import count_tokens, get_tokenizer, truncate_to_tokens

logger = logging.getLogger("tldr.semantic")

ALL_LANGUAGES = ["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "ruby", "php", "kotlin", "swift", "csharp", "scala", "lua", "elixir"]

# Token budget for embeddings (Qwen3 supports 32K, TEI configured for 16K)
MAX_EMBEDDING_TOKENS = 8192  # Conservative limit for good retrieval quality
MAX_CODE_PREVIEW_TOKENS = 6000  # Leave room for metadata in embedding text
CHUNK_OVERLAP_TOKENS = 200  # Overlap between chunks for context continuity

# Semantic pattern categories for code tagging
SEMANTIC_PATTERNS = {
    # Data operations
    "crud": r"\b(create|read|update|delete|insert|select|save|load|fetch|store|persist|get|set|add|remove)\b",
    "validation": r"\b(valid|validate|check|verify|assert|ensure|sanitize|normalize|parse|format)\b",
    "transform": r"\b(convert|transform|map|reduce|filter|sort|merge|split|join|serialize|deserialize)\b",

    # Control flow patterns
    "error_handling": r"\b(try|catch|except|raise|throw|error|exception|fail|panic)\b",
    "async_ops": r"\b(async|await|promise|future|callback|then|concurrent|parallel|thread)\b",
    "iteration": r"\b(for|while|loop|iterate|each|map|reduce|filter)\b",

    # Architecture patterns
    "api_endpoint": r"\b(route|endpoint|handler|controller|get|post|put|delete|patch|request|response)\b",
    "database": r"\b(query|sql|select|insert|update|delete|table|schema|migration|model|entity)\b",
    "auth": r"\b(auth|login|logout|session|token|jwt|oauth|permission|role|access)\b",
    "cache": r"\b(cache|memoize|memo|store|redis|memcache|ttl|expire|invalidate)\b",

    # Code quality
    "test": r"\b(test|spec|mock|stub|assert|expect|should|describe|it)\b",
    "logging": r"\b(log|logger|debug|info|warn|error|trace|print|console)\b",
    "config": r"\b(config|setting|option|env|environment|parameter|argument)\b",
}

def extract_code_by_tokens(content: str, start_offset: int, max_tokens: int) -> Tuple[str, int]:
    """Extract code from content starting at offset, limited by tokens.

    Unlike line-based extraction, this scans character-by-character and
    counts tokens to get the exact maximum context that fits.

    Args:
        content: Full file content.
        start_offset: Character offset to start extraction (clamped to 0 if negative).
        max_tokens: Maximum tokens to extract.

    Returns:
        Tuple of (extracted_code, end_offset).
    """
    start_offset = max(0, start_offset)
    if start_offset >= len(content):
        return "", start_offset

    # Extract from start_offset to end
    candidate = content[start_offset:]

    # Count tokens and truncate if needed
    token_count = count_tokens(candidate)
    if token_count <= max_tokens:
        return candidate, len(content)

    # Binary search for the right length
    tokenizer = get_tokenizer()
    if tokenizer is None:
        # Fallback: estimate ~4 chars per token
        estimated_chars = max_tokens * 4
        return candidate[:estimated_chars], start_offset + estimated_chars

    # Encode and decode truncated
    encoded = tokenizer.encode(candidate, add_special_tokens=False)
    truncated_ids = encoded.ids[:max_tokens]
    result = tokenizer.decode(truncated_ids)

    return result, start_offset + len(result)


def detect_semantic_patterns(code: str) -> Set[str]:
    """Detect semantic patterns in code for tagging.

    Scans code for common patterns (CRUD, validation, error handling, etc.)
    and returns matching categories for semantic enrichment.

    Note: Patterns use word boundaries (\\b) to match whole words only.
    This prevents false positives like "created" matching "crud" pattern.
    For example, "create_user" matches but "recreate" does not.

    Args:
        code: Code string to analyze.

    Returns:
        Set of matched pattern categories (e.g., {"crud", "validation"}).
    """
    if not code:
        return set()

    code_lower = code.lower()
    matched = set()

    for pattern_name, pattern_regex in SEMANTIC_PATTERNS.items():
        if re.search(pattern_regex, code_lower):
            matched.add(pattern_name)

    return matched


def _get_indent_depth(line: str) -> int:
    """Calculate indent depth handling both tabs and spaces consistently.

    Expands tabs to 4 spaces before counting, ensuring consistent indentation
    measurement regardless of whether the file uses tabs, spaces, or mixed
    indentation. This matches common editor behavior (tabstop=4).

    Args:
        line: A single line of code.

    Returns:
        Indentation depth (number of 4-space levels).
    """
    stripped = line.lstrip()
    if not stripped:
        return 0

    # BUG SE-11 FIX: Expand tabs to 4 spaces for consistent counting
    # This handles mixed tabs/spaces correctly by normalizing to spaces first
    leading_len = len(line) - len(stripped)
    leading = line[:leading_len]
    expanded = leading.replace('\t', '    ')
    return len(expanded) // 4


def detect_code_complexity(code: str) -> Dict[str, Any]:
    """Analyze code complexity without full AST parsing.

    Quick heuristic analysis of code structure for embedding enrichment.

    Args:
        code: Code string to analyze.

    Returns:
        Dict with complexity metrics.
    """
    if not code:
        return {"depth": 0, "branches": 0, "loops": 0}

    # Count indentation depth using helper that properly handles tabs vs spaces
    max_depth = 0
    for line in code.split("\n"):
        depth = _get_indent_depth(line)
        max_depth = max(max_depth, depth)

    # Count control structures
    branches = len(re.findall(r"\b(if|elif|else|case|switch|match)\b", code))
    loops = len(re.findall(r"\b(for|while|loop)\b", code))

    return {
        "depth": max_depth,
        "branches": branches,
        "loops": loops,
    }


def split_into_chunks(
    code: str,
    max_tokens: int,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Tuple[str, int, int]]:
    """Split code into token-limited chunks with overlap.

    Attempts to split at logical boundaries (blank lines, function defs).
    Each chunk overlaps with the previous for context continuity.

    Args:
        code: Code string to split.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between chunks.

    Returns:
        List of (chunk_text, start_char, end_char) tuples.
    """
    if not code:
        return []

    total_tokens = count_tokens(code)
    if total_tokens <= max_tokens:
        return [(code, 0, len(code))]

    chunks = []
    tokenizer = get_tokenizer()

    if tokenizer is None:
        # Fallback: split by estimated character count
        chars_per_chunk = max_tokens * 4
        overlap_chars = overlap_tokens * 4
        pos = 0
        while pos < len(code):
            end = min(pos + chars_per_chunk, len(code))
            chunks.append((code[pos:end], pos, end))
            # Ensure we always advance by at least 1 to avoid infinite loop
            # when overlap_chars >= chars_per_chunk
            pos = max(end - overlap_chars, pos + 1)
        return chunks

    # Token-based splitting with boundary detection
    lines = code.split("\n")
    current_chunk = []
    current_tokens = 0
    chunk_start = 0
    char_offset = 0

    for i, line in enumerate(lines):
        line_with_newline = line + "\n" if i < len(lines) - 1 else line
        line_tokens = count_tokens(line_with_newline)

        # Check if adding this line would exceed limit
        if current_tokens + line_tokens > max_tokens and current_chunk:
            # Save current chunk - use original slice to preserve exact byte offsets
            chunk_text = code[chunk_start:char_offset]
            chunks.append((chunk_text, chunk_start, char_offset))

            # Start new chunk with overlap
            # Find lines for overlap
            overlap_lines = []
            overlap_count = 0
            for prev_line in reversed(current_chunk):
                prev_tokens = count_tokens(prev_line)
                if overlap_count + prev_tokens > overlap_tokens:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_count += prev_tokens

            current_chunk = overlap_lines
            current_tokens = overlap_count
            # SE-4 fix: Calculate chunk_start accounting for newlines between lines
            # overlap_lines stores lines WITHOUT trailing newlines
            # The source text has newlines after each line (we're mid-file, not at end)
            # Total overlap length = sum of line lengths + number of newlines
            overlap_char_count = sum(len(l) for l in overlap_lines) + len(overlap_lines)
            chunk_start = char_offset - overlap_char_count

        current_chunk.append(line)
        current_tokens += line_tokens
        char_offset += len(line_with_newline)

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = code[chunk_start:char_offset]
        chunks.append((chunk_text, chunk_start, char_offset))

    return chunks


@dataclass(slots=True)
class EmbeddingUnit:
    """A code unit (function/method/class) for embedding.

    Contains information from all 5 TLDR layers:
    - L1: signature, docstring
    - L2: calls, called_by
    - L3: cfg_summary
    - L4: dfg_summary
    - L5: dependencies

    Plus semantic enrichment:
    - semantic_tags: Auto-detected patterns (crud, validation, async_ops, etc.)
    - complexity: Quick complexity metrics (depth, branches, loops)
    - chunk_index: For oversized units split into chunks
    - chunk_total: Total chunks for this unit
    - parent_name: For chunks, reference to parent unit
    - token_count: Actual token count for this unit
    """

    name: str
    qualified_name: str
    file: str
    line: int
    language: str
    unit_type: str  # "function" | "method" | "class" | "chunk"
    signature: str
    docstring: str
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    cfg_summary: str = ""
    dfg_summary: str = ""
    dependencies: str = ""
    code_preview: str = ""
    # Semantic enrichment
    semantic_tags: List[str] = field(default_factory=list)
    complexity: Dict[str, int] = field(default_factory=dict)
    # Chunking support
    chunk_index: int = 0
    chunk_total: int = 1
    parent_name: str = ""
    token_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        from dataclasses import asdict
        return asdict(self)

    def is_chunk(self) -> bool:
        """Check if this unit is a chunk of a larger unit."""
        return self.chunk_total > 1

    def needs_chunking(self) -> bool:
        """Check if this unit needs to be split into chunks."""
        return self.token_count > MAX_EMBEDDING_TOKENS


def chunk_unit(unit: EmbeddingUnit) -> List[EmbeddingUnit]:
    """Split an oversized unit into token-limited chunks.

    For classes: splits by method boundaries when possible.
    For functions: splits at logical block boundaries.

    Each chunk inherits metadata from parent and includes:
    - Signature and docstring in first chunk
    - Parent reference in all chunks
    - Chunk index/total for reconstruction

    Args:
        unit: The EmbeddingUnit to split.

    Returns:
        List of EmbeddingUnit chunks. Returns [unit] if no chunking needed.
    """
    if not unit.code_preview:
        return [unit]

    code_tokens = count_tokens(unit.code_preview)
    if code_tokens <= MAX_CODE_PREVIEW_TOKENS:
        # Update token count and return as-is
        unit.token_count = code_tokens
        return [unit]

    # Split the code into chunks
    chunks = split_into_chunks(unit.code_preview, MAX_CODE_PREVIEW_TOKENS)

    if len(chunks) <= 1:
        # Couldn't split effectively, just truncate
        unit.code_preview = truncate_to_tokens(unit.code_preview, MAX_CODE_PREVIEW_TOKENS)
        unit.token_count = count_tokens(unit.code_preview)
        return [unit]

    # Create chunk units
    chunk_units = []
    for i, (chunk_text, start_char, end_char) in enumerate(chunks):
        chunk_name = f"{unit.name}[{i+1}/{len(chunks)}]"

        chunk_unit = EmbeddingUnit(
            name=chunk_name,
            qualified_name=f"{unit.qualified_name}#chunk{i+1}",
            file=unit.file,
            line=unit.line + unit.code_preview[:start_char].count("\n") if i > 0 else unit.line,
            language=unit.language,
            unit_type="chunk",
            # First chunk gets full signature/docstring, others get abbreviated
            signature=unit.signature if i == 0 else f"// continued from {unit.name}",
            docstring=unit.docstring if i == 0 else "",
            calls=unit.calls if i == 0 else [],
            called_by=unit.called_by if i == 0 else [],
            cfg_summary=unit.cfg_summary if i == 0 else "",
            dfg_summary=unit.dfg_summary if i == 0 else "",
            dependencies=unit.dependencies,
            code_preview=chunk_text,
            # Semantic enrichment - detect for each chunk
            semantic_tags=list(detect_semantic_patterns(chunk_text)),
            complexity=detect_code_complexity(chunk_text),
            # Chunk metadata
            chunk_index=i,
            chunk_total=len(chunks),
            parent_name=unit.name,
            token_count=count_tokens(chunk_text),
        )
        chunk_units.append(chunk_unit)

    return chunk_units


def enrich_unit(unit: EmbeddingUnit) -> None:
    """Add semantic enrichment to a unit. Modifies unit in place.

    Detects patterns, calculates complexity, and computes token count.

    Args:
        unit: The EmbeddingUnit to enrich.

    Note:
        This function mutates the input unit directly. It does not return
        a value to avoid the confusing pattern of mutate-and-return.
    """
    if unit.code_preview:
        # Detect semantic patterns
        unit.semantic_tags = list(detect_semantic_patterns(unit.code_preview))
        # Calculate complexity
        unit.complexity = detect_code_complexity(unit.code_preview)
        # Count tokens
        unit.token_count = count_tokens(unit.code_preview)
    else:
        unit.semantic_tags = []
        unit.complexity = {}
        unit.token_count = 0


def _parse_identifier_to_words(name: str) -> str:
    """Parse camelCase/snake_case/PascalCase identifier to space-separated words.

    Converts code identifiers into natural language for better semantic search.

    Examples:
        getUserData -> get user data
        get_user_data -> get user data
        XMLParser -> xml parser
        _private_method -> private method
        HTMLElement -> html element

    Args:
        name: The identifier to parse.

    Returns:
        Space-separated lowercase words.
    """
    # Remove leading/trailing underscores
    name = name.strip("_")
    if not name:
        return ""

    # Handle snake_case: replace underscores with spaces
    name = name.replace("_", " ")

    # Use regex for camelCase/PascalCase splitting
    # This handles acronyms correctly: XMLParser -> XML Parser -> xml parser
    # Pattern: split before uppercase that follows lowercase, or before uppercase followed by lowercase
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)  # camelCase
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words)  # ACRONYMWord

    # Clean up multiple spaces and lowercase
    words = " ".join(words.split()).lower()

    return words


def _is_char_escaped(code: str, pos: int) -> bool:
    """Check if character at position is escaped by counting preceding backslashes.

    A character is escaped if preceded by an odd number of backslashes.
    Examples:
        "foo\\"  - quote at end is NOT escaped (2 backslashes = even)
        "foo\\"" - second quote IS escaped (1 backslash = odd)
        "foo\\\\" - quote after 4 backslashes is NOT escaped (even)

    Args:
        code: The code string.
        pos: Position of the character to check.

    Returns:
        True if character at pos is escaped (preceded by odd number of backslashes).
    """
    if pos <= 0:
        return False
    num_backslashes = 0
    j = pos - 1
    while j >= 0 and code[j] == '\\':
        num_backslashes += 1
        j -= 1
    return num_backslashes % 2 == 1


def _extract_inline_comments(code: str) -> List[str]:
    """Extract inline comments from code preview.

    Parses # comments and extracts their text for semantic embedding.
    Comments often contain valuable natural language describing intent.
    Uses state machine to avoid false positives from # inside strings
    (e.g., f-string format specifiers like f"{value:#.2f}").

    Handles edge cases:
    - Triple-quoted strings (docstrings)
    - Escaped quotes with proper backslash counting
    - Raw strings (r"...") where backslashes don't escape
    - Shebang lines (#!/usr/bin/env python)

    Args:
        code: The code string to parse.

    Returns:
        List of comment strings (without # prefix).
    """
    comments = []
    in_string = False
    string_char = None  # Single char ('"' or "'") or triple ('"""' or "'''")
    is_raw_string = False  # r"..." strings where backslashes don't escape
    i = 0
    code_len = len(code)

    while i < code_len:
        char = code[i]

        # Track string state
        if char in ('"', "'"):
            if not in_string:
                # Check for raw string prefix (r" or r')
                is_raw = i > 0 and code[i - 1] in ('r', 'R')
                # Also check for f-raw strings (fr" or rf")
                if not is_raw and i > 1:
                    prev_two = code[i - 2:i].lower()
                    is_raw = prev_two in ('fr', 'rf', 'br', 'rb')

                # Check for triple-quoted strings first
                if i + 2 < code_len and code[i:i + 3] in ('"""', "'''"):
                    in_string = True
                    string_char = code[i:i + 3]
                    is_raw_string = is_raw
                    i += 3
                    continue
                # Single-quoted string
                in_string = True
                string_char = char
                is_raw_string = is_raw
            elif in_string:
                # Check for closing quote - must match opener and not be escaped
                # In raw strings, backslashes don't escape quotes
                char_escaped = not is_raw_string and _is_char_escaped(code, i)

                if len(string_char) == 3:
                    # Triple-quoted: check for matching triple quote
                    if i + 2 < code_len and code[i:i + 3] == string_char:
                        # Triple quotes can't be escaped (even in raw strings,
                        # you can't have \" at end of raw string)
                        in_string = False
                        string_char = None
                        is_raw_string = False
                        i += 3
                        continue
                elif char == string_char and not char_escaped:
                    # Single-quoted: matching unescaped quote closes string
                    in_string = False
                    string_char = None
                    is_raw_string = False

        # Only match # outside strings
        elif char == '#' and not in_string:
            # BUG SE-13 FIX: Skip shebang lines properly by advancing to end of line
            # Check for shebang at file start (i == 0) or after newline
            is_shebang = (
                i + 1 < code_len and code[i + 1] == '!' and
                (i == 0 or (i > 0 and code[i - 1] == '\n'))
            )
            if is_shebang:
                # Skip entire shebang line
                end = code.find('\n', i)
                if end == -1:
                    break  # Shebang is only content in file
                i = end
                continue

            # Extract comment text until end of line
            end = code.find('\n', i)
            if end == -1:
                end = code_len
            comment_text = code[i + 1:end].strip()
            # Filter out noise comments (too short or special markers)
            if len(comment_text) > 3 and not comment_text.startswith("!"):
                comments.append(comment_text)
            i = end
            continue

        i += 1

    return comments


def _generate_semantic_description(unit: "EmbeddingUnit") -> str:
    """Generate natural language description when docstring is missing.

    Creates a semantic description from code structure for functions
    that lack docstrings, improving embedding quality.

    Args:
        unit: The EmbeddingUnit to describe.

    Returns:
        Generated natural language description.
    """
    parts = []

    # Parse function name into natural language
    name_words = _parse_identifier_to_words(unit.name)
    if name_words:
        # Create a sentence from the name
        if unit.unit_type == "method":
            parts.append(f"Method that {name_words}")
        elif unit.unit_type == "class":
            parts.append(f"Class for {name_words}")
        else:
            parts.append(f"Function that {name_words}")

    # Extract parameter semantics from signature
    if unit.signature:
        # Extract parameter names from signature
        param_match = re.search(r"\((.*?)\)", unit.signature)
        if param_match:
            params_str = param_match.group(1)
            if params_str and params_str not in ("self", "cls"):
                # Parse parameter names
                param_names = []
                for param in params_str.split(","):
                    param = param.strip()
                    if not param or param in ("self", "cls"):
                        continue
                    # Extract name before : or =
                    name = param.split(":")[0].split("=")[0].strip()
                    if name and name not in ("self", "cls"):
                        param_names.append(_parse_identifier_to_words(name))

                if param_names:
                    parts.append(f"Takes {', '.join(param_names[:5])} as input")

    # Describe complexity in natural language
    if unit.cfg_summary:
        complexity_match = re.search(r"complexity:(\d+)", unit.cfg_summary)
        if complexity_match:
            complexity = int(complexity_match.group(1))
            if complexity == 1:
                parts.append("Simple linear logic")
            elif complexity <= 3:
                parts.append("Contains conditional logic")
            elif complexity <= 7:
                parts.append("Moderate complexity with multiple branches")
            else:
                parts.append("Complex control flow with many decision points")

    # Describe data handling
    if unit.dfg_summary:
        vars_match = re.search(r"vars:(\d+)", unit.dfg_summary)
        if vars_match:
            var_count = int(vars_match.group(1))
            if var_count > 10:
                parts.append("Processes multiple data variables")

    # Extract inline comments for additional context
    if unit.code_preview:
        comments = _extract_inline_comments(unit.code_preview)
        if comments:
            parts.append("Notes: " + "; ".join(comments[:3]))

    return ". ".join(parts) if parts else ""


def build_embedding_text(unit: EmbeddingUnit) -> str:
    """Build rich text for embedding from all 5 layers plus semantic enrichment.

    Creates a single text string containing information from all
    analysis layers, suitable for embedding with a language model.
    Prioritizes natural language over code syntax for better semantic search.

    Includes:
    - Natural language description (docstring or generated)
    - Parsed identifier as natural language
    - Signature with type hints
    - Call graph relationships
    - Semantic tags (crud, validation, async_ops, etc.)
    - Complexity metrics
    - Chunk context (for oversized units)
    - Code preview

    Args:
        unit: The EmbeddingUnit containing code analysis.

    Returns:
        A text string combining all layer information.
    """
    parts = []

    # Header with type and name
    type_str = unit.unit_type if unit.unit_type else "function"
    header = f"{type_str.capitalize()}: {unit.name}"

    # For chunks, include parent context
    if unit.is_chunk():
        header = f"Chunk [{unit.chunk_index + 1}/{unit.chunk_total}] of {unit.parent_name}"

    parts.append(header)

    # Semantic tags - very important for semantic search
    if unit.semantic_tags:
        tags_str = ", ".join(sorted(unit.semantic_tags))
        parts.append(f"Categories: {tags_str}")

    # Primary: Natural language description (most important for semantic search)
    # Use docstring if available, otherwise generate description
    if unit.docstring:
        parts.append(f"Description: {unit.docstring}")
    else:
        # Generate semantic description from code structure
        generated = _generate_semantic_description(unit)
        if generated:
            parts.append(f"Description: {generated}")

    # Parse function name as natural language (helps match semantic queries)
    name_words = _parse_identifier_to_words(unit.name)
    if name_words and name_words != unit.name.lower():
        parts.append(f"Purpose: {name_words}")

    # L1: Signature (contains type hints which have semantic value)
    if unit.signature:
        parts.append(f"Signature: {unit.signature}")

    # Complexity info
    if unit.complexity:
        complexity_parts = []
        if unit.complexity.get("depth", 0) > 3:
            complexity_parts.append("deep nesting")
        if unit.complexity.get("branches", 0) > 5:
            complexity_parts.append("many branches")
        if unit.complexity.get("loops", 0) > 2:
            complexity_parts.append("multiple loops")
        if complexity_parts:
            parts.append(f"Complexity: {', '.join(complexity_parts)}")

    # L2: Call graph with natural language framing
    if unit.calls:
        calls_words = [_parse_identifier_to_words(c) for c in unit.calls[:5]]
        calls_str = ", ".join(filter(None, calls_words))
        if calls_str:
            parts.append(f"Uses: {calls_str}")

    if unit.called_by:
        callers_words = [_parse_identifier_to_words(c) for c in unit.called_by[:5]]
        callers_str = ", ".join(filter(None, callers_words))
        if callers_str:
            parts.append(f"Used by: {callers_str}")

    # L5: Dependencies (module names can have semantic meaning)
    if unit.dependencies:
        parts.append(f"Dependencies: {unit.dependencies}")

    # Code preview - include full context for better semantic matching
    if unit.code_preview:
        # Include comments from code which are natural language
        comments = _extract_inline_comments(unit.code_preview)
        if comments:
            parts.append(f"Code comments: {'; '.join(comments[:20])}")

        # Include code preview
        parts.append(f"Code:\n{unit.code_preview}")

    # Join and ensure final text fits within embedding token limit
    result = "\n".join(parts)
    return truncate_to_tokens(result, MAX_EMBEDDING_TOKENS)


def _is_binary_file(path: Path, sample_size: int = 8192) -> bool:
    """Check if file is binary by looking for null bytes.

    Binary files with code extensions (e.g., compiled .py files, .pyc renamed)
    can produce garbled UTF-8 and silently fail AST parsing. This check
    prevents wasted processing on such files.

    Args:
        path: Path to the file to check.
        sample_size: Number of bytes to sample (default 8KB).

    Returns:
        True if file appears to be binary (contains null bytes).
    """
    try:
        with open(path, "rb") as f:
            sample = f.read(sample_size)
            return b"\x00" in sample
    except Exception:
        return False


# Threshold for switching to parallel processing
MIN_FILES_FOR_PARALLEL = 15


def extract_units_from_project(
    project_path: str, lang: str = "python", respect_ignore: bool = True
) -> List[EmbeddingUnit]:
    """Extract all functions/methods/classes from a project.

    Uses existing TLDR APIs:
    - tldr.api.get_code_structure() for L1 (signatures)
    - tldr.cross_file_calls for L2 (call graph)
    - CFG/DFG extractors for L3/L4 summaries
    - tldr.api.get_imports for L5 (dependencies)

    Args:
        project_path: Path to project root.
        lang: Programming language ("python", "typescript", "go", "rust").
        respect_ignore: If True, respect .tldrignore patterns (default True).

    Returns:
        List of EmbeddingUnit objects with enriched metadata.
    """
    from tldr.api import get_code_structure, build_project_call_graph
    from tldr.tldrignore import load_ignore_patterns, should_ignore

    project = Path(project_path).resolve()
    units = []

    # Get code structure (L1)
    # Use max_results=0 for unlimited files - the default of 100 would truncate large projects
    structure = get_code_structure(str(project), language=lang, max_results=0)

    # Filter ignored files
    if respect_ignore:
        spec = load_ignore_patterns(project)
        structure["files"] = [
            f
            for f in structure.get("files", [])
            if not should_ignore(project / f.get("path", ""), project, spec)
        ]

    # Build call graph (L2)
    try:
        call_graph = build_project_call_graph(str(project), language=lang)

        # Build call/called_by maps
        calls_map = {}  # func -> [called functions]
        called_by_map = {}  # func -> [calling functions]

        for edge in call_graph.edges:
            src_file, src_func, dst_file, dst_func = edge

            # Forward: src calls dst
            if src_func not in calls_map:
                calls_map[src_func] = []
            calls_map[src_func].append(dst_func)

            # Backward: dst is called by src
            if dst_func not in called_by_map:
                called_by_map[dst_func] = []
            called_by_map[dst_func].append(src_func)
    except Exception as e:
        logger.debug(f"Call graph unavailable for project: {e}")
        calls_map = {}
        called_by_map = {}

    # Process files in parallel for better performance
    files = structure.get("files", [])
    max_workers = int(os.environ.get("TLDR_MAX_WORKERS", os.cpu_count() or 4))

    def _get_file_func_names(file_info: Dict[str, Any]) -> Set[str]:
        """Extract all function/method names from a file_info dict."""
        names = set(file_info.get("functions", []))
        for class_info in file_info.get("classes", []):
            if isinstance(class_info, dict):
                names.update(class_info.get("methods", []))
        return names

    def _filter_call_maps(
        func_names: Set[str],
        calls: Dict[str, List[str]],
        called_by: Dict[str, List[str]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Filter call maps to only include entries relevant to given functions.

        Reduces pickle overhead when passing to worker processes.
        """
        filtered_calls = {k: v for k, v in calls.items() if k in func_names}
        filtered_called_by = {k: v for k, v in called_by.items() if k in func_names}
        return filtered_calls, filtered_called_by

    # Use parallel processing if we have enough files to justify overhead
    if len(files) >= MIN_FILES_FOR_PARALLEL and max_workers > 1:
        try:
            # SE-7 fix: Use explicit multiprocessing context for cross-platform safety
            # macOS defaults to 'spawn' since Python 3.8, Windows requires 'spawn'
            # Linux can use 'fork' for better performance (no pickle overhead)
            if sys.platform == "darwin":
                mp_context = mp.get_context("spawn")
            elif sys.platform == "win32":
                mp_context = mp.get_context("spawn")
            else:
                # Linux: 'fork' is faster but requires fork-safe code
                # Our _process_file_for_extraction is fork-safe (no thread-unsafe globals)
                mp_context = mp.get_context("fork")

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                futures = {}
                for file_info in files:
                    # Filter call maps to reduce pickle overhead (Bug MED-005 fix)
                    func_names = _get_file_func_names(file_info)
                    filtered_calls, filtered_called_by = _filter_call_maps(
                        func_names, calls_map, called_by_map
                    )
                    future = executor.submit(
                        _process_file_for_extraction,
                        file_info,
                        str(project),
                        lang,
                        filtered_calls,
                        filtered_called_by,
                    )
                    futures[future] = file_info

                for future in as_completed(futures):
                    file_info = futures[future]
                    try:
                        file_units = future.result(timeout=60)  # 60s per file timeout
                        units.extend(file_units)
                    except Exception as e:
                        logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")
                        # Continue with other files

        except Exception as e:
            # Fallback to sequential if parallel fails
            logger.warning(f"Parallel extraction failed: {e}, falling back to sequential")
            for file_info in files:
                try:
                    file_units = _process_file_for_extraction(
                        file_info, str(project), lang, calls_map, called_by_map
                    )
                    units.extend(file_units)
                except Exception as fe:
                    logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {fe}")
    else:
        # Sequential processing for single file or when parallel is disabled
        for file_info in files:
            try:
                file_units = _process_file_for_extraction(
                    file_info, str(project), lang, calls_map, called_by_map
                )
                units.extend(file_units)
            except Exception as e:
                logger.warning(f"Failed to process {file_info.get('path', 'unknown')}: {e}")

    return units


def _process_file_for_extraction(
    file_info: Dict[str, Any],
    project_path: str,
    lang: str,
    calls_map: Dict[str, List[str]],
    called_by_map: Dict[str, List[str]],
) -> List[EmbeddingUnit]:
    """Process a single file and extract all units. Top-level for pickling.

    This function reads the file ONCE and extracts all information in a single pass,
    avoiding the O(n*m) file read issue where n=files and m=functions.

    Args:
        file_info: Dict with 'path', 'functions', 'classes' from get_code_structure.
        project_path: Absolute path to project root.
        lang: Programming language.
        calls_map: Map of function name -> list of called functions.
        called_by_map: Map of function name -> list of calling functions.

    Returns:
        List of EmbeddingUnit objects for this file.
    """
    units = []
    project = Path(project_path)
    file_path = file_info.get("path", "")
    full_path = project / file_path

    if not full_path.exists():
        return units

    # SE-8 fix: Skip binary files that may have code extensions
    # Binary files produce garbled UTF-8 and silently fail AST parsing
    if _is_binary_file(full_path):
        logger.debug(f"Skipping binary file: {file_path}")
        return units

    try:
        # Read file content ONCE
        content = full_path.read_text(encoding="utf-8-sig", errors="replace")
        lines = content.split('\n')
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return units

    # Build line offset map for character-based extraction
    # This allows us to convert line numbers to character offsets
    line_offsets = [0]  # Line 1 starts at offset 0
    for i, line in enumerate(lines[:-1]):
        line_offsets.append(line_offsets[-1] + len(line) + 1)  # +1 for newline

    def get_char_offset(line_num: int) -> int:
        """Convert 1-based line number to character offset."""
        idx = max(0, min(line_num - 1, len(line_offsets) - 1))
        return line_offsets[idx]

    # Use tree-sitter based extraction for ALL languages (not just Python)
    # This provides consistent line numbers, signatures, and docstrings
    ast_info = {"functions": {}, "classes": {}, "methods": {}}
    all_signatures = {}
    all_docstrings = {}

    try:
        from tldr.ast_extractor import extract_file
        module_info = extract_file(str(full_path))

        # Build lookup dicts from extracted info
        # Note: We extract precise function boundaries using end_line_number (SE-3 fix)
        for func in module_info.functions:
            start_offset = get_char_offset(func.line_number)

            # Use end_line_number for precise extraction (SE-3 fix)
            if func.end_line_number:
                # Extract only this function's code using precise boundaries
                end_offset = get_char_offset(func.end_line_number + 1)
                raw_code = content[start_offset:end_offset]
            else:
                # Fallback: use token limit when end line is unknown
                raw_code, _ = extract_code_by_tokens(content, start_offset, 32000)

            ast_info["functions"][func.name] = {
                "line": func.line_number,
                "end_line": func.end_line_number,
                "code_preview": raw_code,  # Will be chunked if > 6K tokens
            }
            all_signatures[func.name] = func.signature()
            all_docstrings[func.name] = func.docstring or ""

        for cls in module_info.classes:
            class_offset = get_char_offset(cls.line_number)

            # Use end_line_number for precise extraction (SE-3 fix)
            if cls.end_line_number:
                end_offset = get_char_offset(cls.end_line_number + 1)
                raw_code = content[class_offset:end_offset]
            else:
                # Fallback: use token limit when end line is unknown
                raw_code, _ = extract_code_by_tokens(content, class_offset, 32000)

            ast_info["classes"][cls.name] = {
                "line": cls.line_number,
                "end_line": cls.end_line_number,
                "code_preview": raw_code,  # Will be chunked if > 6K tokens
            }

            # Process methods within each class
            for method in cls.methods:
                method_key = f"{cls.name}.{method.name}"
                start_offset = get_char_offset(method.line_number)

                # Use end_line_number for precise extraction (SE-3 fix)
                if method.end_line_number:
                    end_offset = get_char_offset(method.end_line_number + 1)
                    raw_code = content[start_offset:end_offset]
                else:
                    # Fallback: use token limit when end line is unknown
                    raw_code, _ = extract_code_by_tokens(content, start_offset, 32000)

                ast_info["methods"][method_key] = {
                    "line": method.line_number,
                    "end_line": method.end_line_number,
                    "code_preview": raw_code,  # Will be chunked if > 6K tokens
                }
                all_signatures[method_key] = method.signature()
                all_docstrings[method_key] = method.docstring or ""

    except Exception as e:
        logger.debug(f"AST extraction failed for {file_path}: {e}")

    # Get dependencies (imports) - single call
    dependencies = ""
    try:
        from tldr.api import get_imports
        imports = get_imports(str(full_path), language=lang)
        modules = [imp.get("module", "") for imp in imports[:5] if imp.get("module")]
        dependencies = ", ".join(modules)
    except Exception as e:
        logger.debug(f"Failed to extract imports from {file_path}: {e}")

    # Pre-compute CFG/DFG for all functions at once
    cfg_cache = {}
    dfg_cache = {}

    # Languages with CFG/DFG extractor support
    SUPPORTED_CFG_LANGUAGES = {"python", "typescript", "go", "rust"}

    if lang in SUPPORTED_CFG_LANGUAGES:
        # Import extractors for this language (once per file, not per function)
        cfg_extractor = None
        dfg_extractor = None

        try:
            if lang == "python":
                from tldr.cfg_extractor import extract_python_cfg as cfg_extractor
                from tldr.dfg_extractor import extract_python_dfg as dfg_extractor
            elif lang == "typescript":
                from tldr.cfg_extractor import extract_typescript_cfg as cfg_extractor
                from tldr.dfg_extractor import extract_typescript_dfg as dfg_extractor
            elif lang == "go":
                from tldr.cfg_extractor import extract_go_cfg as cfg_extractor
                from tldr.dfg_extractor import extract_go_dfg as dfg_extractor
            elif lang == "rust":
                from tldr.cfg_extractor import extract_rust_cfg as cfg_extractor
                from tldr.dfg_extractor import extract_rust_dfg as dfg_extractor
        except ImportError as e:
            logger.debug(f"CFG/DFG extractors not available for {lang}: {e}")

        if cfg_extractor or dfg_extractor:
            # Get all function names we need to process
            all_func_names = list(file_info.get("functions", []))
            for class_info in file_info.get("classes", []):
                if isinstance(class_info, dict):
                    all_func_names.extend(class_info.get("methods", []))

            for func_name in all_func_names:
                if cfg_extractor:
                    try:
                        cfg = cfg_extractor(content, func_name)
                        cfg_cache[func_name] = f"complexity:{cfg.cyclomatic_complexity}, blocks:{len(cfg.blocks)}"
                    except Exception:
                        cfg_cache[func_name] = ""

                if dfg_extractor:
                    try:
                        dfg = dfg_extractor(content, func_name)
                        var_names = set(ref.name for ref in dfg.var_refs)
                        dfg_cache[func_name] = f"vars:{len(var_names)}, def-use chains:{len(dfg.dataflow_edges)}"
                    except Exception:
                        dfg_cache[func_name] = ""

    # Process functions - create units, enrich, and chunk if needed
    for func_name in file_info.get("functions", []):
        func_info = ast_info.get("functions", {}).get(func_name, {})
        unit = EmbeddingUnit(
            name=func_name,
            qualified_name=f"{file_path}::{func_name}",
            file=file_path,
            line=func_info.get("line", 1),
            language=lang,
            unit_type="function",
            signature=all_signatures.get(func_name, f"def {func_name}(...)"),
            docstring=all_docstrings.get(func_name, ""),
            calls=calls_map.get(func_name, [])[:5],
            called_by=called_by_map.get(func_name, [])[:5],
            cfg_summary=cfg_cache.get(func_name, ""),
            dfg_summary=dfg_cache.get(func_name, ""),
            dependencies=dependencies,
            code_preview=func_info.get("code_preview", ""),
        )
        # Enrich with semantic patterns and complexity
        enrich_unit(unit)
        # Chunk if oversized
        chunked = chunk_unit(unit)
        units.extend(chunked)

    # Process classes
    for class_info in file_info.get("classes", []):
        if isinstance(class_info, dict):
            class_name = class_info.get("name", "")
            methods = class_info.get("methods", [])
        else:
            class_name = class_info
            methods = []

        class_ast = ast_info.get("classes", {}).get(class_name, {})
        class_line = class_ast.get("line", 1)
        class_code = class_ast.get("code_preview", "")

        # Add class itself with code preview
        unit = EmbeddingUnit(
            name=class_name,
            qualified_name=f"{file_path}::{class_name}",
            file=file_path,
            line=class_line,
            language=lang,
            unit_type="class",
            signature=f"class {class_name}",
            docstring="",
            calls=[],
            called_by=[],
            cfg_summary="",
            dfg_summary="",
            dependencies=dependencies,
            code_preview=class_code,
        )
        # Enrich with semantic patterns and complexity
        enrich_unit(unit)
        # Chunk if oversized (large classes get split)
        chunked = chunk_unit(unit)
        units.extend(chunked)

        # Add methods
        for method in methods:
            method_key = f"{class_name}.{method}"
            method_info = ast_info.get("methods", {}).get(method_key, {})

            unit = EmbeddingUnit(
                name=method,
                qualified_name=f"{file_path}::{method_key}",
                file=file_path,
                line=method_info.get("line", 1),
                language=lang,
                unit_type="method",
                signature=all_signatures.get(method_key, f"def {method}(self, ...)"),
                docstring=all_docstrings.get(method_key, ""),
                calls=calls_map.get(method, [])[:5],
                called_by=called_by_map.get(method, [])[:5],
                cfg_summary=cfg_cache.get(method, ""),
                dfg_summary=dfg_cache.get(method, ""),
                dependencies=dependencies,
                code_preview=method_info.get("code_preview", ""),
            )
            # Enrich with semantic patterns and complexity
            enrich_unit(unit)
            # Chunk if oversized
            chunked = chunk_unit(unit)
            units.extend(chunked)

    return units
