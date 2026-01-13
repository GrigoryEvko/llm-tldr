#!/usr/bin/env python3
"""
TLDR-Code CLI - Token-efficient code analysis for LLMs.

Usage:
    tldr tree [path]                    Show file tree
    tldr structure [path]               Show code structure (codemaps)
    tldr search <pattern> [path]        Search files for pattern
    tldr extract <file>                 Extract full file info
    tldr context <entry> [--project]    Get relevant context for LLM
    tldr cfg <file> <function>          Control flow graph
    tldr dfg <file> <function>          Data flow graph
    tldr slice <file> <func> <line>     Program slice
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Fix for Windows: Explicitly import tree-sitter bindings early to prevent
# silent DLL loading failures when running as a console script entry point.
if os.name == 'nt':
    try:
        import tree_sitter
        import tree_sitter_python
        import tree_sitter_javascript
        import tree_sitter_typescript
    except ImportError:
        pass

from . import __version__
from .languages import EXTENSION_TO_LANGUAGE, detect_language_with_default


def _validate_path(
    path: str,
    must_exist: bool = True,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """Validate a path with consistent error handling.

    Args:
        path: Path string to validate
        must_exist: If True, path must exist on filesystem
        must_be_file: If True, path must be a file (not directory)
        must_be_dir: If True, path must be a directory (not file)

    Returns:
        Path object if valid

    Raises:
        SystemExit: If validation fails (prints error to stderr)
    """
    p = Path(path)
    if must_exist and not p.exists():
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)
    if must_be_file and not p.is_file():
        print(f"Error: Not a file: {path}", file=sys.stderr)
        sys.exit(1)
    if must_be_dir and not p.is_dir():
        print(f"Error: Not a directory: {path}", file=sys.stderr)
        sys.exit(1)
    return p


def _get_subprocess_detach_kwargs() -> dict:
    """Get platform-specific kwargs for detaching subprocess."""
    import subprocess

    if os.name == "nt":  # Windows
        # CREATE_NEW_PROCESS_GROUP is Windows-only; use getattr for type safety
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    else:  # Unix (Mac/Linux)
        return {"start_new_session": True}


def _find_project_root(start_path: str | Path) -> Path:
    """Find the project root by looking for marker directories.

    Walks up the directory tree from start_path looking for .tldr/ or .git/
    markers that indicate the project root. This allows subdirectory paths
    to work correctly with cached call graphs.

    Args:
        start_path: Starting directory to search from

    Returns:
        Project root Path. Returns the resolved start_path if no markers found.
    """
    current = Path(start_path).resolve()

    # If start_path is a file, start from its parent directory
    if current.is_file():
        current = current.parent

    # First pass: look for .git/ (definitive project root marker)
    # .git/ is preferred over .tldr/ since .git/ definitively marks
    # the repository root, while .tldr/ caches can exist in subdirectories.
    search = current
    while search != search.parent:  # Stop at filesystem root
        if (search / ".git").is_dir():
            return search
        search = search.parent

    # Second pass: fall back to .tldr/ if no .git/ found
    search = current
    while search != search.parent:
        if (search / ".tldr").is_dir():
            return search
        search = search.parent

    # No markers found - return the original resolved path
    return Path(start_path).resolve()


def _detect_project_language(path: str) -> str:
    """Auto-detect dominant language for a project.

    Detection priority:
    1. Check .tldr/cache/call_graph.json for cached "languages" field
    2. Scan project files and count by extension

    Args:
        path: Project root directory

    Returns:
        Detected language name (defaults to 'python' if detection fails)
    """
    project = Path(path).resolve()

    # Priority 1: Check cached call graph for language info
    cache_file = project / ".tldr" / "cache" / "call_graph.json"
    if cache_file.exists():
        try:
            cache_data = json.loads(cache_file.read_text())
            languages = cache_data.get("languages", [])
            if languages:
                # Return first language from cache (primary language used for indexing)
                return languages[0]
        except (json.JSONDecodeError, OSError):
            pass  # Cache invalid or unreadable, fall through to scanning

    # Priority 2: Scan project files and count by extension
    extension_counts: dict[str, int] = {}

    # Prefer src/lib directories if they exist, otherwise scan project root
    scan_dir = project
    if (project / "src").is_dir():
        scan_dir = project / "src"
    elif (project / "lib").is_dir():
        scan_dir = project / "lib"

    try:
        # Scan files (limit depth to avoid slow scanning of large node_modules etc)
        for item in scan_dir.rglob("*"):
            # Skip hidden directories and common vendor directories
            parts = item.relative_to(scan_dir).parts
            if any(
                p.startswith(".") or p in ("node_modules", "__pycache__", "vendor", "dist", "build")
                for p in parts
            ):
                continue

            if item.is_file():
                ext = item.suffix.lower()
                if ext in EXTENSION_TO_LANGUAGE:
                    lang = EXTENSION_TO_LANGUAGE[ext]
                    extension_counts[lang] = extension_counts.get(lang, 0) + 1
    except OSError:
        pass

    if extension_counts:
        # Return language with most files
        return max(extension_counts, key=lambda k: extension_counts[k])

    # Default fallback
    return "python"


def _show_first_run_tip():
    """Show a one-time tip about Swift support on first run."""
    marker = Path.home() / ".tldr_first_run"
    if marker.exists():
        return

    # Check if Swift is already installed
    import importlib.util

    if importlib.util.find_spec("tree_sitter_swift") is not None:
        # Swift already works, no tip needed
        marker.touch()
        return

    # Show tip
    import sys

    print("Tip: For Swift support, run: python -m tldr.install_swift", file=sys.stderr)
    print("     (This message appears once)", file=sys.stderr)
    print(file=sys.stderr)

    marker.touch()


def main():
    _show_first_run_tip()
    parser = argparse.ArgumentParser(
        prog="tldr",
        description="Token-efficient code analysis for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Version: %(prog)s """
        + __version__
        + """

Examples:
    tldr tree src/                      # File tree for src/
    tldr structure . --lang python      # Code structure for Python files
    tldr search "def process" .         # Search for pattern
    tldr extract src/main.py            # Full file analysis
    tldr context main --project .       # LLM context starting from main()
    tldr cfg src/main.py process        # Control flow for process()
    tldr slice src/main.py func 42      # Lines affecting line 42

Ignore Patterns:
    TLDR respects .tldrignore files (gitignore syntax).
    First run creates .tldrignore with sensible defaults.
    Use --no-ignore to bypass ignore patterns.

Daemon:
    TLDR runs a per-project daemon for fast repeated queries.
    - Socket: /tmp/tldr-{hash}.sock (hash from project path)
    - Auto-shutdown: 30 minutes idle
    - Memory: ~50-100MB base, +500MB-1GB with semantic search

    Start explicitly:  tldr daemon start
    Check status:      tldr daemon status
    Stop:              tldr daemon stop

Semantic Search:
    First run downloads embedding model (1.3GB default).
    Use --model all-MiniLM-L6-v2 for smaller 80MB model.
    Set TLDR_AUTO_DOWNLOAD=1 to skip download prompts.
        """,
    )

    # Global flags
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--no-ignore",
        action="store_true",
        help="Ignore .tldrignore patterns (include all files)",
    )

    # Shell completion support
    try:
        import shtab

        shtab.add_argument_to(parser, ["--print-completion", "-s"])
    except ImportError:
        pass  # shtab is optional

    subparsers = parser.add_subparsers(dest="command", required=True)

    # tldr tree [path]
    tree_p = subparsers.add_parser(
        "tree",
        help="Show file tree",
        description="Display the file tree structure of a directory in JSON format.",
        epilog="Example: tldr tree src/ --ext .py .ts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tree_p.add_argument("path", nargs="?", default=".", help="Directory to scan")
    tree_p.add_argument(
        "--ext", nargs="+", help="Filter by extensions (e.g., --ext .py .ts)"
    )
    tree_p.add_argument(
        "--show-hidden", action="store_true", help="Include hidden files"
    )

    # tldr structure [path]
    struct_p = subparsers.add_parser(
        "structure",
        help="Show code structure (codemaps)",
        description="Extract functions, classes, and methods from source files.",
        epilog="Example: tldr structure src/ --lang python --max 100",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    struct_p.add_argument("path", nargs="?", default=".", help="Directory to analyze")
    struct_p.add_argument(
        "--lang",
        default=None,
        choices=[
            "python",
            "typescript",
            "javascript",
            "go",
            "rust",
            "java",
            "c",
            "cpp",
            "ruby",
            "php",
            "kotlin",
            "swift",
            "csharp",
            "scala",
            "lua",
            "elixir",
        ],
        help="Language to analyze (auto-detected if not specified)",
    )
    struct_p.add_argument(
        "--max", type=int, default=50, help="Max files to analyze (default: 50)"
    )

    # tldr search <pattern> [path]
    search_p = subparsers.add_parser(
        "search",
        help="Search files for pattern",
        description="Search files for a regex pattern with optional context lines.",
        epilog="Example: tldr search 'def process' src/ --ext .py -C 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search_p.add_argument("pattern", help="Regex pattern to search")
    search_p.add_argument("path", nargs="?", default=".", help="Directory to search")
    search_p.add_argument("--ext", nargs="+", help="Filter by extensions")
    search_p.add_argument(
        "-C", "--context", type=int, default=0, help="Context lines around match"
    )
    search_p.add_argument(
        "--max", type=int, default=100, help="Max results (default: 100, 0=unlimited)"
    )
    search_p.add_argument(
        "--max-files",
        type=int,
        default=10000,
        help="Max files to scan (default: 10000)",
    )

    # tldr extract <file> [--class X] [--function Y] [--method Class.method]
    extract_p = subparsers.add_parser(
        "extract",
        help="Extract full file info",
        description="Extract complete AST info from a file: functions, classes, methods, docstrings.",
        epilog="Examples:\n  tldr extract src/main.py\n  tldr extract src/api.py --class UserController\n  tldr extract src/api.py --method UserController.get_user",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    extract_p.add_argument("file", help="File to analyze")
    extract_p.add_argument(
        "--class", dest="filter_class", help="Filter to specific class"
    )
    extract_p.add_argument(
        "--function", dest="filter_function", help="Filter to specific function"
    )
    extract_p.add_argument(
        "--method",
        dest="filter_method",
        help="Filter to specific method (Class.method)",
    )

    # tldr context <entry>
    ctx_p = subparsers.add_parser(
        "context",
        help="Get relevant context for LLM",
        description="Build LLM-ready context by following the call graph from an entry point.",
        epilog="Examples:\n  tldr context main --project . --depth 3\n  tldr context UserController.get_user --project src/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ctx_p.add_argument("entry", help="Entry point (function_name or Class.method)")
    ctx_p.add_argument("--project", default=".", help="Project root directory")
    ctx_p.add_argument("--depth", type=int, default=2, help="Call depth (default: 2)")
    ctx_p.add_argument(
        "--lang",
        default=None,
        choices=[
            "python",
            "typescript",
            "javascript",
            "go",
            "rust",
            "java",
            "c",
            "cpp",
            "ruby",
            "php",
            "kotlin",
            "swift",
            "csharp",
            "scala",
            "lua",
            "elixir",
        ],
        help="Language (auto-detected if not specified)",
    )

    # tldr cfg <file> <function>
    cfg_p = subparsers.add_parser(
        "cfg",
        help="Control flow graph",
        description="Generate a control flow graph (CFG) for a function, showing branches and loops.",
        epilog="Example: tldr cfg src/processor.py process_data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cfg_p.add_argument("file", help="Source file")
    cfg_p.add_argument("function", help="Function name")
    cfg_p.add_argument(
        "--lang",
        default=None,
        help="Language (auto-detected from extension if not specified)",
    )

    # tldr dfg <file> <function>
    dfg_p = subparsers.add_parser(
        "dfg",
        help="Data flow graph",
        description="Generate a data flow graph (DFG) for a function, showing variable dependencies.",
        epilog="Example: tldr dfg src/processor.py process_data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dfg_p.add_argument("file", help="Source file")
    dfg_p.add_argument("function", help="Function name")
    dfg_p.add_argument(
        "--lang",
        default=None,
        help="Language (auto-detected from extension if not specified)",
    )

    # tldr slice <file> <function> <line>
    slice_p = subparsers.add_parser(
        "slice",
        help="Program slice",
        description="Compute a program slice: find all lines that affect (backward) or are affected by (forward) a given line.",
        epilog="Examples:\n  tldr slice src/main.py process 42              # What affects line 42?\n  tldr slice src/main.py process 42 --direction forward  # What does line 42 affect?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    slice_p.add_argument("file", help="Source file")
    slice_p.add_argument("function", help="Function name")
    slice_p.add_argument("line", type=int, help="Line number to slice from")
    slice_p.add_argument(
        "--direction",
        default="backward",
        choices=["backward", "forward"],
        help="Slice direction",
    )
    slice_p.add_argument("--var", help="Variable to track (optional)")
    slice_p.add_argument(
        "--lang",
        default=None,
        help="Language (auto-detected from extension if not specified)",
    )

    # tldr calls <path>
    calls_p = subparsers.add_parser(
        "calls",
        help="Build cross-file call graph",
        description="Build a project-wide call graph showing which functions call which.",
        epilog="Example: tldr calls src/ --lang python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    calls_p.add_argument("path", nargs="?", default=".", help="Project root")
    calls_p.add_argument("--lang", default=None, help="Language (auto-detected if not specified)")

    # tldr impact <func> [path]
    impact_p = subparsers.add_parser(
        "impact",
        help="Find all callers of a function (reverse call graph)",
        description="Analyze impact: find all functions that call a given function (transitively).\nUseful before refactoring to understand what will be affected.",
        epilog="Examples:\n  tldr impact process_data src/ --depth 5\n  tldr impact get_user . --file api",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    impact_p.add_argument("func", help="Function name to find callers of")
    impact_p.add_argument("path", nargs="?", default=".", help="Project root")
    impact_p.add_argument("--depth", type=int, default=3, help="Max depth (default: 3)")
    impact_p.add_argument("--file", help="Filter by file containing this string")
    impact_p.add_argument("--lang", default=None, help="Language (auto-detected if not specified)")

    # tldr dead [path]
    dead_p = subparsers.add_parser(
        "dead",
        help="Find unreachable (dead) code",
        description="Find functions that are never called (dead code).\nExcludes common entry points like main, test_*, cli, etc.",
        epilog="Examples:\n  tldr dead src/\n  tldr dead . --entry cli main",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    dead_p.add_argument("path", nargs="?", default=".", help="Project root")
    dead_p.add_argument(
        "--entry", nargs="*", default=[], help="Additional entry point patterns"
    )
    dead_p.add_argument("--lang", default=None, help="Language (auto-detected if not specified)")

    # tldr arch [path]
    arch_p = subparsers.add_parser(
        "arch",
        help="Detect architectural layers from call patterns",
        description="Detect architectural layers (entry, middle, leaf) from call patterns.\nIdentifies circular dependencies.",
        epilog="Example: tldr arch src/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arch_p.add_argument("path", nargs="?", default=".", help="Project root")
    arch_p.add_argument("--lang", default=None, help="Language (auto-detected if not specified)")

    # tldr imports <file>
    imports_p = subparsers.add_parser(
        "imports",
        help="Parse imports from a source file",
        description="Parse all import statements from a source file.\nReturns JSON with module names, imported names, and aliases.",
        epilog="Example: tldr imports src/main.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    imports_p.add_argument("file", help="Source file to analyze")
    imports_p.add_argument(
        "--lang",
        default=None,
        help="Language (auto-detected from extension if not specified)",
    )

    # tldr importers <module> [path]
    importers_p = subparsers.add_parser(
        "importers",
        help="Find all files that import a module (reverse import lookup)",
        description="Find all files that import a given module.\nComplements 'tldr impact' which tracks function calls.",
        epilog="Examples:\n  tldr importers json src/\n  tldr importers UserController .",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    importers_p.add_argument("module", help="Module name to search for importers")
    importers_p.add_argument("path", nargs="?", default=".", help="Project root")
    importers_p.add_argument("--lang", default=None, help="Language (auto-detected if not specified)")

    # tldr change-impact [files...]
    change_impact_p = subparsers.add_parser(
        "change-impact",
        help="Find tests affected by changed files",
        description="Find which tests to run based on changed files.\nUses call graph + import analysis to find affected tests.",
        epilog="Examples:\n  tldr change-impact                    # Auto-detect changes\n  tldr change-impact src/api.py         # Explicit files\n  tldr change-impact --git               # Use git diff\n  tldr change-impact --run               # Actually run affected tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    change_impact_p.add_argument(
        "files",
        nargs="*",
        help="Files to analyze (default: auto-detect from session/git)",
    )
    change_impact_p.add_argument(
        "--session", action="store_true", help="Use session-modified files (dirty_flag)"
    )
    change_impact_p.add_argument(
        "--git", action="store_true", help="Use git diff to find changed files"
    )
    change_impact_p.add_argument(
        "--git-base", default="HEAD~1", help="Git ref to diff against (default: HEAD~1)"
    )
    change_impact_p.add_argument("--lang", default=None, help="Language (auto-detected if not specified)")
    change_impact_p.add_argument(
        "--depth", type=int, default=5, help="Max call graph depth (default: 5)"
    )
    change_impact_p.add_argument(
        "--run", action="store_true", help="Actually run the affected tests"
    )
    change_impact_p.add_argument(
        "--project", "-p",
        default=".",
        help="Project directory to analyze (default: current directory)"
    )

    # tldr diagnostics <file|path>
    diag_p = subparsers.add_parser(
        "diagnostics",
        help="Get type and lint diagnostics",
        description="Run type checker (pyright) and linter (ruff) on code.\nReturns structured errors. Use before tests to catch type errors early.",
        epilog="Examples:\n  tldr diagnostics src/main.py\n  tldr diagnostics . --project --format text\n  tldr diagnostics src/ --no-lint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    diag_p.add_argument("target", help="File or project directory to check")
    diag_p.add_argument(
        "--project",
        action="store_true",
        help="Check entire project (default: single file)",
    )
    diag_p.add_argument(
        "--no-lint", action="store_true", help="Skip linter, only run type checker"
    )
    diag_p.add_argument(
        "--format", choices=["json", "text"], default="json", help="Output format"
    )
    diag_p.add_argument("--lang", default=None, help="Override language detection")

    # tldr warm <path>
    warm_p = subparsers.add_parser(
        "warm",
        help="Pre-build call graph cache for faster queries",
        description="Pre-build the call graph cache to speed up subsequent queries.\nRun this once per project before using impact/dead/calls.",
        epilog="Examples:\n  tldr warm . --lang python\n  tldr warm src/ --lang all --background",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    warm_p.add_argument("path", help="Project root directory")
    warm_p.add_argument(
        "--background", action="store_true", help="Build in background process"
    )
    warm_p.add_argument(
        "--lang",
        default="python",
        choices=["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "all"],
        help="Language (use 'all' for multi-language)",
    )

    # tldr semantic index <path> / tldr semantic search <query>
    semantic_p = subparsers.add_parser(
        "semantic",
        help="Semantic code search using embeddings",
        description="Semantic code search using embeddings.\nRequires: tldr semantic index . (first run downloads 1.3GB model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    semantic_sub = semantic_p.add_subparsers(dest="action", required=True)

    # tldr semantic index [path]
    index_p = semantic_sub.add_parser(
        "index",
        help="Build semantic index for project",
        description="Build unified semantic index for a project (auto-detects all languages).\nFirst run downloads embedding model (1.3GB default, 80MB for MiniLM).",
        epilog="Examples:\n  tldr semantic index .                    # Index all languages\n  tldr semantic index . --lang python      # Index only Python\n  tldr semantic index . --model all-MiniLM-L6-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    index_p.add_argument("path", nargs="?", default=".", help="Project root")
    index_p.add_argument(
        "--lang",
        default="all",
        choices=["python", "typescript", "javascript", "go", "rust", "java", "c", "cpp", "all"],
        help="Language to index (default: all = auto-detect)",
    )
    index_p.add_argument(
        "--model",
        default=None,
        help="Embedding model: bge-large-en-v1.5 (1.3GB, default) or all-MiniLM-L6-v2 (80MB)",
    )
    index_p.add_argument(
        "--backend",
        choices=["tei", "sentence_transformers", "auto"],
        default="auto",
        help="Inference backend (default: auto - prefers TEI server if available)",
    )
    index_p.add_argument(
        "--dimension",
        type=int,
        help="MRL embedding dimension (Qwen3 only). Smaller = faster search, less memory.",
    )

    # tldr semantic search <query> [path]
    semantic_search_p = semantic_sub.add_parser(
        "search",
        help="Search semantically",
        description="Search code using natural language queries.\nSearches unified index (all languages). Requires: tldr semantic index .",
        epilog="Examples:\n  tldr semantic search 'authentication logic' .\n  tldr semantic search 'database connection' src/ --k 10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    semantic_search_p.add_argument("query", help="Natural language query")
    semantic_search_p.add_argument("path", nargs="?", default=".", help="Project root")
    semantic_search_p.add_argument("--k", type=int, default=5, help="Number of results")
    semantic_search_p.add_argument(
        "--expand", action="store_true", help="Include call graph expansion"
    )
    semantic_search_p.add_argument(
        "--model",
        default=None,
        help="Embedding model (uses index model if not specified)",
    )
    semantic_search_p.add_argument(
        "--backend",
        choices=["tei", "sentence_transformers", "auto"],
        default="auto",
        help="Inference backend (tei = text-embeddings-inference server)",
    )
    semantic_search_p.add_argument(
        "--task",
        choices=["code_search", "code_retrieval", "semantic_search", "default"],
        default="code_search",
        help="Query task type for instruction-aware models (default: code_search)",
    )
    semantic_search_p.add_argument(
        "--force-reload",
        action="store_true",
        help="Bypass index cache and reload from disk",
    )

    # tldr semantic warmup
    warmup_p = semantic_sub.add_parser(
        "warmup",
        help="Pre-load and warm up embedding model for faster first query",
        description="Pre-load the embedding model into memory (and GPU if available).\nThis speeds up the first semantic search by eliminating model load time.",
        epilog="Example: tldr semantic warmup --model Qwen3-Embedding-0.6B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    warmup_p.add_argument(
        "--model",
        default=None,
        help="Model name (default: Qwen3-Embedding-0.6B)",
    )

    # tldr semantic unload
    unload_p = semantic_sub.add_parser(
        "unload",
        help="Unload model and free GPU memory",
        description="Unload the embedding model from memory to free GPU/RAM.\nUseful when you need to free resources for other tasks.",
        epilog="Example: tldr semantic unload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # tldr semantic cache (clear|stats|invalidate)
    cache_p = semantic_sub.add_parser(
        "cache",
        help="Index cache management",
        description="Manage the semantic index cache.\nCached indexes speed up repeated searches but consume memory.",
        epilog="Subcommands:\n  clear      Clear all cached indexes\n  stats      Show cache statistics\n  invalidate Invalidate index for a specific project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cache_sub = cache_p.add_subparsers(dest="cache_action", required=True)

    cache_sub.add_parser(
        "clear",
        help="Clear all cached indexes",
        description="Remove all indexes from the in-memory cache.\nFrees memory but next search will need to reload from disk.",
    )
    cache_sub.add_parser(
        "stats",
        help="Show cache statistics",
        description="Display current cache usage: number of cached projects, memory usage, limits.",
    )

    cache_invalidate_p = cache_sub.add_parser(
        "invalidate",
        help="Invalidate index for a specific project",
        description="Remove a specific project's index from cache.\nUseful after manual index rebuild or when freeing memory for one project.",
    )
    cache_invalidate_p.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project path to invalidate (default: current directory)",
    )

    # tldr semantic device
    device_p = semantic_sub.add_parser(
        "device",
        help="Show compute device and backend info",
        description="Show detected compute device (CUDA, MPS, CPU) and available backends.",
        epilog="Example: tldr semantic device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # tldr semantic memory
    memory_p = semantic_sub.add_parser(
        "memory",
        help="Show GPU/memory statistics",
        description="Show GPU memory usage and model statistics.\nRequires model to be loaded (run semantic search first).",
        epilog="Example: tldr semantic memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # tldr daemon start/stop/status/query
    daemon_p = subparsers.add_parser(
        "daemon",
        help="Daemon management subcommands",
        description="Manage the TLDR daemon for faster repeated queries.\nDaemon runs in background, auto-shuts down after 30 min idle.",
        epilog="Subcommands:\n  start   Start daemon for current project\n  stop    Stop daemon gracefully\n  status  Check if daemon is running\n  query   Send raw command to daemon\n  notify  Notify daemon of file change",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    daemon_sub = daemon_p.add_subparsers(dest="action", required=True)

    # tldr daemon start [--project PATH]
    daemon_start_p = daemon_sub.add_parser(
        "start",
        help="Start daemon for project (background)",
        description="Start the TLDR daemon for faster repeated queries.",
        epilog="Example: tldr daemon start -p /path/to/project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    daemon_start_p.add_argument(
        "--project", "-p", default=".", help="Project path (default: current directory)"
    )

    # tldr daemon stop [--project PATH]
    daemon_stop_p = daemon_sub.add_parser("stop", help="Stop daemon gracefully")
    daemon_stop_p.add_argument(
        "--project", "-p", default=".", help="Project path (default: current directory)"
    )

    # tldr daemon status [--project PATH]
    daemon_status_p = daemon_sub.add_parser("status", help="Check if daemon running")
    daemon_status_p.add_argument(
        "--project", "-p", default=".", help="Project path (default: current directory)"
    )

    # tldr daemon query CMD [--project PATH]
    daemon_query_p = daemon_sub.add_parser(
        "query", help="Send raw JSON command to daemon"
    )
    daemon_query_p.add_argument(
        "cmd", help="Command to send (e.g., ping, status, search)"
    )
    daemon_query_p.add_argument(
        "--project", "-p", default=".", help="Project path (default: current directory)"
    )

    # tldr daemon notify FILE [--project PATH]
    daemon_notify_p = daemon_sub.add_parser(
        "notify", help="Notify daemon of file change (triggers reindex at threshold)"
    )
    daemon_notify_p.add_argument("file", help="Path to changed file")
    daemon_notify_p.add_argument(
        "--project", "-p", default=".", help="Project path (default: current directory)"
    )

    # tldr doctor [--install LANG]
    doctor_p = subparsers.add_parser(
        "doctor",
        help="Check and install diagnostic tools (type checkers, linters)",
        description="Check which diagnostic tools (type checkers, linters) are installed.\nCan auto-install missing tools for supported languages.",
        epilog="Examples:\n  tldr doctor                 # Check all tools\n  tldr doctor --json          # Output as JSON\n  tldr doctor --install python  # Install Python tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor_p.add_argument(
        "--install",
        metavar="LANG",
        help="Install missing tools for language (e.g., python, go)",
    )
    doctor_p.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from .api import (
        build_project_call_graph,
        extract_file,
        get_cfg_context,
        get_code_structure,
        get_dfg_context,
        get_file_tree,
        get_imports,
        get_relevant_context,
        get_slice,
        scan_project_files,
        search as api_search,
    )
    from .analysis import (
        analyze_architecture,
        analyze_dead_code,
        analyze_impact,
    )
    from .dirty_flag import is_dirty, get_dirty_files, clear_dirty
    from .patch import patch_call_graph
    from .cross_file_calls import ProjectCallGraph

    def _get_or_build_graph(project_path, lang, build_fn):
        """Get cached graph with incremental patches, or build fresh.

        This implements P4 incremental updates:
        1. If no cache exists, do full build
        2. If cache exists but no dirty files, load cache
        3. If cache exists with dirty files, patch incrementally
        """
        import time

        project = Path(project_path).resolve()
        cache_dir = project / ".tldr" / "cache"
        cache_file = cache_dir / "call_graph.json"

        # Check if we have a cached graph
        if cache_file.exists():
            try:
                cache_data = json.loads(cache_file.read_text())
                
                # Validate cache language compatibility
                cache_langs = cache_data.get("languages", [])
                if cache_langs and lang not in cache_langs and lang != "all":
                    # Cache was built with different languages; rebuild
                    raise ValueError("Cache language mismatch")
                
                # Reconstruct graph from cache
                graph = ProjectCallGraph()
                for e in cache_data.get("edges", []):
                    graph.add_edge(
                        e["from_file"], e["from_func"], e["to_file"], e["to_func"]
                    )

                # Check for dirty files
                if is_dirty(project):
                    dirty_files = get_dirty_files(project)
                    # Patch incrementally for each dirty file
                    for rel_file in dirty_files:
                        abs_file = project / rel_file
                        if abs_file.exists():
                            graph = patch_call_graph(
                                graph, str(abs_file), str(project), lang=lang
                            )

                    # Update cache with patched graph
                    cache_data = {
                        "edges": [
                            {
                                "from_file": e[0],
                                "from_func": e[1],
                                "to_file": e[2],
                                "to_func": e[3],
                            }
                            for e in graph.edges
                        ],
                        "languages": cache_langs if cache_langs else [lang],
                        "timestamp": time.time(),
                    }
                    cache_file.write_text(json.dumps(cache_data, indent=2))

                    # Clear dirty flag
                    clear_dirty(project)

                return graph
            except (json.JSONDecodeError, KeyError, ValueError):
                # Invalid cache or language mismatch, fall through to fresh build
                pass

        # No cache or invalid cache - do fresh build
        graph = build_fn(project_path, language=lang)

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "edges": [
                {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                for e in graph.edges
            ],
            "languages": [lang],
            "timestamp": time.time(),
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))

        # Clear any dirty flag since we just rebuilt
        clear_dirty(project)

        return graph

    try:
        if args.command == "tree":
            _validate_path(args.path, must_exist=True, must_be_dir=True)
            ext = set(args.ext) if args.ext else None
            result = get_file_tree(
                args.path, extensions=ext, exclude_hidden=not args.show_hidden
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "structure":
            _validate_path(args.path, must_exist=True, must_be_dir=True)
            # Auto-detect language if not specified
            lang = args.lang or _detect_project_language(args.path)
            respect_ignore = not getattr(args, "no_ignore", False)
            result = get_code_structure(
                args.path,
                language=lang,
                max_results=args.max,
                respect_ignore=respect_ignore,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "search":
            _validate_path(args.path, must_exist=True, must_be_dir=True)
            ext = set(args.ext) if args.ext else None
            result = api_search(
                args.pattern,
                args.path,
                extensions=ext,
                context_lines=args.context,
                max_results=args.max,
                max_files=args.max_files,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "extract":
            _validate_path(args.file, must_exist=True, must_be_file=True)
            result = extract_file(args.file)

            # Apply filters if specified
            filter_class = getattr(args, "filter_class", None)
            filter_function = getattr(args, "filter_function", None)
            filter_method = getattr(args, "filter_method", None)

            if filter_class or filter_function or filter_method:
                # Filter classes
                if filter_class:
                    result["classes"] = [
                        c
                        for c in result.get("classes", [])
                        if c.get("name") == filter_class
                    ]
                elif filter_method:
                    # Parse Class.method syntax
                    parts = filter_method.split(".", 1)
                    if len(parts) == 2:
                        class_name, method_name = parts
                        filtered_classes = []
                        for c in result.get("classes", []):
                            if c.get("name") == class_name:
                                # Filter to only the requested method
                                c_copy = dict(c)
                                c_copy["methods"] = [
                                    m
                                    for m in c.get("methods", [])
                                    if m.get("name") == method_name
                                ]
                                filtered_classes.append(c_copy)
                        result["classes"] = filtered_classes
                else:
                    # No class filter, clear classes
                    result["classes"] = []

                # Filter functions
                if filter_function:
                    result["functions"] = [
                        f
                        for f in result.get("functions", [])
                        if f.get("name") == filter_function
                    ]
                elif not filter_method:
                    # No function filter (and not method filter), clear functions if class filter active
                    if filter_class:
                        result["functions"] = []

            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "context":
            _validate_path(args.project, must_exist=True, must_be_dir=True)
            # Auto-detect language if not specified
            lang = args.lang or _detect_project_language(args.project)
            ctx = get_relevant_context(
                args.project, args.entry, depth=args.depth, language=lang
            )
            # Output LLM-ready string directly
            print(ctx.to_llm_string())

        elif args.command == "cfg":
            if not Path(args.file).exists():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            lang = args.lang or detect_language_with_default(args.file)
            result = get_cfg_context(args.file, args.function, language=lang)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "dfg":
            if not Path(args.file).exists():
                print(f"Error: File not found: {args.file}", file=sys.stderr)
                sys.exit(1)
            lang = args.lang or detect_language_with_default(args.file)
            result = get_dfg_context(args.file, args.function, language=lang)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "slice":
            _validate_path(args.file, must_exist=True, must_be_file=True)
            lang = args.lang or detect_language_with_default(args.file)
            lines = get_slice(
                args.file,
                args.function,
                args.line,
                direction=args.direction,
                variable=args.var,
                language=lang,
            )
            result = {"lines": sorted(lines), "count": len(lines)}
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "calls":
            _validate_path(args.path, must_exist=True, must_be_dir=True)
            # Find project root for consistent cache handling
            # This allows `tldr calls src/` to work the same as `tldr calls .`
            project_root = _find_project_root(args.path)
            target_path = Path(args.path).resolve()

            # Auto-detect language from project root
            lang = args.lang or _detect_project_language(str(project_root))

            # Use project root for cache lookup and graph building
            graph = _get_or_build_graph(str(project_root), lang, build_project_call_graph)

            # Filter edges if a subdirectory was specified
            edges = graph.edges
            if target_path != project_root:
                # Get relative path prefix for filtering (use as_posix for cross-platform)
                try:
                    rel_prefix = target_path.relative_to(project_root).as_posix()
                    # Filter to edges where from_file or to_file is in the subdirectory
                    edges = [
                        e for e in edges
                        if e[0].startswith(rel_prefix) or e[2].startswith(rel_prefix)
                    ]
                except ValueError:
                    # target_path is not under project_root - use all edges
                    pass

            result = {
                "edges": [
                    {
                        "from_file": e[0],
                        "from_func": e[1],
                        "to_file": e[2],
                        "to_func": e[3],
                    }
                    for e in edges
                ],
                "count": len(edges),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "impact":
            _validate_path(args.path, must_exist=True, must_be_dir=True)
            # Auto-detect language if not specified
            lang = args.lang or _detect_project_language(args.path)
            result = analyze_impact(
                args.path,
                args.func,
                max_depth=args.depth,
                target_file=args.file,
                language=lang,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "dead":
            _validate_path(args.path, must_exist=True, must_be_dir=True)
            # Auto-detect language if not specified
            lang = args.lang or _detect_project_language(args.path)
            result = analyze_dead_code(
                args.path,
                entry_points=args.entry if args.entry else None,
                language=lang,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "arch":
            _validate_path(args.path, must_exist=True, must_be_dir=True)
            # Auto-detect language if not specified
            lang = args.lang or _detect_project_language(args.path)
            result = analyze_architecture(args.path, language=lang)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "imports":
            file_path = _validate_path(args.file, must_exist=True, must_be_file=True).resolve()
            lang = args.lang or detect_language_with_default(args.file)
            result = get_imports(str(file_path), language=lang)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "importers":
            # Find all files that import the given module
            from .daemon.cached_queries import module_matches

            project = _validate_path(args.path, must_exist=True, must_be_dir=True).resolve()

            # Auto-detect language if not specified
            lang = args.lang or _detect_project_language(args.path)

            # Scan all source files and check their imports
            respect_ignore = not getattr(args, "no_ignore", False)
            files = scan_project_files(
                str(project), language=lang, respect_ignore=respect_ignore
            )
            importers = []
            for file_path in files:
                try:
                    imports = get_imports(file_path, language=lang)
                    for imp in imports:
                        mod = imp.get("module", "")
                        names = imp.get("names", [])
                        # Check using normalized matching (supports both aliases and file paths)
                        if module_matches(args.module, mod, names):
                            importers.append(
                                {
                                    "file": str(Path(file_path).relative_to(project)),
                                    "import": imp,
                                }
                            )
                except Exception:
                    # Skip files that can't be parsed
                    pass

            print(json.dumps({"module": args.module, "importers": importers}, indent=2, ensure_ascii=False))

        elif args.command == "change-impact":
            from .change_impact import analyze_change_impact

            # Resolve project path
            project_path = Path(args.project).resolve()
            if not project_path.exists():
                print(f"Error: Project path not found: {args.project}", file=sys.stderr)
                sys.exit(1)

            # Auto-detect language if not specified
            lang = args.lang or _detect_project_language(str(project_path))

            result = analyze_change_impact(
                project_path=str(project_path),
                files=args.files if args.files else None,
                use_session=args.session,
                use_git=args.git,
                git_base=args.git_base,
                language=lang,
                max_depth=args.depth,
            )

            if args.run and result.get("test_command"):
                # Actually run the tests (test_command is a list to avoid shell injection)
                import shlex
                import subprocess as sp

                cmd = result["test_command"]
                print(f"Running: {shlex.join(cmd)}", file=sys.stderr)
                sp.run(cmd)  # No shell=True - safe from injection
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "diagnostics":
            from .diagnostics import (
                get_diagnostics,
                get_project_diagnostics,
                format_diagnostics_for_llm,
            )

            target = Path(args.target).resolve()
            if not target.exists():
                print(f"Error: Target not found: {args.target}", file=sys.stderr)
                sys.exit(1)

            if args.project or target.is_dir():
                result = get_project_diagnostics(
                    str(target),
                    language=args.lang or "python",
                    include_lint=not args.no_lint,
                )
            else:
                result = get_diagnostics(
                    str(target),
                    language=args.lang,
                    include_lint=not args.no_lint,
                )

            if args.format == "text":
                print(format_diagnostics_for_llm(result))
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.command == "warm":
            import subprocess
            import time

            project_path = Path(args.path).resolve()

            # Validate path exists
            if not project_path.exists():
                print(f"Error: Path not found: {args.path}", file=sys.stderr)
                sys.exit(1)

            if args.background:
                # Spawn background process (cross-platform)
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tldr.cli",
                        "warm",
                        str(project_path),
                        "--lang",
                        args.lang,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    **_get_subprocess_detach_kwargs(),
                )
                print(f"Background indexing spawned for {project_path}")
            else:
                # Build call graph
                from .cross_file_calls import scan_project, ProjectCallGraph

                respect_ignore = not getattr(args, 'no_ignore', False)

                # Determine languages to process
                if args.lang == "all":
                    try:
                        from .semantic import _detect_project_languages
                        target_languages = _detect_project_languages(project_path, respect_ignore=respect_ignore)
                        print(f"Detected languages: {', '.join(target_languages)}")
                    except ImportError:
                        # Fallback if semantic module issue
                        target_languages = ["python", "typescript", "javascript", "go", "rust"]
                else:
                    target_languages = [args.lang]

                all_files = set()
                combined_edges = []
                processed_languages = []

                for target_lang in target_languages:
                    try:
                        # Scan files
                        files = scan_project(project_path, language=target_lang, respect_ignore=respect_ignore)
                        all_files.update(files)

                        # Build graph
                        graph = build_project_call_graph(project_path, language=target_lang)
                        combined_edges.extend([
                            {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                            for e in graph.edges
                        ])
                        print(f"Processed {target_lang}: {len(files)} files, {len(graph.edges)} edges")
                        processed_languages.append(target_lang)
                    except ValueError as e:
                        # Expected for unsupported languages
                        print(f"Warning: {target_lang}: {e}", file=sys.stderr)
                    except Exception as e:
                        # Unexpected error - show traceback if debug enabled
                        print(f"Warning: Failed to process {target_lang}: {e}", file=sys.stderr)
                        if os.environ.get("TLDR_DEBUG"):
                            import traceback
                            traceback.print_exc()

                # Create cache directory
                cache_dir = project_path / ".tldr" / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)

                # Save cache file
                cache_file = cache_dir / "call_graph.json"
                # Deduplicate edges
                unique_edges = list({(e["from_file"], e["from_func"], e["to_file"], e["to_func"]): e for e in combined_edges}.values())
                
                cache_data = {
                    "edges": unique_edges,
                    "languages": processed_languages if processed_languages else target_languages,
                    "timestamp": time.time(),
                }
                cache_file.write_text(json.dumps(cache_data, indent=2))

                # Print stats
                print(f"Total: Indexed {len(all_files)} files, found {len(unique_edges)} edges")

        elif args.command == "semantic":
            from .ml_engine import build_index, search, SUPPORTED_MODELS, DEFAULT_MODEL

            if args.action == "index":
                project_path = Path(args.path).resolve()
                if not project_path.exists():
                    print(f"Error: Path not found: {args.path}", file=sys.stderr)
                    sys.exit(1)
                if not project_path.is_dir():
                    print(f"Error: Not a directory: {args.path}", file=sys.stderr)
                    sys.exit(1)

                respect_ignore = not getattr(args, "no_ignore", False)
                # Resolve model name: use provided or default
                model_name = args.model if args.model else DEFAULT_MODEL
                count = build_index(
                    args.path,
                    lang=args.lang,
                    model_name=model_name,
                    respect_ignore=respect_ignore,
                    dimension=args.dimension,
                    backend=args.backend,
                )
                print(f"Indexed {count} code units")

            elif args.action == "search":
                # Invalidate cache if force-reload requested
                if getattr(args, "force_reload", False):
                    from .ml_engine import get_index_manager
                    im = get_index_manager()
                    im.invalidate(str(Path(args.path).resolve()))

                # ml_engine.search uses model from index metadata
                results = search(
                    args.path,
                    args.query,
                    k=args.k,
                    task=getattr(args, "task", "code_search"),
                    expand_graph=args.expand,
                    backend=args.backend,
                )
                print(json.dumps(results, indent=2, ensure_ascii=False))

            elif args.action == "warmup":
                from .ml_engine import get_model_manager, DEFAULT_MODEL

                mm = get_model_manager()
                model = args.model or DEFAULT_MODEL
                mm.load(model)
                result = mm.warmup()
                print(json.dumps(result, indent=2, ensure_ascii=False))

            elif args.action == "unload":
                from .ml_engine import get_model_manager

                mm = get_model_manager()
                mm.unload()
                print("Model unloaded successfully")

            elif args.action == "cache":
                from .ml_engine import get_index_manager

                im = get_index_manager()

                if args.cache_action == "clear":
                    im.clear()
                    print("Index cache cleared")
                elif args.cache_action == "stats":
                    print(json.dumps(im.stats(), indent=2, ensure_ascii=False))
                elif args.cache_action == "invalidate":
                    project_path = Path(args.path).resolve()
                    im.invalidate(str(project_path))
                    print(f"Index invalidated for: {project_path}")

            elif args.action == "device":
                from .ml_engine import detect_device, check_tei_available

                dev = detect_device()
                # Get free memory dynamically for CUDA devices
                free_memory = 0
                if dev.name == "cuda":
                    import torch
                    torch.cuda.synchronize()
                    free_memory = torch.cuda.mem_get_info()[0]
                info = {
                    "device": dev.name,
                    "device_count": dev.device_count,
                    "total_memory_gb": round(dev.total_memory / 1e9, 2),
                    "free_memory_gb": round(free_memory / 1e9, 2),
                    "supports_bf16": dev.supports_bf16,
                    "tei_available": check_tei_available(),
                }
                print(json.dumps(info, indent=2, ensure_ascii=False))

            elif args.action == "memory":
                from .ml_engine import get_model_manager

                mm = get_model_manager()
                stats = mm.memory_stats()
                print(json.dumps(stats, indent=2, ensure_ascii=False))

        elif args.command == "doctor":
            import shutil
            import subprocess

            # Tool definitions: language -> (type_checker, linter, install_commands)
            TOOL_INFO = {
                "python": {
                    "type_checker": (
                        "pyright",
                        "pip install pyright  OR  npm install -g pyright",
                    ),
                    "linter": ("ruff", "pip install ruff"),
                },
                "typescript": {
                    "type_checker": ("tsc", "npm install -g typescript"),
                    "linter": None,
                },
                "javascript": {
                    "type_checker": None,
                    "linter": ("eslint", "npm install -g eslint"),
                },
                "go": {
                    "type_checker": ("go", "https://go.dev/dl/"),
                    "linter": (
                        "golangci-lint",
                        "brew install golangci-lint  OR  go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest",
                    ),
                },
                "rust": {
                    "type_checker": ("cargo", "https://rustup.rs/"),
                    "linter": ("cargo-clippy", "rustup component add clippy"),
                },
                "java": {
                    "type_checker": ("javac", "Install JDK: https://adoptium.net/"),
                    "linter": (
                        "checkstyle",
                        "brew install checkstyle  OR  download from checkstyle.org",
                    ),
                },
                "c": {
                    "type_checker": (
                        "gcc",
                        "xcode-select --install  OR  apt install gcc",
                    ),
                    "linter": (
                        "cppcheck",
                        "brew install cppcheck  OR  apt install cppcheck",
                    ),
                },
                "cpp": {
                    "type_checker": (
                        "g++",
                        "xcode-select --install  OR  apt install g++",
                    ),
                    "linter": (
                        "cppcheck",
                        "brew install cppcheck  OR  apt install cppcheck",
                    ),
                },
                "ruby": {
                    "type_checker": None,
                    "linter": ("rubocop", "gem install rubocop"),
                },
                "php": {
                    "type_checker": None,
                    "linter": ("phpstan", "composer global require phpstan/phpstan"),
                },
                "kotlin": {
                    "type_checker": (
                        "kotlinc",
                        "brew install kotlin  OR  sdk install kotlin",
                    ),
                    "linter": ("ktlint", "brew install ktlint"),
                },
                "swift": {
                    "type_checker": ("swiftc", "xcode-select --install"),
                    "linter": ("swiftlint", "brew install swiftlint"),
                },
                "csharp": {
                    "type_checker": ("dotnet", "https://dotnet.microsoft.com/download"),
                    "linter": None,
                },
                "scala": {
                    "type_checker": (
                        "scalac",
                        "brew install scala  OR  sdk install scala",
                    ),
                    "linter": None,
                },
                "elixir": {
                    "type_checker": (
                        "elixir",
                        "brew install elixir  OR  asdf install elixir",
                    ),
                    "linter": ("mix", "Included with Elixir"),
                },
                "lua": {
                    "type_checker": None,
                    "linter": ("luacheck", "luarocks install luacheck"),
                },
            }

            # Install commands for --install flag
            INSTALL_COMMANDS = {
                "python": ["pip", "install", "pyright", "ruff"],
                "go": [
                    "go",
                    "install",
                    "github.com/golangci/golangci-lint/cmd/golangci-lint@latest",
                ],
                "rust": ["rustup", "component", "add", "clippy"],
                "ruby": ["gem", "install", "rubocop"],
                "kotlin": ["brew", "install", "kotlin", "ktlint"],
                "swift": ["brew", "install", "swiftlint"],
                "lua": ["luarocks", "install", "luacheck"],
            }

            if args.install:
                lang = args.install.lower()
                if lang not in INSTALL_COMMANDS:
                    print(
                        f"Error: No auto-install available for '{lang}'",
                        file=sys.stderr,
                    )
                    print(
                        f"Available: {', '.join(sorted(INSTALL_COMMANDS.keys()))}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                # Check which tools are already installed
                tools_info = TOOL_INFO.get(lang, {})
                tool_names = []
                if tools_info.get("type_checker"):
                    tool_names.append(tools_info["type_checker"][0])
                if tools_info.get("linter"):
                    tool_names.append(tools_info["linter"][0])

                missing_tools = []
                for tool in tool_names:
                    if shutil.which(tool):
                        print(f"  Already installed: {tool}")
                    else:
                        missing_tools.append(tool)

                if not missing_tools:
                    print(f"All tools for {lang} are already installed")
                else:
                    print(f"Missing tools: {', '.join(missing_tools)}")
                    cmd = INSTALL_COMMANDS[lang]
                    print(f"Installing tools for {lang}: {' '.join(cmd)}")
                    try:
                        subprocess.run(cmd, check=True)
                        print(f"Installed {lang} tools")
                    except subprocess.CalledProcessError as e:
                        print(f"Install failed: {e}", file=sys.stderr)
                        sys.exit(1)
                    except FileNotFoundError:
                        print(f"Command not found: {cmd[0]}", file=sys.stderr)
                        sys.exit(1)
            else:
                # Check all tools
                results = {}
                for lang, tools in TOOL_INFO.items():
                    lang_result = {"type_checker": None, "linter": None}

                    if tools["type_checker"]:
                        tool_name, install_cmd = tools["type_checker"]
                        path = shutil.which(tool_name)
                        lang_result["type_checker"] = {
                            "name": tool_name,
                            "installed": path is not None,
                            "path": path,
                            "install": install_cmd if not path else None,
                        }

                    if tools["linter"]:
                        tool_name, install_cmd = tools["linter"]
                        path = shutil.which(tool_name)
                        lang_result["linter"] = {
                            "name": tool_name,
                            "installed": path is not None,
                            "path": path,
                            "install": install_cmd if not path else None,
                        }

                    results[lang] = lang_result

                if args.json:
                    print(json.dumps(results, indent=2, ensure_ascii=False))
                else:
                    print("TLDR Diagnostics Check")
                    print("=" * 50)
                    print()

                    missing_count = 0
                    for lang, checks in sorted(results.items()):
                        lines = []

                        tc = checks["type_checker"]
                        if tc:
                            if tc["installed"]:
                                lines.append(f"  ✓ {tc['name']} - {tc['path']}")
                            else:
                                lines.append(f"  ✗ {tc['name']} - not found")
                                lines.append(f"    → {tc['install']}")
                                missing_count += 1

                        linter = checks["linter"]
                        if linter:
                            if linter["installed"]:
                                lines.append(f"  ✓ {linter['name']} - {linter['path']}")
                            else:
                                lines.append(f"  ✗ {linter['name']} - not found")
                                lines.append(f"    → {linter['install']}")
                                missing_count += 1

                        if lines:
                            print(f"{lang.capitalize()}:")
                            for line in lines:
                                print(line)
                            print()

                    if missing_count > 0:
                        print(
                            f"Missing {missing_count} tool(s). Run: tldr doctor --install <lang>"
                        )
                    else:
                        print("All diagnostic tools installed!")

        elif args.command == "daemon":
            from .daemon import start_daemon, stop_daemon, query_daemon

            project_path = Path(args.project).resolve()

            if args.action == "start":
                # Ensure .tldr directory exists
                tldr_dir = project_path / ".tldr"
                tldr_dir.mkdir(parents=True, exist_ok=True)
                # Start daemon (will fork to background on Unix)
                start_daemon(project_path, foreground=False)

            elif args.action == "stop":
                if stop_daemon(project_path):
                    print("Daemon stopped")
                else:
                    print("Daemon not running")

            elif args.action == "status":
                try:
                    result = query_daemon(project_path, {"cmd": "status"})
                    print(f"Status: {result.get('status', 'unknown')}")
                    if "uptime" in result:
                        uptime = int(result["uptime"])
                        mins, secs = divmod(uptime, 60)
                        hours, mins = divmod(mins, 60)
                        print(f"Uptime: {hours}h {mins}m {secs}s")
                except (ConnectionRefusedError, FileNotFoundError):
                    print("Daemon not running")

            elif args.action == "query":
                try:
                    result = query_daemon(project_path, {"cmd": args.cmd})
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                except (ConnectionRefusedError, FileNotFoundError):
                    print("Error: Daemon not running", file=sys.stderr)
                    sys.exit(1)

            elif args.action == "notify":
                try:
                    file_path = Path(args.file).resolve()
                    result = query_daemon(
                        project_path, {"cmd": "notify", "file": str(file_path)}
                    )
                    if result.get("status") == "ok":
                        dirty = result.get("dirty_count", 0)
                        threshold = result.get("threshold", 20)
                        if result.get("reindex_triggered"):
                            print(f"Reindex triggered ({dirty}/{threshold} files)")
                        else:
                            print(f"Tracked: {dirty}/{threshold} files")
                    else:
                        print(
                            f"Error: {result.get('message', 'Unknown error')}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                except (ConnectionRefusedError, FileNotFoundError):
                    # Daemon not running - silently ignore, file edits shouldn't fail
                    pass

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
