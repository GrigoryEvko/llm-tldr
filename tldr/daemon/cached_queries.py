"""
Salsa-memoized query functions for the TLDR daemon.

These functions wrap the TLDR API with automatic caching via SalsaDB.
Results are memoized and automatically invalidated when source files change.
"""

import re
from pathlib import Path

from tldr.salsa import SalsaDB, salsa_query


# File extensions to strip when normalizing module paths
_SOURCE_EXTENSIONS = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs",
    ".java", ".kt", ".kts", ".scala", ".rb", ".php",
    ".swift", ".cs", ".lua", ".ex", ".exs", ".c", ".h",
    ".cpp", ".cc", ".cxx", ".hpp",
})

# Common path prefixes that represent aliases or source directories
_PATH_PREFIXES = re.compile(r"^(@/|~/|\./|\.{2,}/|src/|lib/|app/|packages/)")


def normalize_module_name(module: str) -> str:
    """Normalize a module name or file path for comparison.

    Strips common prefixes (@/, ./, src/, etc.) and file extensions,
    converting to a canonical form for matching.

    Args:
        module: Module name or file path (e.g., "@/utils/async", "src/utils/async.ts")

    Returns:
        Normalized module name (e.g., "utils/async")

    Examples:
        >>> normalize_module_name("@/utils/async")
        'utils/async'
        >>> normalize_module_name("src/utils/async.ts")
        'utils/async'
        >>> normalize_module_name("./components/Button")
        'components/Button'
    """
    if not module:
        return ""

    result = module

    # Strip common path prefixes (may need multiple passes for nested prefixes)
    prev = None
    while prev != result:
        prev = result
        result = _PATH_PREFIXES.sub("", result)

    # Strip file extension if present
    path = Path(result)
    if path.suffix.lower() in _SOURCE_EXTENSIONS:
        result = str(path.with_suffix(""))

    # Handle index files (utils/index -> utils)
    if result.endswith("/index"):
        result = result[:-6]

    return result


def module_matches(search_pattern: str, import_module: str, imported_names: list[str] | None = None) -> bool:
    """Check if a search pattern matches an import module or imported names.

    Supports both exact matches and normalized path matching, allowing
    file paths like "src/utils/async" to match imports like "@/utils/async".

    Args:
        search_pattern: Module name or file path to search for
        import_module: The module string from an import statement
        imported_names: Optional list of imported names (from X import a, b)

    Returns:
        True if the search pattern matches the import

    Examples:
        >>> module_matches("@/utils/async", "@/utils/async", [])
        True
        >>> module_matches("src/utils/async", "@/utils/async", [])
        True
        >>> module_matches("utils/async", "@/utils/async", [])
        True
        >>> module_matches("async", "@/utils/async", ["async"])
        True
    """
    if not search_pattern:
        return False

    # Quick substring check (original behavior for exact matches)
    if search_pattern in import_module:
        return True

    if imported_names and search_pattern in imported_names:
        return True

    # Normalize both for path-agnostic matching
    normalized_pattern = normalize_module_name(search_pattern)
    normalized_import = normalize_module_name(import_module)

    if not normalized_pattern:
        return False

    # Check if normalized forms match (exact or suffix)
    if normalized_pattern == normalized_import:
        return True

    # Allow suffix matching: "utils/async" matches "some/path/utils/async"
    if normalized_import.endswith("/" + normalized_pattern):
        return True

    # Allow prefix matching for submodule queries
    if normalized_import.startswith(normalized_pattern + "/"):
        return True

    # Check if the pattern matches the final segment (basename match)
    # e.g., "async" matches "utils/async"
    if "/" not in normalized_pattern and normalized_import.endswith("/" + normalized_pattern):
        return True

    # Handle bare name matching the final segment
    if "/" not in normalized_pattern and normalized_import == normalized_pattern:
        return True

    # For simple names without slashes, check if it matches the last component
    if "/" not in normalized_pattern:
        import_parts = normalized_import.split("/")
        if import_parts and import_parts[-1] == normalized_pattern:
            return True

    return False


@salsa_query
def cached_search(db: SalsaDB, project: str, pattern: str, max_results: int) -> dict:
    """Cached search query - memoized by SalsaDB."""
    from tldr import api
    results = api.search(pattern=pattern, root=Path(project), max_results=max_results)
    return {"status": "ok", "results": results}


@salsa_query
def cached_extract(db: SalsaDB, file_path: str) -> dict:
    """Cached file extraction - memoized by SalsaDB."""
    from tldr import api
    result = api.extract_file(file_path)
    return {"status": "ok", "result": result}


@salsa_query
def cached_dead_code(db: SalsaDB, project: str, entry_points: tuple, language: str) -> dict:
    """Cached dead code analysis - memoized by SalsaDB."""
    from tldr.analysis import analyze_dead_code
    # Convert tuple back to list for the API
    entry_list = list(entry_points) if entry_points else None
    result = analyze_dead_code(project, entry_points=entry_list, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_architecture(db: SalsaDB, project: str, language: str) -> dict:
    """Cached architecture analysis - memoized by SalsaDB."""
    from tldr.analysis import analyze_architecture
    result = analyze_architecture(project, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_cfg(db: SalsaDB, file_path: str, function: str, language: str) -> dict:
    """Cached CFG extraction - memoized by SalsaDB."""
    from tldr.api import get_cfg_context
    result = get_cfg_context(file_path, function, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_dfg(db: SalsaDB, file_path: str, function: str, language: str) -> dict:
    """Cached DFG extraction - memoized by SalsaDB."""
    from tldr.api import get_dfg_context
    result = get_dfg_context(file_path, function, language=language)
    return {"status": "ok", "result": result}


@salsa_query
def cached_slice(db: SalsaDB, file_path: str, function: str, line: int, direction: str, variable: str) -> dict:
    """Cached program slice - memoized by SalsaDB."""
    from tldr.api import get_slice
    var = variable if variable else None
    lines = get_slice(file_path, function, line, direction=direction, variable=var)
    return {"status": "ok", "lines": sorted(lines), "count": len(lines)}


@salsa_query
def cached_tree(db: SalsaDB, project: str, extensions: tuple, exclude_hidden: bool) -> dict:
    """Cached file tree - memoized by SalsaDB."""
    from tldr.api import get_file_tree
    ext_set = set(extensions) if extensions else None
    result = get_file_tree(project, extensions=ext_set, exclude_hidden=exclude_hidden)
    return {"status": "ok", "result": result}


@salsa_query
def cached_structure(db: SalsaDB, project: str, language: str, max_results: int) -> dict:
    """Cached code structure - memoized by SalsaDB."""
    from tldr.api import get_code_structure
    result = get_code_structure(project, language=language, max_results=max_results)
    return {"status": "ok", "result": result}


@salsa_query
def cached_context(db: SalsaDB, project: str, entry: str, language: str, depth: int) -> dict:
    """Cached relevant context - memoized by SalsaDB."""
    from tldr.api import get_relevant_context
    result = get_relevant_context(project, entry, language=language, depth=depth)
    return {"status": "ok", "result": result}


@salsa_query
def cached_imports(db: SalsaDB, file_path: str, language: str) -> dict:
    """Cached imports extraction - memoized by SalsaDB."""
    from tldr.api import get_imports
    result = get_imports(file_path, language=language)
    return {"status": "ok", "imports": result}


@salsa_query
def cached_importers(db: SalsaDB, project: str, module: str, language: str) -> dict:
    """Cached reverse import lookup - memoized by SalsaDB.

    Supports both module aliases (e.g., "@/utils/async") and file paths
    (e.g., "src/utils/async") through normalized matching.
    """
    from tldr.api import get_imports, scan_project_files

    files = scan_project_files(project, language=language)
    importers = []
    project_path = Path(project)

    for file_path in files:
        try:
            imports = get_imports(file_path, language=language)
            for imp in imports:
                mod = imp.get("module", "")
                names = imp.get("names", [])
                if module_matches(module, mod, names):
                    importers.append({
                        "file": str(Path(file_path).relative_to(project_path)),
                        "import": imp,
                    })
        except Exception:
            pass

    return {"status": "ok", "module": module, "importers": importers}
