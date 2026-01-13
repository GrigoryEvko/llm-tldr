"""
Cross-file call graph resolution.

Builds a project-wide call graph that resolves function calls across files
by analyzing import statements and matching call sites to definitions.

Supported languages for call graph analysis (SUPPORTED_CALL_GRAPH_LANGUAGES):
- Python (.py)
- TypeScript (.ts, .tsx)
- Go (.go)
- Rust (.rs)
- Java (.java)
- C (.c, .h)

File discovery (scan_project) supports all languages in SUPPORTED_LANGUAGES:
python, typescript, javascript, go, rust, java, c, cpp, kotlin, scala,
csharp, php, ruby, swift, lua, elixir.

Note: Languages not in SUPPORTED_CALL_GRAPH_LANGUAGES can be scanned for
file discovery, but call graph analysis will raise ValueError.

Key functions:
- scan_project(root, language) - find all source files in a project
- parse_imports(file) - extract import statements from a file
- build_function_index(root, language) - map {module.func: file_path} for all functions
- resolve_calls(file, index) - match call sites to definitions
- build_project_call_graph(root, language) - orchestrate all to build complete graph
"""

import ast
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from tldr.workspace import WorkspaceConfig, load_workspace_config, filter_paths

# Tree-sitter base library check
try:
    import tree_sitter

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

import importlib

# Module-level parser cache to avoid expensive parser recreation per file.
# Creating a tree-sitter parser takes ~10-50ms. For a 100-file project,
# caching saves 1-5 seconds of parser creation time.
_PARSER_CACHE: dict[str, "tree_sitter.Parser"] = {}

# Language module and function mapping for tree-sitter grammars.
# Maps language name -> (module_name, language_function_name)
# Some languages have non-standard function names (e.g., language_typescript)
LANGUAGE_CONFIG: dict[str, tuple[str, str]] = {
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "lua": ("tree_sitter_lua", "language"),
    "elixir": ("tree_sitter_elixir", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "swift": ("tree_sitter_swift", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "scala": ("tree_sitter_scala", "language"),
}

# Supported languages for file discovery (used by scan_project).
# Single source of truth for language -> file extensions mapping.
# Must be kept in sync with LANGUAGE_CONFIG above.
SUPPORTED_LANGUAGES: dict[str, set[str]] = {
    "python": {".py"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx"},
    "go": {".go"},
    "rust": {".rs"},
    "java": {".java"},
    "c": {".c", ".h"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp"},
    "kotlin": {".kt", ".kts"},
    "scala": {".scala", ".sc"},
    "csharp": {".cs"},
    "php": {".php"},
    "ruby": {".rb"},
    "swift": {".swift"},
    "lua": {".lua"},
    "elixir": {".ex", ".exs"},
}

# Languages with full call graph support (have indexing + call graph building).
# scan_project() supports more languages for file discovery, but call graph
# analysis requires language-specific AST traversal that's only implemented
# for these languages. Others will fail fast with a clear error.
SUPPORTED_CALL_GRAPH_LANGUAGES: frozenset[str] = frozenset({
    "python", "typescript", "go", "rust", "java", "c"
})


def get_parser(language: str) -> Optional["tree_sitter.Parser"]:
    """Get or create a cached tree-sitter parser for the given language.

    Uses lazy loading via importlib to avoid importing unused language grammars.
    Returns None if tree-sitter base library or language-specific grammar is unavailable.

    Args:
        language: Language identifier (e.g., "python", "typescript", "go")

    Returns:
        Cached Parser instance, or None if unavailable
    """
    if not TREE_SITTER_AVAILABLE:
        return None

    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    if language not in LANGUAGE_CONFIG:
        return None

    module_name, func_name = LANGUAGE_CONFIG[language]
    try:
        mod = importlib.import_module(module_name)
        lang_func = getattr(mod, func_name)
        lang = tree_sitter.Language(lang_func())
        parser = tree_sitter.Parser(lang)
        _PARSER_CACHE[language] = parser
        return parser
    except (ImportError, AttributeError):
        return None

# Threshold for parallel processing. Below this, sequential is faster due to
# process spawn overhead (~50-100ms per worker).
MIN_FILES_FOR_PARALLEL = 15


@dataclass
class ProjectCallGraph:
    """Cross-file call graph with edges as (src_file, src_func, dst_file, dst_func).

    Maintains a secondary index (_edges_by_src_file) for O(1) edge removal by file,
    critical for incremental patching performance.
    """

    _edges: set[tuple[str, str, str, str]] = field(default_factory=set)
    # Secondary index: src_file -> set of edges originating from that file
    # Enables O(1) edge lookup/removal instead of O(E) full scan
    _edges_by_src_file: dict[str, set[tuple[str, str, str, str]]] = field(
        default_factory=dict
    )

    def add_edge(self, src_file: str, src_func: str, dst_file: str, dst_func: str):
        """Add a call edge from src_file:src_func to dst_file:dst_func.

        Maintains both primary edge set and secondary index for O(1) operations.
        """
        edge = (src_file, src_func, dst_file, dst_func)
        self._edges.add(edge)
        # Maintain secondary index
        if src_file not in self._edges_by_src_file:
            self._edges_by_src_file[src_file] = set()
        self._edges_by_src_file[src_file].add(edge)

    def remove_edges_for_file(self, src_file: str) -> int:
        """Remove all edges originating from a source file in O(1).

        Args:
            src_file: Relative path to the source file

        Returns:
            Number of edges removed
        """
        if src_file not in self._edges_by_src_file:
            return 0

        edges_to_remove = self._edges_by_src_file.pop(src_file)
        self._edges -= edges_to_remove
        return len(edges_to_remove)

    def get_edges_for_file(self, src_file: str) -> set[tuple[str, str, str, str]]:
        """Get all edges originating from a source file in O(1).

        Args:
            src_file: Relative path to the source file

        Returns:
            Set of edges (possibly empty) from that file
        """
        return self._edges_by_src_file.get(src_file, set()).copy()

    @property
    def edges(self) -> set[tuple[str, str, str, str]]:
        """Return all edges as a set of tuples."""
        return self._edges

    def __contains__(self, edge: tuple[str, str, str, str]) -> bool:
        """Check if an edge exists in the graph."""
        return edge in self._edges


def scan_project(
    root: str | Path,
    language: str = "python",
    workspace_config: Optional[WorkspaceConfig] = None,
    respect_ignore: bool = True,
) -> list[str]:
    """
    Find all source files in the project for the given language.

    Args:
        root: Project root directory
        language: One of the languages in SUPPORTED_LANGUAGES (python, typescript,
                 javascript, go, rust, java, c, cpp, kotlin, scala, csharp, php,
                 ruby, swift, lua, elixir)
        workspace_config: Optional WorkspaceConfig for monorepo scoping.
                         If provided, filters files by activePackages and excludePatterns.
        respect_ignore: If True, respect .tldrignore patterns (default True)

    Returns:
        List of absolute paths to source files

    Raises:
        ValueError: If language is not in SUPPORTED_LANGUAGES
    """
    from .tldrignore import load_ignore_patterns, should_ignore

    root_str = str(Path(root))  # Convert to str to ensure os.walk/relpath return str
    files: list[str] = []

    # Load ignore patterns if respecting .tldrignore
    ignore_spec = load_ignore_patterns(root_str) if respect_ignore else None

    # Look up extensions from single source of truth
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {sorted(SUPPORTED_LANGUAGES.keys())}"
        )
    extensions = SUPPORTED_LANGUAGES[language]

    for dirpath, dirnames, filenames in os.walk(root_str):
        # Skip ignored directories (modifying dirnames in-place prunes os.walk)
        # Cast to str to satisfy type checker (os.walk/relpath return str for str input)
        dirpath_str = str(dirpath)
        if respect_ignore and ignore_spec:
            rel_dir = str(os.path.relpath(dirpath_str, root_str))
            # Check if current directory should be ignored
            if rel_dir != "." and should_ignore(rel_dir + "/", root_str, ignore_spec):
                dirnames.clear()  # Don't descend into ignored directories
                continue
            # Filter subdirectories
            dirnames[:] = [
                d
                for d in dirnames
                if not should_ignore(os.path.join(rel_dir, str(d)) + "/", root_str, ignore_spec)
            ]

        for filename in filenames:
            if filename.endswith(tuple(extensions)):
                file_path = os.path.join(dirpath_str, str(filename))
                # Check individual file against ignore patterns
                if respect_ignore and ignore_spec:
                    rel_path = str(os.path.relpath(file_path, root_str))
                    if should_ignore(rel_path, root_str, ignore_spec):
                        continue
                files.append(file_path)

    # Apply workspace config filtering if provided
    if workspace_config is not None:
        # Convert absolute paths to relative for filtering, then back to absolute
        rel_files = [os.path.relpath(f, root_str) for f in files]
        filtered_rel = filter_paths(rel_files, workspace_config)
        files = [os.path.join(root_str, f) for f in filtered_rel]

    return files


def parse_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Python file.

    Args:
        file_path: Path to Python file

    Returns:
        List of import info dicts with keys: module, names, is_from, aliases
    """
    file_path = Path(file_path)
    try:
        source = file_path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return []

    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "module": alias.name,
                        "names": [],
                        "is_from": False,
                        "alias": alias.asname,
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            # Handle both 'from .module import x' (module='module', level=1)
            # and 'from . import x' (module=None, level=1)
            if node.module or node.level > 0:
                names = []
                aliases = {}
                for alias in node.names:
                    names.append(alias.name)
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                # Build module name with relative import prefix
                # e.g., level=2, module='utils' -> '..utils'
                # e.g., level=1, module=None -> '.'
                prefix = "." * node.level
                module_name = prefix + (node.module or "")
                imports.append(
                    {
                        "module": module_name,
                        "names": names,
                        "is_from": True,
                        "aliases": aliases,
                        "level": node.level,  # For downstream resolution
                    }
                )

    return imports


def parse_ts_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a TypeScript file.

    Args:
        file_path: Path to TypeScript file

    Returns:
        List of import info dicts with keys: module, names, is_default, aliases
    """
    file_path = Path(file_path)
    parser = get_parser("typescript")
    if parser is None:
        return []

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        if node.type == "import_statement":
            import_info = _parse_ts_import_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)
    return imports


def _parse_ts_import_node(node, source: bytes) -> dict | None:
    """Parse a single TypeScript import statement."""
    module = None
    names = []
    aliases = {}
    default_name = None

    for child in node.children:
        if child.type == "string":
            # Module path - strip quotes
            module = (
                source[child.start_byte : child.end_byte].decode("utf-8").strip("'\"")
            )
        elif child.type == "import_clause":
            for clause_child in child.children:
                if clause_child.type == "identifier":
                    # Default import: import Foo from "module"
                    default_name = source[
                        clause_child.start_byte : clause_child.end_byte
                    ].decode("utf-8")
                elif clause_child.type == "named_imports":
                    # Named imports: import { foo, bar as baz } from "module"
                    for named in clause_child.children:
                        if named.type == "import_specifier":
                            orig_name = None
                            alias = None
                            for spec_child in named.children:
                                if spec_child.type == "identifier":
                                    if orig_name is None:
                                        orig_name = source[
                                            spec_child.start_byte : spec_child.end_byte
                                        ].decode("utf-8")
                                    else:
                                        alias = source[
                                            spec_child.start_byte : spec_child.end_byte
                                        ].decode("utf-8")
                            if orig_name:
                                names.append(orig_name)
                                if alias:
                                    aliases[alias] = orig_name
                elif clause_child.type == "namespace_import":
                    # Namespace import: import * as foo from "module"
                    for ns_child in clause_child.children:
                        if ns_child.type == "identifier":
                            alias = source[
                                ns_child.start_byte : ns_child.end_byte
                            ].decode("utf-8")
                            aliases[alias] = "*"

    if module:
        return {
            "module": module,
            "names": names,
            "default": default_name,
            "aliases": aliases,
        }
    return None


def parse_go_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Go file.

    Args:
        file_path: Path to Go file

    Returns:
        List of import info dicts with keys: module, alias
    """
    file_path = Path(file_path)
    parser = get_parser("go")
    if parser is None:
        return []

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        if node.type == "import_declaration":
            _parse_go_import_node(node, source, imports)
        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)
    return imports


def _parse_go_import_node(node, source: bytes, imports: list):
    """Parse Go import declaration - handles both single and grouped imports."""
    for child in node.children:
        if child.type == "import_spec":
            _parse_go_import_spec(child, source, imports)
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    _parse_go_import_spec(spec, source, imports)


def _parse_go_import_spec(spec_node, source: bytes, imports: list):
    """Parse a single Go import spec (potentially with alias)."""
    alias = None
    module = None

    for child in spec_node.children:
        if child.type == "package_identifier":
            # This is the alias: import alias "path"
            alias = source[child.start_byte : child.end_byte].decode("utf-8")
        elif child.type == "interpreted_string_literal":
            # This is the module path
            module = (
                source[child.start_byte : child.end_byte].decode("utf-8").strip('"')
            )

    if module:
        imports.append(
            {
                "module": module,
                "alias": alias,
            }
        )


def parse_rust_imports(file_path: str | Path) -> list[dict]:
    """
    Extract use statements and mod declarations from a Rust file.

    Args:
        file_path: Path to Rust file

    Returns:
        List of import info dicts with keys: module, names, aliases, is_mod
        - aliases: Dict mapping alias -> original_name for 'as' renames
    """
    file_path = Path(file_path)
    parser = get_parser("rust")
    if parser is None:
        return []

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        # Use declarations: use crate::utils::helper;
        if node.type == "use_declaration":
            import_info = _parse_rust_use_node(node, source)
            if import_info:
                imports.append(import_info)

        # Mod declarations: mod utils;
        elif node.type == "mod_item":
            # Check if it's a mod declaration (not an inline module)
            has_body = False
            name = None
            for child in node.children:
                if child.type == "identifier":
                    name = source[child.start_byte : child.end_byte].decode("utf-8")
                elif child.type == "declaration_list":
                    has_body = True

            if name and not has_body:
                imports.append(
                    {
                        "module": name,
                        "names": [],
                        "is_mod": True,
                    }
                )

        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)
    return imports


def _parse_rust_use_node(node, source: bytes) -> dict | None:
    """Parse a single Rust use statement.

    Returns dict with keys:
        module: The module path (e.g., "crate::utils")
        names: List of imported names (original names, not aliases)
        aliases: Dict mapping alias -> original_name for 'as' renames
        is_mod: Always False for use statements
    """
    # Get the full use path text
    text = source[node.start_byte : node.end_byte].decode("utf-8")

    # Strip "use " prefix and trailing semicolon
    text = text.replace("use ", "").rstrip(";").strip()

    # Handle pub use
    if text.startswith("pub "):
        text = text[4:].strip()

    # Parse the path to extract module and names
    # Examples:
    #   std::io              -> module="std::io", names=[]
    #   crate::utils::helper -> module="crate::utils", names=["helper"]
    #   self::inner::*       -> module="self::inner", names=["*"]
    #   std::collections::{HashMap, HashSet} -> module="std::collections", names=["HashMap", "HashSet"]
    #   crate::foo as bar    -> module="crate", names=["foo"], aliases={"bar": "foo"}

    raw_names = []
    module = text

    # Handle glob imports: use foo::*
    if text.endswith("::*"):
        module = text[:-3]
        raw_names = ["*"]
    # Handle grouped imports: use foo::{bar, baz}
    elif "{" in text:
        brace_start = text.index("{")
        module = text[:brace_start].rstrip("::")
        brace_content = text[brace_start + 1 : text.rindex("}")]
        raw_names = [n.strip() for n in brace_content.split(",")]
    # Handle simple imports: use foo::bar or use foo::bar as baz
    elif "::" in text:
        parts = text.rsplit("::", 1)
        module = parts[0]
        raw_names = [parts[1]]

    # Process names to extract 'as' aliases
    # e.g., "foo as bar" -> name="foo", alias="bar"
    names = []
    aliases = {}
    for raw_name in raw_names:
        if " as " in raw_name:
            orig_name, alias = raw_name.split(" as ", 1)
            orig_name = orig_name.strip()
            alias = alias.strip()
            names.append(orig_name)
            aliases[alias] = orig_name
        else:
            names.append(raw_name.strip())

    return {
        "module": module,
        "names": names,
        "aliases": aliases,
        "is_mod": False,
    }


def parse_java_imports(file_path: str | Path) -> list[dict]:
    """
    Extract import statements from a Java file.

    Args:
        file_path: Path to Java file

    Returns:
        List of import info dicts with keys: module, is_static, is_wildcard
    """
    file_path = Path(file_path)
    parser = get_parser("java")
    if parser is None:
        return []

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        if node.type == "import_declaration":
            import_info = _parse_java_import_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)
    return imports


def _parse_java_import_node(node, source: bytes) -> dict | None:
    """Parse a single Java import statement."""
    # Get the full import text
    text = source[node.start_byte : node.end_byte].decode("utf-8")

    # Check for static import
    is_static = "static " in text

    # Check for wildcard import
    is_wildcard = text.rstrip(";").endswith("*")

    # Extract the module path
    # Examples:
    #   import java.util.List;          -> module="java.util.List"
    #   import java.util.*;             -> module="java.util.*"
    #   import static java.lang.Math.PI; -> module="java.lang.Math.PI", is_static=True

    # Find the scoped_identifier or identifier node for the import path
    module = None
    for child in node.children:
        if child.type == "scoped_identifier":
            module = source[child.start_byte : child.end_byte].decode("utf-8")
            break
        elif child.type == "identifier":
            module = source[child.start_byte : child.end_byte].decode("utf-8")
        elif child.type == "asterisk":
            # Handle wildcard - module should have been set by scoped_identifier
            if module:
                module = module + ".*"
            is_wildcard = True

    if not module:
        return None

    return {
        "module": module,
        "is_static": is_static,
        "is_wildcard": is_wildcard,
    }


def parse_c_imports(file_path: str | Path) -> list[dict]:
    """
    Extract #include statements from a C file.

    Args:
        file_path: Path to C file

    Returns:
        List of import info dicts with keys: module, is_system
    """
    file_path = Path(file_path)
    parser = get_parser("c")
    if parser is None:
        return []

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return []

    imports = []

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        if node.type == "preproc_include":
            import_info = _parse_c_include_node(node, source)
            if import_info:
                imports.append(import_info)
        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)
    return imports


def _parse_c_include_node(node, source: bytes) -> dict | None:
    """Parse a single C #include statement."""
    # Get the full include text
    text = source[node.start_byte : node.end_byte].decode("utf-8")

    # Check for system include <...> vs local include "..."
    is_system = "<" in text

    # Extract the module path
    # Examples:
    #   #include <stdio.h>        -> module="stdio.h", is_system=True
    #   #include "utils.h"        -> module="utils.h", is_system=False
    #   #include <sys/types.h>    -> module="sys/types.h", is_system=True

    # Find the string_literal or system_lib_string node for the include path
    module = None
    for child in node.children:
        if child.type == "string_literal":
            # Local include "file.h"
            module_text = source[child.start_byte : child.end_byte].decode("utf-8")
            # Strip quotes
            module = module_text.strip('"')
            is_system = False
            break
        elif child.type == "system_lib_string":
            # System include <file.h>
            module_text = source[child.start_byte : child.end_byte].decode("utf-8")
            # Strip angle brackets
            module = module_text.strip("<>")
            is_system = True
            break

    if not module:
        return None

    return {
        "module": module,
        "is_system": is_system,
    }


def _index_single_file(args: tuple[str, str, str]) -> dict:
    """
    Worker function for parallel file indexing.

    Must be at module level for ProcessPoolExecutor pickling.
    Wrapped in try/except to prevent one file's failure from crashing the pool.

    Args:
        args: Tuple of (src_file_path, root_path, language)

    Returns:
        Partial index dict for this single file, empty dict on error
    """
    try:
        src_file, root_str, language = args
        root = Path(root_str)
        src_path = Path(src_file)

        try:
            rel_path = src_path.relative_to(root)
        except ValueError:
            return {}

        # Derive module name from file path
        module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
        module_name = (
            "/".join(module_parts)
            if language == "typescript"
            else ".".join(module_parts)
        )
        simple_module = rel_path.stem

        # Create local index for this file
        partial_index: dict = {}

        if language == "python":
            _index_python_file(
                src_path, rel_path, module_name, simple_module, partial_index
            )
        elif language == "typescript":
            _index_typescript_file(
                src_path, rel_path, module_name, simple_module, partial_index
            )
        elif language == "go":
            _index_go_file(src_path, rel_path, module_name, simple_module, partial_index)
        elif language == "rust":
            _index_rust_file(
                src_path, rel_path, module_name, simple_module, partial_index
            )
        elif language == "java":
            _index_java_file(
                src_path, rel_path, module_name, simple_module, partial_index
            )
        elif language == "c":
            _index_c_file(src_path, rel_path, module_name, simple_module, partial_index)

        return partial_index
    except Exception:
        # Don't let one file's failure crash the entire pool
        return {}


def build_function_index(
    root: str | Path,
    language: str = "python",
    workspace_config: Optional[WorkspaceConfig] = None,
    file_list: Optional[list[str]] = None,
) -> dict[tuple[str, str], str]:
    """
    Build an index mapping (module_name, function_name) to file paths.

    Uses parallel processing via ProcessPoolExecutor when file count exceeds
    MIN_FILES_FOR_PARALLEL threshold to improve performance on large codebases.

    Args:
        root: Project root directory
        language: "python" or "typescript"
        workspace_config: Optional WorkspaceConfig for monorepo scoping
        file_list: Optional pre-scanned list of source files (avoids duplicate scan)

    Returns:
        Dict mapping (module, func_name) tuples to relative file paths

    Raises:
        ValueError: If language is not supported for call graph analysis.
    """
    if language not in SUPPORTED_CALL_GRAPH_LANGUAGES:
        raise ValueError(
            f"Call graph analysis not supported for '{language}'. "
            f"Supported languages: {sorted(SUPPORTED_CALL_GRAPH_LANGUAGES)}"
        )

    root = Path(root)
    index: dict = {}

    # Use provided file_list or scan project
    source_files = (
        file_list
        if file_list is not None
        else scan_project(root, language, workspace_config)
    )

    # Use parallel processing for larger projects
    if len(source_files) >= MIN_FILES_FOR_PARALLEL:
        max_workers = min(os.cpu_count() or 4, 8)
        root_str = str(root)
        args_list = [(src_file, root_str, language) for src_file in source_files]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            partial_indexes = list(executor.map(_index_single_file, args_list))

        # Merge all partial indexes
        for partial in partial_indexes:
            index.update(partial)
    else:
        # Sequential processing for small projects (avoids process spawn overhead)
        for src_file in source_files:
            src_path = Path(src_file)
            rel_path = src_path.relative_to(root)

            # Derive module name from file path
            # e.g., pkg/core.py -> pkg.core, utils.ts -> utils
            module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            module_name = (
                "/".join(module_parts)
                if language == "typescript"
                else ".".join(module_parts)
            )

            # Also track the simple module name (last component)
            simple_module = rel_path.stem

            if language == "python":
                _index_python_file(
                    src_path, rel_path, module_name, simple_module, index
                )
            elif language == "typescript":
                _index_typescript_file(
                    src_path, rel_path, module_name, simple_module, index
                )
            elif language == "go":
                _index_go_file(src_path, rel_path, module_name, simple_module, index)
            elif language == "rust":
                _index_rust_file(src_path, rel_path, module_name, simple_module, index)
            elif language == "java":
                _index_java_file(src_path, rel_path, module_name, simple_module, index)
            elif language == "c":
                _index_c_file(src_path, rel_path, module_name, simple_module, index)

    return index


def _index_python_file(
    src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict
):
    """Index functions and classes from a Python file."""
    try:
        source = src_path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Map both full and simple module names
            index[(module_name, node.name)] = str(rel_path)
            index[(simple_module, node.name)] = str(rel_path)
            # Also index with string key for convenience
            index[f"{module_name}.{node.name}"] = str(rel_path)
            index[f"{simple_module}.{node.name}"] = str(rel_path)
        elif isinstance(node, ast.ClassDef):
            # Track class definitions too (for instantiation calls)
            index[(module_name, node.name)] = str(rel_path)
            index[(simple_module, node.name)] = str(rel_path)
            index[f"{module_name}.{node.name}"] = str(rel_path)
            index[f"{simple_module}.{node.name}"] = str(rel_path)


def _index_typescript_file(
    src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict
):
    """Index functions and classes from a TypeScript file."""
    parser = get_parser("typescript")
    if parser is None:
        return

    try:
        source = src_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}/{name}"] = str(rel_path)
        index[f"{simple_module}/{name}"] = str(rel_path)

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        # Handle export statements - look inside them
        if node.type == "export_statement":
            for child in node.children:
                walk_tree(child, depth + 1, max_depth)
            return

        # Function declarations
        if node.type in ("function_declaration", "method_definition"):
            name = _get_ts_node_name(node, source)
            if name:
                add_to_index(name)

        # Arrow functions assigned to variables: const foo = () => {}
        elif node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    name = None
                    has_arrow = False
                    for vc in child.children:
                        if vc.type == "identifier":
                            name = source[vc.start_byte : vc.end_byte].decode("utf-8")
                        elif vc.type == "arrow_function":
                            has_arrow = True
                    if name and has_arrow:
                        add_to_index(name)

        # Class declarations
        elif node.type == "class_declaration":
            name = _get_ts_node_name(node, source)
            if name:
                add_to_index(name)

        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)


def _get_ts_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a TypeScript AST node."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier", "type_identifier"):
            return source[child.start_byte : child.end_byte].decode("utf-8")
    return None


def _index_go_file(
    src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict
):
    """Index functions, types, and methods from a Go file."""
    parser = get_parser("go")
    if parser is None:
        return

    try:
        source = src_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}/{name}"] = str(rel_path)
        index[f"{simple_module}/{name}"] = str(rel_path)

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        # Function declarations
        if node.type == "function_declaration":
            name = _get_go_node_name(node, source)
            if name:
                add_to_index(name)

        # Method declarations (function with receiver)
        elif node.type == "method_declaration":
            name = _get_go_node_name(node, source)
            if name:
                add_to_index(name)
                # Also try to get the receiver type for full name
                receiver_type = _get_go_receiver_type(node, source)
                if receiver_type:
                    add_to_index(f"{receiver_type}.{name}")

        # Type declarations (struct, interface)
        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name = _get_go_node_name(child, source)
                    if name:
                        add_to_index(name)

        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)


def _get_go_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a Go AST node."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "field_identifier"):
            return source[child.start_byte : child.end_byte].decode("utf-8")
    return None


def _get_go_receiver_type(node, source: bytes) -> str | None:
    """Get the receiver type from a Go method declaration."""
    for child in node.children:
        if child.type == "parameter_list":
            # First parameter list is the receiver
            for param in child.children:
                if param.type == "parameter_declaration":
                    for pc in param.children:
                        if pc.type == "pointer_type":
                            for pt in pc.children:
                                if pt.type == "type_identifier":
                                    return source[pt.start_byte : pt.end_byte].decode(
                                        "utf-8"
                                    )
                        elif pc.type == "type_identifier":
                            return source[pc.start_byte : pc.end_byte].decode("utf-8")
            break
    return None


def _index_rust_file(
    src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict
):
    """Index functions, structs, and impl blocks from a Rust file."""
    parser = get_parser("rust")
    if parser is None:
        return

    try:
        source = src_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}.{name}"] = str(rel_path)
        index[f"{simple_module}.{name}"] = str(rel_path)

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        # Function definitions
        if node.type == "function_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Struct definitions
        elif node.type == "struct_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Enum definitions
        elif node.type == "enum_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Trait definitions
        elif node.type == "trait_item":
            name = _get_rust_node_name(node, source)
            if name:
                add_to_index(name)

        # Impl blocks - index methods
        elif node.type == "impl_item":
            type_name = None
            for child in node.children:
                if child.type == "type_identifier":
                    type_name = source[child.start_byte : child.end_byte].decode(
                        "utf-8"
                    )
                    break
            # Index methods within impl block
            for child in node.children:
                if child.type == "declaration_list":
                    for item in child.children:
                        if item.type == "function_item":
                            method_name = _get_rust_node_name(item, source)
                            if method_name:
                                # Index as both bare name and Type::method
                                add_to_index(method_name)
                                if type_name:
                                    add_to_index(f"{type_name}::{method_name}")

        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)


def _get_rust_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a Rust AST node."""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte : child.end_byte].decode("utf-8")
        elif child.type == "type_identifier":
            return source[child.start_byte : child.end_byte].decode("utf-8")
    return None


def _index_java_file(
    src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict
):
    """Index methods and classes from a Java file."""
    parser = get_parser("java")
    if parser is None:
        return

    try:
        source = src_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}.{name}"] = str(rel_path)
        index[f"{simple_module}.{name}"] = str(rel_path)

    current_class = None

    def walk_tree(node, depth=0, max_depth=500):
        nonlocal current_class

        if depth > max_depth:
            return

        # Class declarations
        if node.type == "class_declaration":
            class_name = _get_java_node_name(node, source)
            if class_name:
                add_to_index(class_name)
                old_class = current_class
                current_class = class_name
                # Process class body
                for child in node.children:
                    walk_tree(child, depth + 1, max_depth)
                current_class = old_class
                return  # Already processed children

        # Interface declarations
        elif node.type == "interface_declaration":
            interface_name = _get_java_node_name(node, source)
            if interface_name:
                add_to_index(interface_name)

        # Method declarations
        elif node.type == "method_declaration":
            name = _get_java_node_name(node, source)
            if name:
                add_to_index(name)
                # Also index as Class.method if we have a class context
                if current_class:
                    add_to_index(f"{current_class}.{name}")

        # Constructor declarations
        elif node.type == "constructor_declaration":
            name = _get_java_node_name(node, source)
            if name:
                add_to_index(name)

        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)


def _get_java_node_name(node, source: bytes) -> str | None:
    """Get the name identifier from a Java AST node."""
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte : child.end_byte].decode("utf-8")
    return None


def _index_c_file(
    src_path: Path, rel_path: Path, module_name: str, simple_module: str, index: dict
):
    """Index functions from a C file."""
    parser = get_parser("c")
    if parser is None:
        return

    try:
        source = src_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return

    def add_to_index(name: str):
        """Helper to add a name to the index."""
        index[(module_name, name)] = str(rel_path)
        index[(simple_module, name)] = str(rel_path)
        index[f"{module_name}.{name}"] = str(rel_path)
        index[f"{simple_module}.{name}"] = str(rel_path)

    def walk_tree(node, depth=0, max_depth=500):
        if depth > max_depth:
            return
        # Function definitions
        if node.type == "function_definition":
            name = _get_c_node_name(node, source)
            if name:
                add_to_index(name)

        for child in node.children:
            walk_tree(child, depth + 1, max_depth)

    walk_tree(tree.root_node)


def _get_c_node_name(node, source: bytes) -> str | None:
    """Get the function name from a C function_definition node."""
    for child in node.children:
        if child.type == "function_declarator":
            for dc in child.children:
                if dc.type == "identifier":
                    return source[dc.start_byte : dc.end_byte].decode("utf-8")
        elif child.type == "pointer_declarator":
            # Pointer return type like int* func()
            for pc in child.children:
                if pc.type == "function_declarator":
                    for dc in pc.children:
                        if dc.type == "identifier":
                            return source[dc.start_byte : dc.end_byte].decode("utf-8")
    return None


class CallVisitor(ast.NodeVisitor):
    """AST visitor that extracts function calls and references from a function body."""

    def __init__(self, defined_funcs: set[str] | None = None):
        self.calls: list[str] = []
        self.attr_calls: list[tuple[str, str]] = []  # (obj, method) pairs
        self.refs: list[str] = []  # Function references (higher-order usage)
        self._defined_funcs = defined_funcs or set()
        self._in_call = False  # Track if we're inside a Call node

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            # Direct call: func()
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Attribute call: obj.method() or module.func()
            if isinstance(node.func.value, ast.Name):
                self.attr_calls.append((node.func.value.id, node.func.attr))

        # Visit arguments - function references passed as args
        self._in_call = True
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
        self._in_call = False

        # Don't call generic_visit - we handled children manually

    def visit_Name(self, node: ast.Name):
        # Track function references (not calls) when used as values
        # Only track if it matches a known function name
        if node.id in self._defined_funcs and node.id not in self.calls:
            self.refs.append(node.id)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        # Track function references in dict values: {"key": func}
        for value in node.values:
            if isinstance(value, ast.Name) and value.id in self._defined_funcs:
                if value.id not in self.refs:
                    self.refs.append(value.id)
        self.generic_visit(node)

    def visit_List(self, node: ast.List):
        # Track function references in lists: [func1, func2]
        for elt in node.elts:
            if isinstance(elt, ast.Name) and elt.id in self._defined_funcs:
                if elt.id not in self.refs:
                    self.refs.append(elt.id)
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple):
        # Track function references in tuples: (func1, func2)
        for elt in node.elts:
            if isinstance(elt, ast.Name) and elt.id in self._defined_funcs:
                if elt.id not in self.refs:
                    self.refs.append(elt.id)
        self.generic_visit(node)


def _extract_file_calls(
    file_path: Path, root: Path
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    try:
        source = file_path.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return {}

    calls_by_func = {}

    # Collect all function names defined in this file (for intra-file calls)
    defined_funcs = set()
    defined_classes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined_classes.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visitor = CallVisitor(defined_funcs=defined_funcs)
            visitor.visit(node)

            calls = []
            for call in visitor.calls:
                if call in defined_funcs or call in defined_classes:
                    calls.append(("intra", call))
                else:
                    calls.append(("direct", call))

            for obj, method in visitor.attr_calls:
                calls.append(("attr", f"{obj}.{method}"))

            # Add function references (higher-order usage)
            for ref in visitor.refs:
                if ref in defined_funcs:
                    calls.append(("ref", ref))

            calls_by_func[node.name] = calls

    # Also scan module-level code for function calls and references
    # This catches: COMMANDS = {"key": func}, if __name__ == "__main__", etc.
    module_calls = []
    for node in tree.body:
        # Skip function/class definitions - we handle those above
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Visit module-level statements for function references and calls
        visitor = CallVisitor(defined_funcs=defined_funcs)
        visitor.visit(node)

        # Add intra-file references
        for ref in visitor.refs:
            if ref in defined_funcs:
                module_calls.append(("ref", ref))

        # Add ALL calls (both intra-file and external imports)
        for call in visitor.calls:
            if call in defined_funcs:
                module_calls.append(("intra", call))
            else:
                module_calls.append(("direct", call))  # Could be imported function

    # Add module-level calls from a synthetic "<module>" function
    if module_calls:
        calls_by_func["<module>"] = module_calls

    return calls_by_func


def _extract_ts_file_calls(
    file_path: Path, root: Path
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a TypeScript file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    parser = get_parser("typescript")
    if parser is None:
        return {}

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function/class names
    def collect_definitions(node):
        if node.type in ("function_declaration", "class_declaration"):
            name = _get_ts_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    for vc in child.children:
                        if vc.type == "identifier":
                            defined_names.add(
                                source[vc.start_byte : vc.end_byte].decode("utf-8")
                            )
                            break
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the callee
                for child in node.children:
                    if child.type == "identifier":
                        callee = source[child.start_byte : child.end_byte].decode(
                            "utf-8"
                        )
                        if callee in defined_names:
                            calls.append(("intra", callee))
                        else:
                            calls.append(("direct", callee))
                        break
                    elif child.type == "member_expression":
                        # obj.method() call
                        obj_name = None
                        obj_is_this = False
                        method_name = None
                        for mc in child.children:
                            if mc.type == "this":
                                obj_is_this = True
                            elif mc.type == "identifier" and obj_name is None:
                                obj_name = source[mc.start_byte : mc.end_byte].decode(
                                    "utf-8"
                                )
                            elif mc.type == "property_identifier":
                                method_name = source[
                                    mc.start_byte : mc.end_byte
                                ].decode("utf-8")

                        if obj_is_this and method_name:
                            # this.method() - treat as intra-file call to the method
                            calls.append(("intra", method_name))
                        elif obj_name and method_name:
                            calls.append(("attr", f"{obj_name}.{method_name}"))
                        break

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    def process_functions(node):
        # Handle export statements - look inside them
        if node.type == "export_statement":
            for child in node.children:
                process_functions(child)
            return

        if node.type == "function_declaration":
            name = _get_ts_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        elif node.type == "lexical_declaration":
            # Handle arrow functions: const foo = () => {}
            for child in node.children:
                if child.type == "variable_declarator":
                    name = None
                    arrow_node = None
                    for vc in child.children:
                        if vc.type == "identifier":
                            name = source[vc.start_byte : vc.end_byte].decode("utf-8")
                        elif vc.type == "arrow_function":
                            arrow_node = vc
                    if name and arrow_node:
                        calls_by_func[name] = extract_calls_from_func(arrow_node, name)

        elif node.type == "class_declaration":
            class_name = _get_ts_node_name(node, source)
            if class_name:
                # Process methods
                for child in node.children:
                    if child.type == "class_body":
                        for body_child in child.children:
                            if body_child.type == "method_definition":
                                method_name = _get_ts_node_name(body_child, source)
                                if method_name:
                                    full_name = f"{class_name}.{method_name}"
                                    calls_by_func[full_name] = extract_calls_from_func(
                                        body_child, full_name
                                    )

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_go_file_calls(
    file_path: Path, root: Path
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a Go file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    parser = get_parser("go")
    if parser is None:
        return {}

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function/type names
    def collect_definitions(node):
        if node.type == "function_declaration":
            name = _get_go_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "method_declaration":
            name = _get_go_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name = _get_go_node_name(child, source)
                    if name:
                        defined_names.add(name)
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the callee - first child is the function being called
                func_child = node.children[0] if node.children else None
                if func_child:
                    if func_child.type == "identifier":
                        callee = source[
                            func_child.start_byte : func_child.end_byte
                        ].decode("utf-8")
                        if callee in defined_names:
                            calls.append(("intra", callee))
                        else:
                            calls.append(("direct", callee))
                    elif func_child.type == "selector_expression":
                        # pkg.Func() or obj.Method() call
                        parts = []
                        for sc in func_child.children:
                            if sc.type == "identifier":
                                parts.append(
                                    source[sc.start_byte : sc.end_byte].decode("utf-8")
                                )
                            elif sc.type == "field_identifier":
                                parts.append(
                                    source[sc.start_byte : sc.end_byte].decode("utf-8")
                                )
                        if len(parts) >= 2:
                            obj, method = parts[0], parts[-1]
                            # Check if method is defined locally
                            if method in defined_names:
                                calls.append(("intra", method))
                            else:
                                calls.append(("attr", f"{obj}.{method}"))

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    def process_functions(node):
        if node.type == "function_declaration":
            name = _get_go_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        elif node.type == "method_declaration":
            name = _get_go_node_name(node, source)
            receiver_type = _get_go_receiver_type(node, source)
            if name:
                full_name = f"{receiver_type}.{name}" if receiver_type else name
                calls_by_func[full_name] = extract_calls_from_func(node, full_name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_rust_file_calls(
    file_path: Path, root: Path
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a Rust file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    parser = get_parser("rust")
    if parser is None:
        return {}

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function/struct names
    def collect_definitions(node):
        if node.type == "function_item":
            name = _get_rust_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type in ("struct_item", "enum_item", "trait_item"):
            name = _get_rust_node_name(node, source)
            if name:
                defined_names.add(name)
        elif node.type == "impl_item":
            # Collect method names from impl blocks
            for child in node.children:
                if child.type == "declaration_list":
                    for item in child.children:
                        if item.type == "function_item":
                            name = _get_rust_node_name(item, source)
                            if name:
                                defined_names.add(name)
        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the callee
                for child in node.children:
                    if child.type == "identifier":
                        callee = source[child.start_byte : child.end_byte].decode(
                            "utf-8"
                        )
                        if callee in defined_names:
                            calls.append(("intra", callee))
                        else:
                            calls.append(("direct", callee))
                        break
                    elif child.type == "scoped_identifier":
                        # Path call: module::func() or Type::method()
                        text = source[child.start_byte : child.end_byte].decode("utf-8")
                        # Get the last segment as the function name
                        if "::" in text:
                            parts = text.rsplit("::", 1)
                            func = parts[1]
                            if func in defined_names:
                                calls.append(("intra", func))
                            else:
                                calls.append(("attr", text))
                        break
                    elif child.type == "field_expression":
                        # Method call: obj.method()
                        method_name = None
                        for fc in child.children:
                            if fc.type == "field_identifier":
                                method_name = source[
                                    fc.start_byte : fc.end_byte
                                ].decode("utf-8")
                        if method_name:
                            if method_name in defined_names:
                                calls.append(("intra", method_name))
                            else:
                                calls.append(("attr", f"self.{method_name}"))
                        break

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    def process_functions(node):
        if node.type == "function_item":
            name = _get_rust_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        elif node.type == "impl_item":
            type_name = None
            for child in node.children:
                if child.type == "type_identifier":
                    type_name = source[child.start_byte : child.end_byte].decode(
                        "utf-8"
                    )
                    break

            for child in node.children:
                if child.type == "declaration_list":
                    for item in child.children:
                        if item.type == "function_item":
                            method_name = _get_rust_node_name(item, source)
                            if method_name:
                                full_name = (
                                    f"{type_name}.{method_name}"
                                    if type_name
                                    else method_name
                                )
                                calls_by_func[full_name] = extract_calls_from_func(
                                    item, full_name
                                )

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_java_file_calls(
    file_path: Path, root: Path
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all method calls from a Java file, grouped by caller method.

    Returns:
        Dict mapping caller method name to list of (call_type, call_target) tuples
        call_type is 'direct', 'attr', or 'intra'
    """
    parser = get_parser("java")
    if parser is None:
        return {}

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()
    current_class = None

    # First pass: collect all defined method/class names
    def collect_definitions(node):
        nonlocal current_class

        if node.type == "class_declaration":
            class_name = _get_java_node_name(node, source)
            if class_name:
                defined_names.add(class_name)
                old_class = current_class
                current_class = class_name
                for child in node.children:
                    collect_definitions(child)
                current_class = old_class
                return

        elif node.type == "method_declaration":
            name = _get_java_node_name(node, source)
            if name:
                defined_names.add(name)
                if current_class:
                    defined_names.add(f"{current_class}.{name}")

        elif node.type == "constructor_declaration":
            name = _get_java_node_name(node, source)
            if name:
                defined_names.add(name)

        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each method
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "method_invocation":
                # Get the method name and object (if any)
                method_name = None
                object_name = None

                for child in node.children:
                    if child.type == "identifier":
                        # Could be method name or object
                        text = source[child.start_byte : child.end_byte].decode("utf-8")
                        if method_name is None:
                            # First identifier could be object or direct call
                            if object_name is None:
                                method_name = text
                            else:
                                method_name = text
                        else:
                            method_name = text
                    elif child.type in ("field_access", "this"):
                        # Object.method() or this.method()
                        if child.type == "this":
                            object_name = "this"
                        else:
                            object_name = source[
                                child.start_byte : child.end_byte
                            ].decode("utf-8")
                    elif child.type == "argument_list":
                        # Skip argument list
                        pass

                # Determine call type
                if method_name:
                    if method_name in defined_names:
                        calls.append(("intra", method_name))
                    elif object_name:
                        calls.append(("attr", f"{object_name}.{method_name}"))
                    else:
                        calls.append(("direct", method_name))

            # Also handle object creation as calls (new ClassName())
            elif node.type == "object_creation_expression":
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = source[child.start_byte : child.end_byte].decode(
                            "utf-8"
                        )
                        if class_name in defined_names:
                            calls.append(("intra", class_name))
                        else:
                            calls.append(("direct", class_name))
                        break

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    # Third pass: process functions
    current_class = None

    def process_functions(node):
        nonlocal current_class

        if node.type == "class_declaration":
            class_name = _get_java_node_name(node, source)
            if class_name:
                old_class = current_class
                current_class = class_name
                for child in node.children:
                    process_functions(child)
                current_class = old_class
                return

        elif node.type == "method_declaration":
            name = _get_java_node_name(node, source)
            if name:
                full_name = f"{current_class}.{name}" if current_class else name
                calls_by_func[name] = extract_calls_from_func(node, name)
                # Also store with full name
                if current_class:
                    calls_by_func[full_name] = calls_by_func[name]

        elif node.type == "constructor_declaration":
            name = _get_java_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def _extract_c_file_calls(
    file_path: Path, root: Path
) -> dict[str, list[tuple[str, str]]]:
    """
    Extract all function calls from a C file, grouped by caller function.

    Returns:
        Dict mapping caller function name to list of (call_type, call_target) tuples
        call_type is 'direct' or 'intra'
    """
    parser = get_parser("c")
    if parser is None:
        return {}

    try:
        source = file_path.read_bytes()
        tree = parser.parse(source)
    except (FileNotFoundError, Exception):
        return {}

    calls_by_func = {}
    defined_names = set()

    # First pass: collect all defined function names
    def collect_definitions(node):
        if node.type == "function_definition":
            name = _get_c_node_name(node, source)
            if name:
                defined_names.add(name)

        for child in node.children:
            collect_definitions(child)

    collect_definitions(tree.root_node)

    # Second pass: extract calls from each function
    def extract_calls_from_func(func_node, func_name: str):
        calls = []

        def visit_calls(node):
            if node.type == "call_expression":
                # Get the function name being called
                callee = None
                for child in node.children:
                    if child.type == "identifier":
                        callee = source[child.start_byte : child.end_byte].decode(
                            "utf-8"
                        )
                        break

                if callee:
                    if callee in defined_names:
                        calls.append(("intra", callee))
                    else:
                        calls.append(("direct", callee))

            for child in node.children:
                visit_calls(child)

        visit_calls(func_node)
        return calls

    # Third pass: process functions
    def process_functions(node):
        if node.type == "function_definition":
            name = _get_c_node_name(node, source)
            if name:
                calls_by_func[name] = extract_calls_from_func(node, name)

        for child in node.children:
            process_functions(child)

    process_functions(tree.root_node)
    return calls_by_func


def build_project_call_graph(
    root: str | Path, language: str = "python", use_workspace_config: bool = True
) -> ProjectCallGraph:
    """
    Build a complete project-wide call graph.

    Resolves cross-file calls by:
    1. Scanning all source files for the language
    2. Building a function index
    3. Parsing imports in each file
    4. Matching call sites to definitions

    Args:
        root: Project root directory
        language: "python" or "typescript"
        use_workspace_config: If True, loads .claude/workspace.json to scope
                             indexing to activePackages and excludePatterns.
                             Defaults to True for monorepo support.

    Returns:
        ProjectCallGraph with edges as (src_file, src_func, dst_file, dst_func)

    Raises:
        ValueError: If language is not supported for call graph analysis.
    """
    if language not in SUPPORTED_CALL_GRAPH_LANGUAGES:
        raise ValueError(
            f"Call graph analysis not supported for '{language}'. "
            f"Supported languages: {sorted(SUPPORTED_CALL_GRAPH_LANGUAGES)}"
        )

    root = Path(root)
    graph = ProjectCallGraph()

    # Load workspace config if enabled
    workspace_config = None
    if use_workspace_config:
        workspace_config = load_workspace_config(root)

    # Scan project once and reuse file list for both index building and call graph building
    file_list = scan_project(root, language, workspace_config)

    func_index = build_function_index(
        root, language, workspace_config, file_list=file_list
    )

    if language == "python":
        _build_python_call_graph(
            root, graph, func_index, workspace_config, file_list=file_list
        )
    elif language == "typescript":
        _build_typescript_call_graph(
            root, graph, func_index, workspace_config, file_list=file_list
        )
    elif language == "go":
        _build_go_call_graph(
            root, graph, func_index, workspace_config, file_list=file_list
        )
    elif language == "rust":
        _build_rust_call_graph(
            root, graph, func_index, workspace_config, file_list=file_list
        )
    elif language == "java":
        _build_java_call_graph(
            root, graph, func_index, workspace_config, file_list=file_list
        )
    elif language == "c":
        _build_c_call_graph(
            root, graph, func_index, workspace_config, file_list=file_list
        )

    return graph


def _process_python_file_for_callgraph(args: tuple[str, str]) -> dict:
    """
    Worker function for parallel call graph extraction.

    Extracts imports and function calls from a single Python file.
    Must be at module level for ProcessPoolExecutor compatibility.
    Wrapped in try/except to prevent one file's failure from crashing the pool.

    Args:
        args: Tuple of (py_file_path, root_path) as strings for pickle compatibility

    Returns:
        Dict with keys: file, rel_path, imports, calls_by_func
    """
    try:
        py_file, root_str = args
        py_path = Path(py_file)
        root = Path(root_str)

        try:
            rel_path = str(py_path.relative_to(root))
        except ValueError:
            # File is not under root (shouldn't happen, but be safe)
            return {
                "file": py_file,
                "rel_path": py_file,
                "imports": [],
                "calls_by_func": {},
            }

        imports = parse_imports(py_path)
        calls_by_func = _extract_file_calls(py_path, root)

        return {
            "file": py_file,
            "rel_path": rel_path,
            "imports": imports,
            "calls_by_func": calls_by_func,
        }
    except Exception:
        # Don't let one file's failure crash the entire pool
        # Return empty data structure so processing can continue
        py_file = args[0] if args else "<unknown>"
        return {
            "file": py_file,
            "rel_path": py_file,
            "imports": [],
            "calls_by_func": {},
        }


def _build_python_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None,
    file_list: Optional[list[str]] = None,
):
    """Build call graph for Python files.

    For projects with more than MIN_FILES_FOR_PARALLEL files, uses parallel
    extraction via ProcessPoolExecutor followed by sequential graph merge.
    """
    # Use provided file_list or scan project
    source_files = (
        file_list
        if file_list is not None
        else scan_project(root, "python", workspace_config)
    )
    source_files = list(source_files)  # Materialize for len() and reuse

    # Phase 1: Extract imports and calls from each file
    # For large projects, parallelize I/O-bound file reading + CPU-bound AST parsing
    if len(source_files) >= MIN_FILES_FOR_PARALLEL:
        max_workers = min(os.cpu_count() or 4, 8)
        args_list = [(str(f), str(root)) for f in source_files]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            file_results = list(
                executor.map(_process_python_file_for_callgraph, args_list)
            )
    else:
        # Sequential extraction for small projects (avoids process spawn overhead)
        file_results = [
            _process_python_file_for_callgraph((str(f), str(root)))
            for f in source_files
        ]

    # Phase 2: Sequential merge into graph (fast O(N) dict/set operations)
    for result in file_results:
        rel_path = result["rel_path"]
        imports = result["imports"]
        calls_by_func = result["calls_by_func"]

        # Build import resolution map
        import_map = {}
        module_imports = {}

        for imp in imports:
            if imp["is_from"]:
                module = imp["module"]
                level = imp.get("level", 0)
                # Resolve relative imports to absolute module paths
                if level > 0:
                    # Strip leading dots from module for resolution
                    module_without_dots = module.lstrip(".")
                    resolved = _resolve_python_relative_import(
                        rel_path, module_without_dots, level, str(root)
                    )
                    module = resolved
                aliases = imp.get("aliases", {})
                for name in imp["names"]:
                    alias = None
                    for alias_name, orig_name in aliases.items():
                        if orig_name == name:
                            alias = alias_name
                            break
                    if alias:
                        import_map[alias] = (module, name)
                    import_map[name] = (module, name)
            else:
                module = imp["module"]
                alias = imp.get("alias")
                if alias:
                    module_imports[alias] = module
                else:
                    module_imports[module] = module

        # Add edges to graph
        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == "intra":
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)
                elif call_type == "direct":
                    if call_target in import_map:
                        module, orig_name = import_map[call_target]
                        key = (module.split(".")[-1], orig_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, orig_name)
                        else:
                            key = (module, orig_name)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(
                                    rel_path, caller_func, dst_file, orig_name
                                )
                elif call_type == "attr":
                    parts = call_target.split(".", 1)
                    if len(parts) == 2:
                        obj, method = parts
                        if obj in module_imports:
                            module = module_imports[obj]
                            simple_module = module.split(".")[-1]
                            key = (simple_module, method)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, method)
                elif call_type == "ref":
                    # Function reference (higher-order usage) - intra-file only
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)


def _build_typescript_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None,
    file_list: Optional[list[str]] = None,
):
    """Build call graph for TypeScript files."""
    # Use provided file_list or scan project
    source_files = (
        file_list
        if file_list is not None
        else scan_project(root, "typescript", workspace_config)
    )

    for ts_file in source_files:
        ts_path = Path(ts_file)
        rel_path = str(ts_path.relative_to(root))

        # Get imports for this file
        imports = parse_ts_imports(ts_path)

        # Build import resolution map
        # For TypeScript, imports are relative paths or package names
        import_map = {}  # local_name -> (module_path, original_name)
        default_imports = {}  # local_name -> module_path
        namespace_imports = {}  # local_name -> module_path

        for imp in imports:
            module = imp["module"]
            # Resolve relative imports
            if module.startswith("."):
                # Convert relative path to file path with index.ts resolution
                module_path = _resolve_ts_import(rel_path, module, str(root))
            else:
                module_path = module

            # Named imports: import { foo, bar as baz } from "./module"
            for name in imp.get("names", []):
                import_map[name] = (module_path, name)

            # Handle aliases
            for alias, orig_name in imp.get("aliases", {}).items():
                if orig_name == "*":
                    namespace_imports[alias] = module_path
                else:
                    import_map[alias] = (module_path, orig_name)

            # Default import: import Foo from "./module"
            if imp.get("default"):
                default_imports[imp["default"]] = module_path

        # Get calls from this file
        calls_by_func = _extract_ts_file_calls(ts_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == "intra":
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == "direct":
                    if call_target in import_map:
                        module_path, orig_name = import_map[call_target]
                        # Try to find in function index
                        simple_module = Path(module_path).stem
                        key = (simple_module, orig_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, orig_name)
                    elif call_target in default_imports:
                        module_path = default_imports[call_target]
                        simple_module = Path(module_path).stem
                        # Default export often matches the module name or 'default'
                        key = (simple_module, call_target)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, call_target)

                elif call_type == "attr":
                    parts = call_target.split(".", 1)
                    if len(parts) == 2:
                        obj, method = parts
                        if obj in namespace_imports:
                            module_path = namespace_imports[obj]
                            simple_module = Path(module_path).stem
                            key = (simple_module, method)
                            if key in func_index:
                                dst_file = func_index[key]
                                graph.add_edge(rel_path, caller_func, dst_file, method)


@lru_cache(maxsize=1024)
def _resolve_ts_import(from_file: str, import_path: str, root: str = "") -> str:
    """Resolve a relative TypeScript import path to a file path.

    Handles TypeScript module resolution order:
    1. Exact file with .ts extension
    2. Exact file with .tsx extension
    3. Directory with index.ts
    4. Directory with index.tsx

    Args:
        from_file: The file containing the import (relative path)
        import_path: The import path (e.g., './utils', '../config')
        root: Project root directory for file existence checks

    Returns:
        Resolved file path (relative to root), or base path if no file found
    """
    if not import_path.startswith("."):
        return import_path  # External package, return as-is

    from_dir = str(Path(from_file).parent)
    if from_dir == ".":
        from_dir = ""

    # Handle ./ and ../
    if import_path.startswith("./"):
        resolved = import_path[2:]
        if from_dir:
            resolved = f"{from_dir}/{resolved}"
    elif import_path.startswith("../"):
        parts = from_dir.split("/") if from_dir else []
        import_parts = import_path.split("/")
        while import_parts and import_parts[0] == "..":
            import_parts.pop(0)
            if parts:
                parts.pop()
        resolved = "/".join(parts + import_parts)
    else:
        resolved = import_path

    # Check for actual file existence in TypeScript resolution order
    if root:
        root_path = Path(root)
        target = root_path / resolved

        # TypeScript module resolution precedence
        candidates = [
            target.with_suffix(".ts"),
            target.with_suffix(".tsx"),
            target / "index.ts",
            target / "index.tsx",
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate.relative_to(root_path))

    return resolved


@lru_cache(maxsize=1024)
def _resolve_python_relative_import(
    from_file: str, import_module: str, level: int, root: str = ""
) -> str:
    """Resolve a Python relative import to a module path.

    Handles Python's relative import semantics:
    - level=1 (from . import x): import from same package
    - level=2 (from .. import x): import from parent package
    - level=N: go up N-1 directories from current file's package

    Args:
        from_file: The file containing the import (relative path)
        import_module: The module part after dots (e.g., 'utils' from 'from .utils import')
        level: Number of dots (1 for '.', 2 for '..', etc.)
        root: Project root for file existence checks

    Returns:
        Resolved module path (e.g., 'pkg.subpkg.utils') or original if not relative
    """
    if level == 0:
        # Absolute import - return as-is
        return import_module

    # Get directory parts from the importing file
    from_path = Path(from_file)
    parts = list(from_path.parts[:-1])  # Remove filename

    # Go up (level - 1) directories for relative import
    # level=1 means current package, level=2 means parent package
    for _ in range(level - 1):
        if parts:
            parts.pop()

    # Add the import module to the path
    if import_module:
        parts.extend(import_module.split("."))

    # Build the resolved module name
    resolved_module = ".".join(parts) if parts else import_module

    # Verify file exists if root provided
    if root and resolved_module:
        root_path = Path(root)
        # Try as a module file
        module_file = root_path / "/".join(parts)
        candidates = [
            module_file.with_suffix(".py"),
            module_file / "__init__.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                # Return relative path stem as module name
                rel = candidate.relative_to(root_path)
                if rel.name == "__init__.py":
                    return ".".join(rel.parts[:-1])
                return ".".join(rel.parts[:-1]) + "." + rel.stem if rel.parts[:-1] else rel.stem

    return resolved_module


def _build_name_to_files_index(func_index: dict) -> dict:
    """
    Build reverse index for O(1) lookup by function name.

    Transforms func_index {(mod, name): file_path, ...} into
    {name: [(mod, file_path), ...]} for efficient name-based lookup.

    Args:
        func_index: Dictionary mapping (module, func_name) tuples to file paths

    Returns:
        Dictionary mapping function names to list of (module, file_path) tuples
    """
    name_index: dict[str, list[tuple[str, str]]] = {}
    for key, file_path in func_index.items():
        if isinstance(key, tuple) and len(key) == 2:
            mod, name = key
            if name not in name_index:
                name_index[name] = []
            name_index[name].append((mod, file_path))
    return name_index


def _build_go_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None,
    file_list: Optional[list[str]] = None,
):
    """Build call graph for Go files."""
    # Build reverse index once for O(1) lookup by function name
    name_index = _build_name_to_files_index(func_index)

    # Use provided file_list or scan project
    source_files = (
        file_list
        if file_list is not None
        else scan_project(root, "go", workspace_config)
    )

    for go_file in source_files:
        go_path = Path(go_file)
        rel_path = str(go_path.relative_to(root))

        # Get imports for this file
        imports = parse_go_imports(go_path)

        # Build import resolution map
        # For Go, imports are package paths with optional aliases
        package_imports = {}  # local_name -> package_path

        for imp in imports:
            module = imp["module"]
            alias = imp.get("alias")

            # Resolve imports (relative ./pkg or module-prefixed)
            module_path = _resolve_go_import(rel_path, module, str(root))

            # Determine the local name (alias or last path component)
            if alias:
                local_name = alias
            else:
                # Use last component of path as package name
                local_name = module.rstrip("/").split("/")[-1]

            package_imports[local_name] = module_path

        # Get calls from this file
        calls_by_func = _extract_go_file_calls(go_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == "intra":
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == "attr":
                    parts = call_target.split(".", 1)
                    if len(parts) == 2:
                        pkg, func_name = parts
                        if pkg in package_imports:
                            pkg_path = package_imports[pkg]
                            # O(1) lookup by function name instead of O(n) scan
                            candidates = name_index.get(func_name, [])
                            for mod, file_path in candidates:
                                # Check if this file is in the right package
                                if pkg_path.lstrip("./") in file_path or mod == pkg:
                                    graph.add_edge(
                                        rel_path, caller_func, file_path, func_name
                                    )
                                    break


@lru_cache(maxsize=128)
def _get_go_module_path(root: str) -> Optional[str]:
    """Parse go.mod to get the module path.

    Args:
        root: Project root directory (as string for caching)

    Returns:
        Module path (e.g., 'github.com/user/repo') or None if go.mod not found
    """
    go_mod = Path(root) / "go.mod"
    if not go_mod.exists():
        return None

    try:
        content = go_mod.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("module "):
                # Handle: module github.com/user/repo
                return line.split()[1]
    except (OSError, IndexError):
        pass

    return None


@lru_cache(maxsize=1024)
def _resolve_go_import(from_file: str, import_path: str, root: str = "") -> str:
    """Resolve a Go import path to a directory path.

    Handles:
    1. Relative imports (./pkg, ../pkg)
    2. Module-prefixed imports (github.com/user/repo/internal/pkg)

    Args:
        from_file: The file containing the import (relative path)
        import_path: The import path
        root: Project root directory for go.mod parsing

    Returns:
        Resolved directory path (relative to root for internal packages)
    """
    # Check for module-prefixed internal imports first
    if root and not import_path.startswith("."):
        module_path = _get_go_module_path(root)
        if module_path and import_path.startswith(module_path):
            # Internal package - strip module prefix to get relative path
            rel_path = import_path[len(module_path) :].lstrip("/")
            if rel_path:
                # Verify the directory exists
                pkg_dir = Path(root) / rel_path
                if pkg_dir.exists() and pkg_dir.is_dir():
                    return rel_path
            return rel_path if rel_path else "."

    from_dir = str(Path(from_file).parent)
    if from_dir == ".":
        from_dir = ""

    # Handle ./ and ../
    if import_path.startswith("./"):
        resolved = import_path[2:]
        if from_dir:
            resolved = f"{from_dir}/{resolved}"
    elif import_path.startswith("../"):
        parts = from_dir.split("/") if from_dir else []
        import_parts = import_path.split("/")
        while import_parts and import_parts[0] == "..":
            import_parts.pop(0)
            if parts:
                parts.pop()
        resolved = "/".join(parts + import_parts)
    else:
        # External package, return as-is
        resolved = import_path

    return resolved


def _build_rust_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None,
    file_list: Optional[list[str]] = None,
):
    """Build call graph for Rust files."""
    # Use provided file_list or scan project
    source_files = (
        file_list
        if file_list is not None
        else scan_project(root, "rust", workspace_config)
    )

    for rs_file in source_files:
        rs_path = Path(rs_file)
        rel_path = str(rs_path.relative_to(root))

        # Get imports for this file
        imports = parse_rust_imports(rs_path)

        # Build import resolution map
        # For Rust, use statements map names to modules
        import_map = {}  # local_name -> (module_path, original_name)
        mod_imports = {}  # mod_name -> potential file path

        for imp in imports:
            module = imp["module"]
            names = imp["names"]

            if imp.get("is_mod"):
                # mod declaration: mod utils; -> maps to utils.rs or utils/mod.rs
                mod_name = module
                # Try to find the file
                parent_dir = rs_path.parent
                mod_file = parent_dir / f"{mod_name}.rs"
                if mod_file.exists():
                    mod_imports[mod_name] = str(mod_file.relative_to(root))
                else:
                    mod_dir_file = parent_dir / mod_name / "mod.rs"
                    if mod_dir_file.exists():
                        mod_imports[mod_name] = str(mod_dir_file.relative_to(root))
            else:
                # use declaration
                # Resolve crate::, self::, super:: prefixes
                resolved_module = _resolve_rust_module(module, rel_path, root)
                aliases = imp.get("aliases", {})

                for name in names:
                    if name == "*":
                        # Glob import - can't resolve specific names
                        continue
                    # Map original name to itself
                    import_map[name] = (resolved_module, name)

                # Also map aliases to their original names
                # e.g., `use crate::foo as bar;` -> import_map['bar'] = (module, 'foo')
                for alias, orig_name in aliases.items():
                    import_map[alias] = (resolved_module, orig_name)

        # Get calls from this file
        calls_by_func = _extract_rust_file_calls(rs_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == "intra":
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == "direct":
                    if call_target in import_map:
                        module_path, orig_name = import_map[call_target]
                        # Try to find in function index
                        simple_module = Path(module_path).stem if module_path else ""
                        key = (simple_module, orig_name)
                        if key in func_index:
                            dst_file = func_index[key]
                            graph.add_edge(rel_path, caller_func, dst_file, orig_name)

                elif call_type == "attr":
                    # Scoped call like module::func or Type::method
                    if "::" in call_target:
                        parts = call_target.split("::")
                        func_name = parts[-1]
                        module_prefix = parts[0]

                        # Check if it's a mod import
                        if module_prefix in mod_imports:
                            dst_file = mod_imports[module_prefix]
                            simple_module = Path(dst_file).stem
                            key = (simple_module, func_name)
                            if key in func_index:
                                graph.add_edge(
                                    rel_path, caller_func, func_index[key], func_name
                                )
                        else:
                            # Try to find in function index by simple name
                            key = (module_prefix, func_name)
                            if key in func_index:
                                graph.add_edge(
                                    rel_path, caller_func, func_index[key], func_name
                                )


@lru_cache(maxsize=1024)
def _resolve_rust_module(module: str, from_file: str, root: Path) -> str:
    """
    Resolve a Rust module path to a potential file path.

    Handles:
    - crate:: -> project root
    - self:: -> current module
    - super:: -> parent module
    """
    from_path = Path(from_file)
    from_dir = from_path.parent

    if module.startswith("crate::"):
        # crate:: refers to the crate root
        remainder = module[7:]  # Strip "crate::"
        parts = remainder.split("::")
        return "/".join(parts)

    elif module.startswith("self::"):
        # self:: refers to current module
        remainder = module[6:]  # Strip "self::"
        parts = remainder.split("::")
        if from_dir == Path("."):
            return "/".join(parts)
        return str(from_dir / "/".join(parts))

    elif module.startswith("super::"):
        # super:: refers to parent module
        remainder = module[7:]  # Strip "super::"
        parts = remainder.split("::")
        parent = from_dir.parent if from_dir != Path(".") else Path(".")
        return str(parent / "/".join(parts))

    else:
        # External crate or std library - return as is
        return module.replace("::", "/")


def _build_java_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None,
    file_list: Optional[list[str]] = None,
):
    """Build call graph for Java files."""
    # Build reverse index once for O(1) lookup by function name
    name_index = _build_name_to_files_index(func_index)

    # Use provided file_list or scan project
    source_files = (
        file_list
        if file_list is not None
        else scan_project(root, "java", workspace_config)
    )

    for java_file in source_files:
        java_path = Path(java_file)
        rel_path = str(java_path.relative_to(root))

        # Get imports for this file
        imports = parse_java_imports(java_path)

        # Build import resolution map
        # For Java, imports are fully qualified class names
        import_map = {}  # simple_name -> full_module

        for imp in imports:
            module = imp["module"]
            is_wildcard = imp.get("is_wildcard", False)

            if is_wildcard:
                # Wildcard import - can't resolve specific names easily
                # Store the package prefix for later matching
                package = module.rstrip(".*")
                import_map[f"*:{package}"] = package
            else:
                # Get simple name from full import
                # e.g., java.util.List -> List
                simple_name = module.split(".")[-1]
                import_map[simple_name] = module

        # Get calls from this file
        calls_by_func = _extract_java_file_calls(java_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == "intra":
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == "direct":
                    # Direct call might be to a same-package class or an imported one
                    # O(1) lookup by function name instead of O(n) scan
                    candidates = name_index.get(call_target, [])
                    for mod, file_path in candidates:
                        graph.add_edge(rel_path, caller_func, file_path, call_target)
                        break

                elif call_type == "attr":
                    # Object.method() call
                    if "." in call_target:
                        parts = call_target.split(".")
                        method_name = parts[-1]

                        # O(1) lookup by method name instead of O(n) scan
                        candidates = name_index.get(method_name, [])
                        for mod, file_path in candidates:
                            graph.add_edge(
                                rel_path, caller_func, file_path, method_name
                            )
                            break


def _build_c_call_graph(
    root: Path,
    graph: ProjectCallGraph,
    func_index: dict,
    workspace_config: Optional[WorkspaceConfig] = None,
    file_list: Optional[list[str]] = None,
):
    """Build call graph for C files."""
    # Build reverse index once for O(1) lookup by function name
    name_index = _build_name_to_files_index(func_index)

    # Use provided file_list or scan project
    source_files = (
        file_list
        if file_list is not None
        else scan_project(root, "c", workspace_config)
    )

    for c_file in source_files:
        c_path = Path(c_file)
        rel_path = str(c_path.relative_to(root))

        # Get includes for this file
        includes = parse_c_imports(c_path)

        # Build include resolution map
        # For C, includes are header file paths
        include_map = {}  # header_name -> header_path

        for inc in includes:
            module = inc["module"]
            # Map the header file name to its path
            # e.g., "utils.h" -> "utils.h"
            header_name = module.split("/")[-1] if "/" in module else module
            include_map[header_name] = module

        # Get calls from this file
        calls_by_func = _extract_c_file_calls(c_path, root)

        for caller_func, calls in calls_by_func.items():
            for call_type, call_target in calls:
                if call_type == "intra":
                    # Intra-file call
                    graph.add_edge(rel_path, caller_func, rel_path, call_target)

                elif call_type == "direct":
                    # O(1) lookup by function name instead of O(n) scan
                    candidates = name_index.get(call_target, [])
                    for mod, file_path in candidates:
                        graph.add_edge(rel_path, caller_func, file_path, call_target)
                        break
