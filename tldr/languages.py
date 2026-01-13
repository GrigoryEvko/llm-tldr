"""
Language detection and file extension configuration.

Single source of truth for language-extension mappings used throughout tldr-code.
"""

from pathlib import Path

# Extension to language mapping (comprehensive)
# Used for: detecting language from file path
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # TypeScript/JavaScript
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    # Systems languages
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    # JVM languages
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
    # Other languages
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".cs": "csharp",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
}

# Language to extensions mapping (reverse of above)
# Used for: scanning projects for files of a specific language
LANGUAGE_EXTENSIONS: dict[str, set[str]] = {
    "python": {".py", ".pyi"},
    "typescript": {".ts", ".tsx"},
    "javascript": {".js", ".jsx"},
    "go": {".go"},
    "rust": {".rs"},
    "c": {".c", ".h"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hh"},
    "java": {".java"},
    "kotlin": {".kt", ".kts"},
    "scala": {".scala", ".sc"},
    "ruby": {".rb"},
    "php": {".php"},
    "swift": {".swift"},
    "csharp": {".cs"},
    "lua": {".lua"},
    "elixir": {".ex", ".exs"},
}

# Primary extension for each language (used when creating files)
LANGUAGE_PRIMARY_EXTENSION: dict[str, str] = {
    "python": ".py",
    "typescript": ".ts",
    "javascript": ".js",
    "go": ".go",
    "rust": ".rs",
    "c": ".c",
    "cpp": ".cpp",
    "java": ".java",
    "kotlin": ".kt",
    "scala": ".scala",
    "ruby": ".rb",
    "php": ".php",
    "swift": ".swift",
    "csharp": ".cs",
    "lua": ".lua",
    "elixir": ".ex",
}

# Default language when detection fails
DEFAULT_LANGUAGE = "python"


def detect_language(file_path: str | Path) -> str | None:
    """Detect language from file extension.

    Args:
        file_path: Path to the source file (string or Path object)

    Returns:
        Language name if recognized, None otherwise
    """
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def detect_language_with_default(file_path: str | Path) -> str:
    """Detect language from file extension, returning default if unknown.

    Args:
        file_path: Path to the source file (string or Path object)

    Returns:
        Language name (defaults to 'python' if unknown)
    """
    return detect_language(file_path) or DEFAULT_LANGUAGE


def get_extensions(language: str) -> set[str]:
    """Get all file extensions for a language.

    Args:
        language: Language name (e.g., "python", "typescript")

    Returns:
        Set of extensions (e.g., {".py", ".pyi"}), empty set if unknown
    """
    return LANGUAGE_EXTENSIONS.get(language, set())


def get_extensions_with_default(language: str) -> set[str]:
    """Get file extensions for a language, with fallback to Python.

    Args:
        language: Language name (e.g., "python", "typescript")

    Returns:
        Set of extensions, defaults to {".py"} if unknown language
    """
    return LANGUAGE_EXTENSIONS.get(language, {".py"})


def get_primary_extension(language: str) -> str:
    """Get the primary/canonical file extension for a language.

    Args:
        language: Language name (e.g., "python", "typescript")

    Returns:
        Primary extension (e.g., ".py"), defaults to ".py" if unknown
    """
    return LANGUAGE_PRIMARY_EXTENSION.get(language, ".py")
