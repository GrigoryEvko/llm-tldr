# Contributing to llm-tldr

Thank you for considering contributing to llm-tldr! We welcome contributions of all kinds.

## Before You Start

- **Check existing issues** - Someone may already be working on it
- **Open an issue first** for large changes to discuss the approach

## Pull Request Guidelines

### Keep PRs Focused

Each pull request should address **one logical change**. This helps with:
- Faster, more thorough review
- Easier rollback if issues arise
- Cleaner git history

If you have multiple unrelated changes, please submit them as separate PRs.

### PR Checklist

Before submitting:

1. **Rebase on `main`** to avoid merge conflicts
2. **Run tests**: `pytest tests/`
3. **Run linter**: `ruff check tldr/`
4. **Update docs** if you changed public APIs

### Commit Messages

Use conventional commits:
```
fix: description of bug fix
feat: description of new feature
perf: description of performance improvement
docs: description of documentation change
```

## Development Setup

```bash
git clone https://github.com/parcadei/llm-tldr.git
cd llm-tldr
uv venv && uv pip install -e ".[dev]"
```

## Questions?

Open an issue - we're happy to help!
