"""Salsa-style query memoization for TLDR (P5).

Salsa is rust-analyzer's incremental computation framework. Key concepts:

1. **Queries as Functions**: Everything is a query with automatic memoization
2. **Automatic Dependency Tracking**: Queries record which other queries they call
3. **Minimal Re-computation**: Only affected queries re-run on change

Example usage:
    from tldr.salsa import SalsaDB, salsa_query

    @salsa_query
    def read_file(db: SalsaDB, path: str) -> str:
        return db.get_file(path)

    @salsa_query
    def parse_file(db: SalsaDB, path: str) -> dict:
        content = db.query(read_file, db, path)
        return parse(content)

    db = SalsaDB()
    db.set_file("auth.py", "def login(): pass")
    result = db.query(parse_file, db, "auth.py")

    # When file changes, dependent queries auto-invalidate
    db.set_file("auth.py", "def login(): pass\\ndef logout(): pass")
    result = db.query(parse_file, db, "auth.py")  # Recomputes
"""

from __future__ import annotations

import copy
import functools
import json
import threading
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    cast,
)

# Type variables for generic query handling
T = TypeVar("T")
QueryKey = Tuple[Callable, Tuple[Any, ...]]


# Marker for salsa queries
_SALSA_QUERY_MARKER = "_is_salsa_query"


def salsa_query(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to mark a function as a Salsa query.

    Salsa queries:
    - Are automatically memoized when called through SalsaDB.query()
    - Track their dependencies on other queries
    - Can be invalidated, cascading to dependents

    Example:
        @salsa_query
        def get_functions(db: SalsaDB, path: str) -> List[str]:
            content = db.query(read_file, db, path)
            return extract_functions(content)
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # When called directly (not through db.query), just execute
        return func(*args, **kwargs)

    # Mark as salsa query
    setattr(wrapper, _SALSA_QUERY_MARKER, True)
    setattr(wrapper, "_original_func", func)

    return cast(Callable[..., T], wrapper)


def is_salsa_query(func: Callable[..., Any]) -> bool:
    """Check if a function is decorated with @salsa_query."""
    return getattr(func, _SALSA_QUERY_MARKER, False)


@dataclass
class CacheEntry:
    """Cache entry for a query result."""

    result: Any
    dependencies: Set[QueryKey] = field(default_factory=set)
    file_dependencies: Dict[str, int] = field(default_factory=dict)  # path -> revision


@dataclass
class QueryStats:
    """Statistics for query execution."""

    cache_hits: int = 0
    cache_misses: int = 0
    invalidations: int = 0
    recomputations: int = 0


class SalsaDB:
    """Database for Salsa-style query memoization.

    Tracks:
    - File contents and revisions
    - Query results and their dependencies
    - Reverse dependency graph for invalidation cascading

    Thread-safe for concurrent access with fine-grained locking.
    Lock is only held during cache access operations, NOT during query execution.
    This allows multiple queries to execute concurrently without blocking each other.
    """

    # HIGH-010 FIX: Maximum cache entries before LRU eviction kicks in.
    # Prevents unbounded memory growth in long-running processes.
    MAX_CACHE_SIZE: int = 10000

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread_local = threading.local()  # Thread-local state for query tracking

        # File storage (shared, protected by lock)
        self._file_contents: Dict[str, str] = {}
        self._file_revisions: Dict[str, int] = {}

        # Query cache: (func, args) -> CacheEntry (shared, protected by lock)
        self._query_cache: Dict[QueryKey, CacheEntry] = {}

        # Reverse dependencies: query_key -> set of dependent query_keys
        self._reverse_deps: Dict[QueryKey, Set[QueryKey]] = {}

        # File to query dependencies: file_path -> set of query_keys
        self._file_to_queries: Dict[str, Set[QueryKey]] = {}

        # HIGH-014 FIX: In-flight queries tracking to prevent duplicate execution.
        # Maps query_key -> Event that signals when computation completes.
        # Threads waiting for same query block on the Event instead of recomputing.
        self._in_flight: Dict[QueryKey, threading.Event] = {}

        # Stats (shared, protected by lock)
        self._stats = QueryStats()

    # -------------------------------------------------------------------------
    # Thread-local state accessors (no lock needed - per-thread)
    # -------------------------------------------------------------------------

    @property
    def _query_stack(self) -> List[QueryKey]:
        """Thread-local query stack for dependency tracking.

        Each thread has its own stack, so concurrent queries don't interfere.
        """
        if not hasattr(self._thread_local, "query_stack"):
            self._thread_local.query_stack = []
        return cast(List[QueryKey], self._thread_local.query_stack)

    @property
    def _pending_deps(self) -> Dict[QueryKey, Set[QueryKey]]:
        """Thread-local pending dependencies for queries being computed.

        Each thread tracks its own in-progress query dependencies.
        """
        if not hasattr(self._thread_local, "pending_deps"):
            self._thread_local.pending_deps = {}
        return cast(Dict[QueryKey, Set[QueryKey]], self._thread_local.pending_deps)

    @property
    def _file_reads(self) -> Dict[QueryKey, Dict[str, int]]:
        """Thread-local tracker for file reads during query execution.

        Maps query_key -> {path: revision} for files read during that query.
        This captures revisions at read time (not store time), preventing
        race conditions when files are modified during query execution.
        """
        if not hasattr(self._thread_local, "file_reads"):
            self._thread_local.file_reads = {}
        return cast(Dict[QueryKey, Dict[str, int]], self._thread_local.file_reads)

    # -------------------------------------------------------------------------
    # File Management
    # -------------------------------------------------------------------------

    def set_file(self, path: str, content: str) -> None:
        """Set or update file content.

        This increments the file's revision and invalidates any queries
        that depend on this file.

        Args:
            path: File path (used as key)
            content: File content
        """
        with self._lock:
            old_revision = self._file_revisions.get(path, 0)
            self._file_contents[path] = content
            self._file_revisions[path] = old_revision + 1

            # Invalidate queries that depend on this file
            self._invalidate_file_dependents(path)

    def get_file(self, path: str) -> Optional[str]:
        """Get file content.

        If called during a query, registers the file as a dependency.
        File revision is captured at read time in thread-local storage,
        ensuring correct tracking even if another thread modifies the file.

        Args:
            path: File path

        Returns:
            File content or None if not found
        """
        with self._lock:
            content = self._file_contents.get(path)
            revision = self._file_revisions.get(path, 0)

            # Track file dependency if in a query context
            if self._query_stack:
                current_query = self._query_stack[-1]

                # Track in thread-local file reads (captures revision at read time)
                if current_query not in self._file_reads:
                    self._file_reads[current_query] = {}
                self._file_reads[current_query][path] = revision

                # Track for invalidation (shared state)
                if path not in self._file_to_queries:
                    self._file_to_queries[path] = set()
                self._file_to_queries[path].add(current_query)

                # Also track in the cache entry if it exists (for re-reads)
                if current_query in self._query_cache:
                    entry = self._query_cache[current_query]
                    entry.file_dependencies[path] = revision

            return content

    def get_revision(self, path: str) -> int:
        """Get current revision number for a file.

        Args:
            path: File path

        Returns:
            Revision number (0 if file never set)
        """
        with self._lock:
            return self._file_revisions.get(path, 0)

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def query(self, func: Callable[..., T], *args: Any) -> T:
        """Execute a query with memoization and dependency tracking.

        Uses fine-grained locking: lock is only held during cache access,
        NOT during query execution. This allows concurrent query execution
        without blocking.

        If the query result is cached and valid, returns cached result.
        Otherwise, computes the result and caches it.

        Args:
            func: The query function (should be decorated with @salsa_query)
            *args: Arguments to pass to the function

        Returns:
            Query result

        Note:
            HIGH-014 FIX: If another thread is computing the same query,
            we wait for it to finish instead of recomputing.
        """
        key = self._make_key(func, args)

        # =====================================================================
        # Phase 1: Check cache and in-flight (with lock - fast operation)
        # HIGH-011 FIX: Take snapshots inside lock, validate outside
        # =====================================================================
        entry_snapshot: Optional[CacheEntry] = None
        file_revisions_snapshot: Dict[str, int] = {}
        wait_event: Optional[threading.Event] = None

        with self._lock:
            # HIGH-014 FIX: Check if another thread is computing this query
            if key in self._in_flight:
                wait_event = self._in_flight[key]

            if wait_event is None and key in self._query_cache:
                # HIGH-011 FIX: Snapshot entry data for validation outside lock
                entry = self._query_cache[key]
                entry_snapshot = entry
                # Snapshot current file revisions for validation
                for path in entry.file_dependencies:
                    file_revisions_snapshot[path] = self._file_revisions.get(path, 0)

        # HIGH-014 FIX: Wait for in-flight query if another thread is computing
        if wait_event is not None:
            wait_event.wait()
            # After waiting, the result should be in cache - retry
            with self._lock:
                if key in self._query_cache:
                    self._stats.cache_hits += 1
                    self._query_stack.append(key)
                    self._register_dependency_to_parent(key)
                    self._query_stack.pop()
                    return cast(T, copy.deepcopy(self._query_cache[key].result))
            # If not in cache after waiting (invalidated?), fall through to compute

        # HIGH-011 FIX: Validate outside lock using snapshots
        if entry_snapshot is not None:
            if self._is_entry_valid_unlocked(entry_snapshot, file_revisions_snapshot):
                with self._lock:
                    # Double-check entry still exists (could have been invalidated)
                    if key in self._query_cache:
                        self._stats.cache_hits += 1
                        self._query_stack.append(key)
                        self._register_dependency_to_parent(key)
                        self._query_stack.pop()
                        return cast(T, copy.deepcopy(self._query_cache[key].result))

        # =====================================================================
        # Phase 2: Register as in-flight and setup thread-local state
        # =====================================================================
        completion_event = threading.Event()
        with self._lock:
            # HIGH-014 FIX: Check again if another thread started computing
            if key in self._in_flight:
                wait_event = self._in_flight[key]
            else:
                self._in_flight[key] = completion_event
                self._stats.cache_misses += 1

        # If another thread beat us, wait for it
        if wait_event is not None:
            wait_event.wait()
            with self._lock:
                if key in self._query_cache:
                    self._stats.cache_hits += 1
                    self._query_stack.append(key)
                    self._register_dependency_to_parent(key)
                    self._query_stack.pop()
                    return cast(T, copy.deepcopy(self._query_cache[key].result))
            # Fall through to compute if still not in cache

        # Setup thread-local state (no lock needed - per-thread)
        self._pending_deps[key] = set()
        self._file_reads[key] = {}
        self._query_stack.append(key)

        try:
            # =================================================================
            # Phase 3: Execute query (WITHOUT lock - slow operation!)
            # =================================================================
            if is_salsa_query(func):
                original = getattr(func, "_original_func", func)
                result = original(*args)
            else:
                result = func(*args)

            # =================================================================
            # Phase 4: Store result (with lock - fast operation)
            # =================================================================
            with self._lock:
                self._stats.recomputations += 1

                file_deps = self._file_reads.get(key, {}).copy()

                entry = CacheEntry(
                    result=result,
                    dependencies=self._pending_deps.get(key, set()).copy(),
                    file_dependencies=file_deps,
                )

                self._query_cache[key] = entry
                self._register_dependency_to_parent(key)
                self._maybe_evict()

            return copy.deepcopy(result)

        finally:
            # =================================================================
            # Phase 5: Cleanup thread-local state and signal completion
            # =================================================================
            self._query_stack.pop()

            if key in self._pending_deps:
                del self._pending_deps[key]

            if key in self._file_reads:
                del self._file_reads[key]

            # HIGH-014 FIX: Signal waiting threads and remove from in-flight
            with self._lock:
                if key in self._in_flight and self._in_flight[key] is completion_event:
                    del self._in_flight[key]
            completion_event.set()

    def _register_dependency_to_parent(self, child_key: QueryKey) -> None:
        """Register a child query as a dependency of the current parent query.

        The query stack contains all active queries with the current query at [-1].
        The parent (caller) is at [-2]. We need at least 2 items on the stack
        for a parent-child relationship to exist.
        """
        if len(self._query_stack) < 2:
            return  # No parent exists (this is a top-level query)

        parent_key = self._query_stack[-2]  # Parent is SECOND from top

        # Guard against self-dependencies (defensive check)
        if parent_key == child_key:
            return

        # Track in pending deps (for queries still being computed)
        if parent_key in self._pending_deps:
            self._pending_deps[parent_key].add(child_key)

        # Track in cached entry (for queries already computed)
        if parent_key in self._query_cache:
            self._query_cache[parent_key].dependencies.add(child_key)

        # Track reverse dependency
        if child_key not in self._reverse_deps:
            self._reverse_deps[child_key] = set()
        self._reverse_deps[child_key].add(parent_key)

    def _make_key(self, func: Callable, args: Tuple[Any, ...]) -> QueryKey:
        """Create a cache key from function and arguments.

        Handles SalsaDB instances by using id() for hashing.
        """
        hashable_args = []
        for arg in args:
            if isinstance(arg, SalsaDB):
                hashable_args.append(("__salsa_db__", id(arg)))
            elif isinstance(arg, (list, dict, set)):
                hashable_args.append(self._to_hashable(arg))
            else:
                hashable_args.append(arg)

        return (func, tuple(hashable_args))

    def _to_hashable(self, obj: Any) -> Any:
        """Convert an object to a hashable form for cache key.

        Uses JSON serialization as a fast path for large structures,
        falling back to recursive tuple conversion for small ones.
        """
        if isinstance(obj, dict):
            if len(obj) > 10:
                try:
                    json_str = json.dumps(obj, sort_keys=True, default=str)
                    return (
                        "__dict__",
                        len(obj),
                        hash(json_str),
                        json_str[:100],
                    )
                except (TypeError, ValueError):
                    pass
            return tuple(sorted((k, self._to_hashable(v)) for k, v in obj.items()))

        if isinstance(obj, (list, tuple)):
            if len(obj) > 100:
                try:
                    json_str = json.dumps(obj, default=str)
                    return (
                        "__seq__",
                        len(obj),
                        hash(json_str),
                        json_str[:100],
                    )
                except (TypeError, ValueError):
                    pass
            return tuple(self._to_hashable(item) for item in obj)

        if isinstance(obj, set):
            return frozenset(self._to_hashable(item) for item in obj)

        try:
            hash(obj)
            return obj
        except TypeError:
            return str(obj)

    def _is_entry_valid(
        self, key: QueryKey, visited: Optional[Set[QueryKey]] = None
    ) -> bool:
        """Check if a cache entry is still valid (requires lock held).

        An entry is valid if:
        - All file dependencies have the same revision
        - All query dependencies are still valid

        Args:
            key: The query key to validate
            visited: Set of already-visited keys for cycle detection

        Returns:
            True if the entry is valid, False otherwise
        """
        if visited is None:
            visited = set()

        if key in visited:
            return True

        if key not in self._query_cache:
            return False

        visited.add(key)
        entry = self._query_cache[key]

        for path, revision in entry.file_dependencies.items():
            current_revision = self._file_revisions.get(path, 0)
            if current_revision != revision:
                return False

        for dep_key in entry.dependencies:
            if not self._is_entry_valid(dep_key, visited):
                return False

        return True

    def _is_entry_valid_unlocked(
        self,
        entry: CacheEntry,
        file_revisions: Dict[str, int],
    ) -> bool:
        """HIGH-011 FIX: Validate entry using pre-snapshotted data (no lock needed).

        This allows validation to happen outside the lock, reducing contention.
        Only checks file dependencies - query dependencies require lock access.

        Args:
            entry: The cache entry to validate
            file_revisions: Snapshot of current file revisions

        Returns:
            True if file dependencies are valid, False otherwise
        """
        for path, expected_revision in entry.file_dependencies.items():
            current_revision = file_revisions.get(path, 0)
            if current_revision != expected_revision:
                return False
        return True

    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------

    def get_dependencies(self, func: Callable[..., Any], *args: Any) -> Set[QueryKey]:
        """Get the dependencies of a query.

        Args:
            func: Query function
            *args: Query arguments

        Returns:
            Set of (func, args) tuples this query depends on
        """
        key = self._make_key(func, args)
        with self._lock:
            if key in self._query_cache:
                return self._query_cache[key].dependencies.copy()
            return set()

    # -------------------------------------------------------------------------
    # Invalidation
    # -------------------------------------------------------------------------

    def invalidate(self, func: Callable[..., Any], *args: Any) -> None:
        """Invalidate a specific query and its dependents.

        Args:
            func: Query function
            *args: Query arguments (if empty, invalidates all instances)
        """
        with self._lock:
            self._stats.invalidations += 1

            if args:
                key = self._make_key(func, args)
                self._invalidate_key(key)
            else:
                keys_to_invalidate = [
                    k for k in self._query_cache.keys() if k[0] == func
                ]
                for key in keys_to_invalidate:
                    self._invalidate_key(key)

    def _invalidate_key(self, key: QueryKey) -> None:
        """Invalidate a specific query key and cascade to dependents.

        HIGH-012 FIX: Cleans up _file_to_queries references.
        HIGH-013 FIX: Cleans up _reverse_deps references.
        """
        if key not in self._query_cache:
            return

        entry = self._query_cache[key]

        # HIGH-012 FIX: Clean up ALL file dependency references
        for path in entry.file_dependencies:
            if path in self._file_to_queries:
                self._file_to_queries[path].discard(key)
                if not self._file_to_queries[path]:
                    del self._file_to_queries[path]

        # HIGH-013 FIX: Clean up reverse dependency references
        # Remove this key from the reverse_deps sets of its children
        for child_key in entry.dependencies:
            if child_key in self._reverse_deps:
                self._reverse_deps[child_key].discard(key)
                if not self._reverse_deps[child_key]:
                    del self._reverse_deps[child_key]

        # Remove from cache
        del self._query_cache[key]

        # Cascade to dependents (queries that depend on this one)
        if key in self._reverse_deps:
            dependents = list(self._reverse_deps[key])
            del self._reverse_deps[key]
            for dep_key in dependents:
                self._invalidate_key(dep_key)

    def _invalidate_file_dependents(self, path: str) -> None:
        """Invalidate all queries that depend on a file."""
        if path not in self._file_to_queries:
            return

        queries = list(self._file_to_queries[path])
        del self._file_to_queries[path]

        for key in queries:
            self._invalidate_key(key)

    def _maybe_evict(self) -> None:
        """Evict oldest cache entries if cache exceeds MAX_CACHE_SIZE.

        HIGH-010 FIX: Prevents unbounded memory growth by removing oldest
        entries (FIFO eviction - dict preserves insertion order in Python 3.7+).

        Must be called with lock held.
        """
        while len(self._query_cache) > self.MAX_CACHE_SIZE:
            oldest_key = next(iter(self._query_cache))
            self._invalidate_key(oldest_key)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Get query execution statistics.

        Returns:
            Dict with keys: cache_hits, cache_misses, invalidations, recomputations
        """
        with self._lock:
            return {
                "cache_hits": self._stats.cache_hits,
                "cache_misses": self._stats.cache_misses,
                "invalidations": self._stats.invalidations,
                "recomputations": self._stats.recomputations,
            }

    def clear(self) -> None:
        """Clear all cached queries and file data."""
        with self._lock:
            self._query_cache.clear()
            self._reverse_deps.clear()
            self._file_to_queries.clear()
            self._in_flight.clear()
            self._stats = QueryStats()
