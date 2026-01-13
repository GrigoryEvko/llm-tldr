"""
TLDR Daemon core - the main TLDRDaemon server class.

Holds indexes in memory and handles commands via Unix/TCP socket.
"""

import atexit
import hashlib
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Optional

from tldr.dedup import ContentHashedIndex
from tldr.salsa import SalsaDB

# Lazy import for ml_engine to allow daemon to start even if ML deps missing
# (torch, sentence_transformers, etc. are heavy optional dependencies)
_ml_engine_available: bool | None = None


def _check_ml_engine() -> bool:
    """Check if ml_engine module is available (lazy import).

    Returns True if torch/sentence_transformers are installed,
    False otherwise. Result is cached for subsequent calls.
    """
    global _ml_engine_available
    if _ml_engine_available is None:
        try:
            from tldr import ml_engine  # noqa: F401

            _ml_engine_available = True
        except ImportError as e:
            logging.warning(f"ML engine not available: {e}. Semantic features disabled.")
            _ml_engine_available = False
    return _ml_engine_available


from tldr.stats import (
    HookStats,
    HookStatsStore,
    SessionStats,
    StatsStore,
    count_tokens,
    get_default_store,
)

from .cached_queries import (
    cached_architecture,
    cached_cfg,
    cached_context,
    cached_dead_code,
    cached_dfg,
    cached_extract,
    cached_importers,
    cached_imports,
    cached_search,
    cached_slice,
    cached_structure,
    cached_tree,
)

# Idle timeout: 30 minutes
IDLE_TIMEOUT = 30 * 60

# Maximum request size: 10MB - prevents OOM from malicious clients sending
# infinite data without newline terminator
MAX_REQUEST_SIZE = 10 * 1024 * 1024

# DA-7 FIX: Maximum concurrent connections - prevents resource exhaustion
# from connection floods or misbehaving clients. 100 is generous for a
# local daemon; typical usage is 1-5 concurrent connections.
MAX_CONNECTIONS = 100

logger = logging.getLogger(__name__)

# DA-1 FIX: Track all daemon instances for proper atexit cleanup.
# WeakSet automatically removes instances when they are garbage collected,
# preventing memory leaks while ensuring all live instances get cleaned up.
_daemon_instances: weakref.WeakSet["TLDRDaemon"] = weakref.WeakSet()
_cleanup_registered: bool = False


def _cleanup_all_daemons() -> None:
    """Clean up all daemon instances at program exit.

    Iterates through all live daemon instances and calls their cleanup methods.
    Each instance has idempotency guards (_stats_persisted, _model_unloaded)
    so double-cleanup is safe.

    This fixes DA-1: Previously only the first instance's handlers were registered,
    causing stats loss for subsequent daemon instances.
    """
    # Convert to list to avoid set modification during iteration
    # (though WeakSet handles this gracefully, explicit is better)
    for daemon in list(_daemon_instances):
        try:
            daemon._persist_all_stats()
        except Exception:
            # Don't raise in atexit - log and continue to next instance
            logger.debug(f"Failed to persist stats for {daemon.project}")
        try:
            daemon._unload_model()
        except Exception:
            # Don't raise in atexit - log and continue to next instance
            logger.debug(f"Failed to unload model for {daemon.project}")


class TLDRDaemon:
    """
    TLDR daemon server holding indexes in memory.

    Listens on a Unix socket for commands and responds with JSON.
    Automatically shuts down after IDLE_TIMEOUT seconds of inactivity.
    """

    def __init__(self, project_path: Path):
        """
        Initialize the daemon for a project.

        Args:
            project_path: Root path of the project to index
        """
        self.project = project_path
        self.tldr_dir = project_path / ".tldr"
        self.socket_path = self._compute_socket_path()
        self.last_query = time.time()
        self.indexes: dict[str, Any] = {}

        # Internal state
        self._status = "initializing"
        self._start_time = time.time()
        self._shutdown_requested = False
        self._socket: Optional[socket.socket] = None
        self._pidfile: Optional[Any] = None  # Locked PID file handle from startup.py

        # P5 Features: Content-hash deduplication and query memoization
        self.dedup_index: Optional[ContentHashedIndex] = None
        self.salsa_db: SalsaDB = SalsaDB()

        # P6 Features: Dirty-count triggered semantic re-indexing
        # Lock protects _dirty_count, _dirty_files, _reindex_in_progress
        # These are accessed from main thread (notify handler) and background reindex thread
        self._dirty_lock = threading.Lock()
        self._dirty_count: int = 0
        self._dirty_files: set[str] = set()
        self._reindex_in_progress: bool = False
        self._reindex_thread: Optional[threading.Thread] = None  # MED-025 FIX: Store for graceful shutdown
        self._semantic_config = self._load_semantic_config()

        # P7 Features: Per-session token stats tracking
        self._session_stats: dict[str, SessionStats] = {}
        self._session_stats_lock = threading.Lock()  # HIGH-016 FIX: Thread safety for session stats
        self._stats_store: StatsStore = get_default_store()

        # P8 Features: Per-hook activity stats tracking with persistence
        self._hook_stats_store: HookStatsStore = HookStatsStore(project_path)
        self._hook_stats: dict[str, HookStats] = self._hook_stats_store.load()
        self._hook_stats_baseline: dict[str, HookStats] = self._snapshot_hook_stats()
        self._hook_invocation_count: int = 0
        self._hook_flush_threshold: int = 5  # Flush every N invocations
        self._hook_stats_lock = threading.Lock()  # Thread safety for hook stats

        # P9 Features: Persistent ML model for semantic search
        # Model stays loaded in memory while daemon runs (no reload on each request)
        # Type annotations use strings since ml_engine is lazily imported
        self._model_manager: Optional["ModelManager"] = None  # type: ignore[name-defined]
        self._index_manager: Optional["IndexManager"] = None  # type: ignore[name-defined]
        self._model_loaded: bool = False
        self._model_load_lock = threading.Lock()

        # Cross-platform graceful shutdown: register atexit handlers
        # This ensures stats persist and GPU memory is freed even if daemon is killed
        # (works on all platforms). Note: SIGKILL (kill -9) still bypasses atexit.
        self._persist_lock = threading.Lock()  # Protects _stats_persisted check-and-set
        self._stats_persisted = False  # Guard against double-persist
        self._model_unloaded = False  # Guard against double-unload

        # DA-7 FIX: Connection limit enforcement via semaphore.
        # Prevents resource exhaustion from connection floods.
        # Semaphore allows up to MAX_CONNECTIONS concurrent handlers.
        self._connection_semaphore = threading.Semaphore(MAX_CONNECTIONS)
        self._active_connections = 0
        self._connection_count_lock = threading.Lock()

        # DA-1 FIX: Track this instance for cleanup at exit.
        # Uses module-level WeakSet + single atexit handler so ALL daemon instances
        # get cleaned up, not just the first one (the CRIT-008 fix's bug).
        # WeakSet auto-removes GC'd instances, preventing memory leaks.
        global _cleanup_registered
        _daemon_instances.add(self)
        if not _cleanup_registered:
            atexit.register(_cleanup_all_daemons)
            _cleanup_registered = True

    def _compute_socket_path(self) -> Path:
        """Compute deterministic socket path from project path."""
        hash_val = hashlib.sha256(str(self.project).encode()).hexdigest()[:16]
        return Path(f"/tmp/tldr-{hash_val}.sock")

    def _validate_path_in_project(self, file_path: str) -> Path:
        """Validate that a file path is within the project directory.

        Prevents path traversal attacks by resolving the path and checking
        it remains within the project boundary.

        Args:
            file_path: Path to validate (can be relative or absolute)

        Returns:
            Resolved Path object within the project

        Raises:
            ValueError: If the path escapes the project directory
        """
        # Handle relative vs absolute paths
        if Path(file_path).is_absolute():
            resolved = Path(file_path).resolve()
        else:
            resolved = (self.project / file_path).resolve()

        # Security check: ensure resolved path is within project
        project_resolved = self.project.resolve()
        try:
            resolved.relative_to(project_resolved)
        except ValueError:
            raise ValueError(
                f"Path traversal denied: {file_path} resolves to {resolved} "
                f"which is outside project {project_resolved}"
            ) from None

        return resolved


    def _load_semantic_config(self) -> dict:
        """Load semantic search configuration.

        Checks for config in:
        1. .claude/settings.json (Claude Code settings)
        2. .tldr/config.json (TLDR-specific settings)

        Returns default config if no file found.
        """
        default_config = {
            "enabled": True,
            "auto_reindex_threshold": 20,  # Files changed before auto re-index
            "model": "bge-large-en-v1.5",
        }

        # Try Claude settings first
        claude_settings = self.project / ".claude" / "settings.json"
        if claude_settings.exists():
            try:
                settings = json.loads(claude_settings.read_text())
                if "semantic_search" in settings:
                    return {**default_config, **settings["semantic_search"]}
            except Exception as e:
                logger.warning(f"Failed to load Claude settings: {e}")

        # Try TLDR config
        tldr_config = self.tldr_dir / "config.json"
        if tldr_config.exists():
            try:
                config = json.loads(tldr_config.read_text())
                if "semantic" in config:
                    return {**default_config, **config["semantic"]}
            except Exception as e:
                logger.warning(f"Failed to load TLDR config: {e}")

        return default_config

    def _ensure_model_loaded(self) -> "ModelManager":  # type: ignore[name-defined]
        """Lazily load and warm up the ML model.

        Thread-safe: Uses lock to prevent multiple threads from loading simultaneously.
        The model stays in memory until daemon shutdown.

        Returns:
            ModelManager singleton with loaded model.

        Raises:
            RuntimeError: If ML engine (torch, sentence_transformers) is not installed.
        """
        # Check ML engine availability before attempting to load
        if not _check_ml_engine():
            raise RuntimeError(
                "ML engine not available - install with: pip install llm-tldr[ml] "
                "(requires torch, sentence_transformers)"
            )

        if self._model_loaded and self._model_manager is not None:
            return self._model_manager

        with self._model_load_lock:
            # Double-check inside lock
            if self._model_loaded and self._model_manager is not None:
                return self._model_manager

            logger.info("Loading ML model for semantic search...")
            try:
                # Import lazily after availability check
                from tldr.ml_engine import get_model_manager, get_index_manager

                # Get singletons
                self._model_manager = get_model_manager()
                self._index_manager = get_index_manager()

                # Get model name from config
                model_name = self._semantic_config.get("model", "bge-large-en-v1.5")

                # Map short names to full HF paths if needed
                model_mapping = {
                    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
                    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
                }
                hf_model = model_mapping.get(model_name, model_name)

                # Load and warm up model
                self._model_manager.load(hf_model)
                warmup_stats = self._model_manager.warmup()
                logger.info(
                    f"ML model loaded: {hf_model} on {warmup_stats['device']} "
                    f"(backend: {warmup_stats['backend']}, warmup: {warmup_stats['warmup_time_s']:.2f}s)"
                )
                self._model_loaded = True

            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
                # Don't set _model_loaded = True, so we retry next time
                raise

            return self._model_manager

    def _unload_model(self) -> None:
        """Unload the ML model and free GPU/memory.

        Thread-safe and idempotent: Uses lock + _model_unloaded guard to prevent
        double-unload when both atexit and finally block trigger this method.
        Safe to call from atexit - exceptions are caught and logged.

        CRIT-007 FIX: Added lock protection. The original code had a race condition
        where two threads could both pass the guard check before either set the flag,
        causing double-unload which can crash GPU drivers or corrupt memory.
        """
        # CRIT-007 FIX: Wrap entire method in lock for atomic check-and-unload
        with self._model_load_lock:
            # Guard against double-unload (atexit + finally can both trigger)
            if self._model_unloaded or not self._model_loaded:
                return

            if self._model_manager is not None:
                try:
                    logger.info("Unloading ML model...")
                    self._model_manager.unload()
                    logger.info("ML model unloaded")
                except Exception as e:
                    # Log but don't raise - safe for atexit context
                    logger.error(f"Failed to unload ML model: {e}")

            if self._index_manager is not None:
                try:
                    self._index_manager.clear()
                    logger.info("Index cache cleared")
                except Exception as e:
                    # Log but don't raise - safe for atexit context
                    logger.error(f"Failed to clear index cache: {e}")

            self._model_loaded = False
            self._model_unloaded = True

    def _get_connection_info(self) -> tuple[str, int | None]:
        """Return (address, port) - port is None for Unix sockets.

        On Windows, uses TCP on localhost with a deterministic port.
        On Unix (Linux/macOS), uses Unix domain sockets.
        """
        if sys.platform == "win32":
            # TCP on localhost with deterministic port from hash
            hash_val = hashlib.sha256(str(self.project).encode()).hexdigest()[:16]
            port = 49152 + (int(hash_val, 16) % 10000)
            return ("127.0.0.1", port)
        else:
            # Unix socket path
            return (str(self.socket_path), None)

    def is_idle(self) -> bool:
        """Check if daemon has been idle longer than IDLE_TIMEOUT."""
        return (time.time() - self.last_query) > IDLE_TIMEOUT

    @property
    def call_graph(self) -> dict:
        """Get the call graph, loading if necessary."""
        self._ensure_call_graph_loaded()
        return self.indexes.get("call_graph", {"edges": [], "nodes": {}})

    def handle_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """
        Route and handle a command.

        Args:
            command: Dict with 'cmd' key and optional parameters

        Returns:
            Response dict with 'status' and command-specific fields
        """
        # Update last query time for any command
        self.last_query = time.time()

        cmd = command.get("cmd", "")

        handlers = {
            "ping": self._handle_ping,
            "status": self._handle_status,
            "shutdown": self._handle_shutdown,
            "search": self._handle_search,
            "extract": self._handle_extract,
            "impact": self._handle_impact,
            "dead": self._handle_dead,
            "arch": self._handle_arch,
            "cfg": self._handle_cfg,
            "dfg": self._handle_dfg,
            "slice": self._handle_slice,
            "calls": self._handle_calls,
            "warm": self._handle_warm,
            "semantic": self._handle_semantic,
            "tree": self._handle_tree,
            "structure": self._handle_structure,
            "context": self._handle_context,
            "imports": self._handle_imports,
            "importers": self._handle_importers,
            "notify": self._handle_notify,
            "diagnostics": self._handle_diagnostics,
            "change_impact": self._handle_change_impact,
            "track": self._handle_track,
        }

        handler = handlers.get(cmd)
        if handler:
            return handler(command)
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}

    def _handle_ping(self, command: dict) -> dict:
        """Handle ping command."""
        return {"status": "ok"}

    def _get_session_stats(self, session_id: str) -> SessionStats:
        """Get or create session stats for a session ID.

        Normalizes session_id to 8 chars to match status.py convention.
        This allows both full UUIDs and truncated IDs to work.

        Thread-safe: Uses _session_stats_lock to prevent concurrent access corruption.
        Memory safety: Limits dict to 10000 entries by evicting oldest when exceeded.
        Python 3.7+ dicts maintain insertion order, so first key is oldest.

        HIGH-016 FIX: Added lock to prevent race conditions on concurrent access.
        MED-026 FIX: Changed single eviction to loop eviction for batch additions.
        """
        # Normalize to 8 chars (matches status.py truncation)
        session_id = session_id[:8] if session_id else session_id

        with self._session_stats_lock:
            # Memory safety: evict oldest entries until under threshold
            # MED-026 FIX: Loop eviction handles batch additions that could
            # otherwise cause unbounded growth with single-entry eviction
            while len(self._session_stats) > 10000:
                oldest_key = next(iter(self._session_stats))
                del self._session_stats[oldest_key]
                logger.debug(f"Evicted oldest session stats entry: {oldest_key}")

            if session_id not in self._session_stats:
                self._session_stats[session_id] = SessionStats(session_id=session_id)
            return self._session_stats[session_id]

    def _get_hook_stats(self, hook_name: str) -> HookStats:
        """Get or create hook stats for a hook name.

        MUST be called while holding self._hook_stats_lock.

        HIGH-017 FIX: Added memory safety with loop eviction to prevent unbounded growth.
        Memory safety: Limits dict to 1000 entries (hooks are finite, but defensive).
        """
        # Memory safety: evict oldest entries until under threshold
        while len(self._hook_stats) > 1000:
            oldest_key = next(iter(self._hook_stats))
            del self._hook_stats[oldest_key]
            logger.debug(f"Evicted oldest hook stats entry: {oldest_key}")

        if hook_name not in self._hook_stats:
            self._hook_stats[hook_name] = HookStats(hook_name=hook_name)
        return self._hook_stats[hook_name]

    def _snapshot_hook_stats(self) -> dict[str, HookStats]:
        """Create a deep copy of current hook stats for delta tracking."""
        from copy import deepcopy
        return {name: deepcopy(stats) for name, stats in self._hook_stats.items()}

    def _handle_track(self, command: dict) -> dict:
        """Handle track command for hook activity reporting.

        Command format:
            {
                "cmd": "track",
                "hook": "hook-name",
                "success": true/false (default: true),
                "metrics": {"key": value, ...} (optional)
            }

        Flushes to disk every N invocations (default: 5) for durability
        while avoiding excessive I/O.
        """
        hook_name = command.get("hook")
        if not hook_name:
            return {"status": "error", "message": "Missing 'hook' field"}

        success = command.get("success", True)
        metrics = command.get("metrics", {})

        # Record the invocation with thread safety
        with self._hook_stats_lock:
            hook_stats = self._get_hook_stats(hook_name)
            hook_stats.record_invocation(success=success, metrics=metrics)

            # Increment global invocation counter and flush periodically
            self._hook_invocation_count += 1
            flushed = False
            if self._hook_invocation_count >= self._hook_flush_threshold:
                self._flush_hook_stats_unlocked()
                flushed = True

            total_invocations = hook_stats.invocations

        return {
            "status": "ok",
            "hook": hook_name,
            "total_invocations": total_invocations,
            "flushed": flushed,
        }

    def _flush_hook_stats_unlocked(self) -> None:
        """Flush hook stats delta to disk and reset counter.

        MUST be called while holding self._hook_stats_lock.
        """
        try:
            self._hook_stats_store.flush_delta(self._hook_stats, self._hook_stats_baseline)
            self._hook_stats_baseline = self._snapshot_hook_stats()
            self._hook_invocation_count = 0
            logger.debug("Flushed hook stats to disk")
        except Exception as e:
            logger.error(f"Failed to flush hook stats: {e}")

    def _flush_hook_stats(self) -> None:
        """Flush hook stats delta to disk and reset counter (thread-safe)."""
        with self._hook_stats_lock:
            self._flush_hook_stats_unlocked()

    def _handle_status(self, command: dict) -> dict:
        """Handle status command with P5 cache statistics.

        DA-3 FIX: All dict iterations are protected by their respective locks
        to prevent RuntimeError from concurrent modification.
        """
        uptime = time.time() - self._start_time

        # Get SalsaDB stats
        salsa_stats = self.salsa_db.get_stats()

        # Get dedup stats if loaded
        dedup_stats = {}
        if self.dedup_index:
            dedup_stats = self.dedup_index.stats()

        # DA-3 FIX: Gather all session stats under lock to prevent
        # RuntimeError "dictionary changed size during iteration"
        session_id = command.get("session")
        session_stats = None
        with self._session_stats_lock:
            if session_id:
                # Normalize to 8 chars (matches status.py convention)
                normalized_id = session_id[:8] if session_id else session_id
                stats = self._session_stats.get(normalized_id)
                if stats:
                    session_stats = stats.to_dict()

            # Get all sessions summary (must be inside lock for consistent snapshot)
            all_sessions_stats = {
                "active_sessions": len(self._session_stats),
                "total_raw_tokens": sum(s.raw_tokens for s in self._session_stats.values()),
                "total_tldr_tokens": sum(s.tldr_tokens for s in self._session_stats.values()),
                "total_requests": sum(s.requests for s in self._session_stats.values()),
                "session_ids": list(self._session_stats.keys()),  # Debug: show stored IDs
            }

        # DA-3 FIX: Get all hook stats under lock (P8)
        with self._hook_stats_lock:
            hook_stats_dict = {
                name: stats.to_dict() for name, stats in self._hook_stats.items()
            }

        # Get ML model stats (P9)
        ml_model_stats = {}
        if self._model_loaded and self._model_manager:
            try:
                ml_model_stats = self._model_manager.memory_stats()
                ml_model_stats["loaded"] = True
                if self._index_manager:
                    ml_model_stats["index_cache"] = self._index_manager.stats()
            except Exception as e:
                ml_model_stats = {"loaded": True, "error": str(e)}
        else:
            ml_model_stats = {"loaded": False}

        # DA-7 FIX: Include connection stats for monitoring
        with self._connection_count_lock:
            active_conns = self._active_connections

        return {
            "status": self._status,
            "uptime": uptime,
            "files": len(self.indexes.get("files", [])),
            "project": str(self.project),
            "salsa_stats": salsa_stats,
            "dedup_stats": dedup_stats,
            "session_stats": session_stats,
            "all_sessions": all_sessions_stats,
            "hook_stats": hook_stats_dict,
            "ml_model": ml_model_stats,
            "connections": {
                "active": active_conns,
                "max": MAX_CONNECTIONS,
            },
        }

    def _handle_shutdown(self, command: dict) -> dict:
        """Handle shutdown command with stats persistence and model cleanup."""
        # DA-4 FIX: Wait for background reindex thread before shutdown
        # This ensures index consistency - an incomplete reindex could leave
        # the semantic index in a corrupted state
        self._wait_for_reindex_thread()
        # Persist all session stats before shutdown
        self._persist_all_stats()
        # Unload ML model to free GPU/memory
        self._unload_model()
        self._shutdown_requested = True
        return {"status": "shutting_down"}

    def _wait_for_reindex_thread(self, timeout: float = 30.0) -> None:
        """Wait for background reindex thread to complete.

        DA-4 FIX: Ensures index consistency on shutdown. Without this wait,
        an in-progress reindex could leave the semantic index in an inconsistent
        state (partially written files, missing metadata, etc.).

        Args:
            timeout: Maximum seconds to wait for thread completion.
                    30s is generous - typical reindex takes 5-15s.
        """
        thread = self._reindex_thread
        if thread is None or not thread.is_alive():
            return

        logger.info(f"Waiting for background reindex thread to complete (timeout: {timeout}s)...")
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning(
                f"Reindex thread {thread.name} did not finish within {timeout}s. "
                "Index may be in an inconsistent state. Consider running 'tldr semantic index' manually."
            )
        else:
            logger.info("Background reindex thread completed successfully")

    def _persist_all_stats(self) -> None:
        """Persist all session and hook stats to JSONL stores.

        Thread-safe: uses a lock to prevent double-persist when both
        atexit and finally block trigger this method concurrently.

        DA-2 FIX: Copies session_stats items under lock before iterating,
        preventing RuntimeError from dict modification during iteration.
        """
        # Guard against double-persist (atexit + finally can both trigger)
        # Lock ensures atomic check-and-set to prevent race condition
        with self._persist_lock:
            if self._stats_persisted:
                return
            self._stats_persisted = True

        # DA-2 FIX: Copy items under lock to avoid holding lock during I/O.
        # This prevents RuntimeError "dictionary changed size during iteration"
        # when concurrent requests modify _session_stats while we iterate.
        with self._session_stats_lock:
            session_items = list(self._session_stats.items())

        # Persist session stats (now safe to iterate without lock)
        for session_id, stats in session_items:
            if stats.requests > 0:  # Only persist if there were actual requests
                try:
                    self._stats_store.append(stats)
                    logger.info(
                        f"Persisted stats for session {session_id}: "
                        f"{stats.requests} requests, {stats.savings_percent:.1f}% savings"
                    )
                except Exception as e:
                    logger.error(f"Failed to persist stats for session {session_id}: {e}")

        # Persist hook stats (final flush)
        if self._hook_invocation_count > 0:
            self._flush_hook_stats()
            logger.info(f"Persisted hook stats for {len(self._hook_stats)} hooks")

    def _handle_search(self, command: dict) -> dict:
        """Handle search command with SalsaDB caching."""
        pattern = command.get("pattern")
        if not pattern:
            return {"status": "error", "message": "Missing required parameter: pattern"}

        try:
            max_results = command.get("max_results", 100)
            # Use SalsaDB for cached search
            return self.salsa_db.query(
                cached_search,
                self.salsa_db,
                str(self.project),
                pattern,
                max_results,
            )
        except Exception as e:
            logger.exception("Search failed")
            return {"status": "error", "message": str(e)}

    def _handle_extract(self, command: dict) -> dict:
        """Handle extract command with SalsaDB caching and token tracking."""
        file_path = command.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing required parameter: file"}

        try:
            # Security: validate path is within project
            validated_path = self._validate_path_in_project(file_path)
            file_path = str(validated_path)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        try:
            # Track tokens if session ID provided
            session_id = command.get("session")
            raw_tokens = 0

            if session_id:
                # Count raw file tokens (what vanilla Claude would use)
                try:
                    raw_content = Path(file_path).read_text()
                    raw_tokens = count_tokens(raw_content)
                except Exception:
                    pass  # File might not exist or be binary

            # Use SalsaDB for cached extraction
            result = self.salsa_db.query(cached_extract, self.salsa_db, file_path)

            # Track token savings if session ID provided
            if session_id and raw_tokens > 0:
                tldr_tokens = count_tokens(json.dumps(result))
                stats = self._get_session_stats(session_id)
                stats.record_request(raw_tokens=raw_tokens, tldr_tokens=tldr_tokens)

                # Incremental persistence: save every 10 requests
                if stats.requests % 10 == 0:
                    try:
                        self._stats_store.append(stats)
                        logger.debug(f"Persisted stats for session {session_id}: {stats.requests} requests")
                    except Exception as e:
                        logger.warning(f"Failed to persist stats: {e}")

            return result
        except Exception as e:
            logger.exception("Extract failed")
            return {"status": "error", "message": str(e)}

    def _handle_impact(self, command: dict) -> dict:
        """Handle impact command - find callers of a function.

        Uses pre-built reverse index for O(1) lookup instead of O(n) edge scan.
        For 100K edges, this is ~5000x faster.
        """
        func_name = command.get("func")
        if not func_name:
            return {"status": "error", "message": "Missing required parameter: func"}

        try:
            self._ensure_call_graph_loaded()

            # O(1) lookup using reverse index instead of O(n) edge scan
            reverse_index = self.indexes.get("reverse_call_graph", {})
            callers = reverse_index.get(func_name, [])

            return {"status": "ok", "callers": callers}
        except Exception as e:
            logger.exception("Impact analysis failed")
            return {"status": "error", "message": str(e)}

    def _build_reverse_index(self, call_graph: dict) -> dict[str, list[dict]]:
        """Build reverse index from call graph for O(1) caller lookups.

        The call graph edges may use different key formats:
        - Legacy format: {"caller", "callee", "file", "line"}
        - New format: {"from_func", "to_func", "from_file", "to_file"}

        Args:
            call_graph: Call graph dict with "edges" list

        Returns:
            Reverse index mapping callee -> list of {caller, file, line}
        """
        reverse_index: dict[str, list[dict]] = {}
        for edge in call_graph.get("edges", []):
            # Handle both legacy and new edge formats
            caller = edge.get("caller") or edge.get("from_func")
            callee = edge.get("callee") or edge.get("to_func")
            caller_file = edge.get("file") or edge.get("from_file")
            line = edge.get("line")

            if callee:
                if callee not in reverse_index:
                    reverse_index[callee] = []
                reverse_index[callee].append(
                    {
                        "caller": caller,
                        "file": caller_file,
                        "line": line,
                    }
                )
        return reverse_index

    def _ensure_call_graph_loaded(self):
        """Load call graph and build reverse index for O(1) caller lookups."""
        if "call_graph" in self.indexes:
            return

        call_graph_path = self.tldr_dir / "call_graph.json"
        if call_graph_path.exists():
            try:
                call_graph = json.loads(call_graph_path.read_text())
                self.indexes["call_graph"] = call_graph
                self.indexes["reverse_call_graph"] = self._build_reverse_index(
                    call_graph
                )
                logger.info(
                    f"Loaded call graph from {call_graph_path}: "
                    f"{len(call_graph.get('edges', []))} edges, "
                    f"{len(self.indexes['reverse_call_graph'])} unique callees indexed"
                )
            except Exception as e:
                logger.error(f"Failed to load call graph: {e}")
                self.indexes["call_graph"] = {"edges": [], "nodes": {}}
                self.indexes["reverse_call_graph"] = {}
        else:
            logger.warning(f"No call graph found at {call_graph_path}")
            self.indexes["call_graph"] = {"edges": [], "nodes": {}}
            self.indexes["reverse_call_graph"] = {}

    def _handle_dead(self, command: dict) -> dict:
        """Handle dead code analysis command."""
        try:
            language = command.get("language", "python")
            entry_points = command.get("entry_points")
            # Convert to tuple for hashability (SalsaDB cache key)
            entry_tuple = tuple(entry_points) if entry_points else ()
            return self.salsa_db.query(
                cached_dead_code,
                self.salsa_db,
                str(self.project),
                entry_tuple,
                language,
            )
        except Exception as e:
            logger.exception("Dead code analysis failed")
            return {"status": "error", "message": str(e)}

    def _handle_arch(self, command: dict) -> dict:
        """Handle architecture analysis command."""
        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_architecture,
                self.salsa_db,
                str(self.project),
                language,
            )
        except Exception as e:
            logger.exception("Architecture analysis failed")
            return {"status": "error", "message": str(e)}

    def _handle_cfg(self, command: dict) -> dict:
        """Handle CFG extraction command."""
        file_path = command.get("file")
        function = command.get("function")
        if not file_path or not function:
            return {"status": "error", "message": "Missing required parameters: file, function"}

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_cfg,
                self.salsa_db,
                file_path,
                function,
                language,
            )
        except Exception as e:
            logger.exception("CFG extraction failed")
            return {"status": "error", "message": str(e)}

    def _handle_dfg(self, command: dict) -> dict:
        """Handle DFG extraction command."""
        file_path = command.get("file")
        function = command.get("function")
        if not file_path or not function:
            return {"status": "error", "message": "Missing required parameters: file, function"}

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_dfg,
                self.salsa_db,
                file_path,
                function,
                language,
            )
        except Exception as e:
            logger.exception("DFG extraction failed")
            return {"status": "error", "message": str(e)}

    def _handle_slice(self, command: dict) -> dict:
        """Handle program slice command."""
        file_path = command.get("file")
        function = command.get("function")
        line = command.get("line")
        if not file_path or not function or line is None:
            return {"status": "error", "message": "Missing required parameters: file, function, line"}

        try:
            direction = command.get("direction", "backward")
            variable = command.get("variable", "")
            return self.salsa_db.query(
                cached_slice,
                self.salsa_db,
                file_path,
                function,
                int(line),
                direction,
                variable,
            )
        except Exception as e:
            logger.exception("Program slice failed")
            return {"status": "error", "message": str(e)}

    def _handle_calls(self, command: dict) -> dict:
        """Handle call graph building command."""
        try:
            language = command.get("language", "python")
            from tldr.cross_file_calls import build_project_call_graph
            graph = build_project_call_graph(self.project, language=language)
            result = {
                "edges": [
                    {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                    for e in graph.edges
                ],
                "count": len(graph.edges),
            }
            return {"status": "ok", "result": result}
        except Exception as e:
            logger.exception("Call graph building failed")
            return {"status": "error", "message": str(e)}

    def _handle_warm(self, command: dict) -> dict:
        """Handle cache warming command (builds call graph cache)."""
        try:
            language = command.get("language", "python")
            from tldr.cross_file_calls import scan_project, build_project_call_graph

            files = scan_project(self.project, language=language)
            graph = build_project_call_graph(self.project, language=language)

            # Create cache directory and save
            cache_dir = self.tldr_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "call_graph.json"
            cache_data = {
                "edges": [
                    {"from_file": e[0], "from_func": e[1], "to_file": e[2], "to_func": e[3]}
                    for e in graph.edges
                ],
                "languages": [language],
                "timestamp": time.time(),
            }
            cache_file.write_text(json.dumps(cache_data, indent=2))

            # Update in-memory index and rebuild reverse index for O(1) lookups
            self.indexes["call_graph"] = cache_data
            self.indexes["reverse_call_graph"] = self._build_reverse_index(cache_data)

            return {"status": "ok", "files": len(files), "edges": len(graph.edges)}
        except Exception as e:
            logger.exception("Cache warming failed")
            return {"status": "error", "message": str(e)}

    def _handle_semantic(self, command: dict) -> dict:
        """Handle semantic search/index command using persistent ML model.

        The model is loaded once on first use and stays in GPU/memory
        until daemon shutdown. This avoids 5-10s model load on each request.

        Gracefully handles missing ML dependencies (torch, sentence_transformers)
        by returning helpful error messages with installation instructions.
        """
        action = command.get("action", "search")

        # For status action, report availability even if ML deps missing
        if action == "status":
            ml_available = _check_ml_engine()
            if not ml_available:
                return {
                    "status": "ok",
                    "model_loaded": False,
                    "ml_available": False,
                    "install_hint": "pip install llm-tldr[ml]",
                }
            if self._model_loaded and self._model_manager:
                stats = self._model_manager.memory_stats()
                idx_stats = self._index_manager.stats() if self._index_manager else {}
                return {
                    "status": "ok",
                    "model_loaded": True,
                    "ml_available": True,
                    "model_stats": stats,
                    "index_stats": idx_stats,
                }
            else:
                return {
                    "status": "ok",
                    "model_loaded": False,
                    "ml_available": True,
                }

        # All other actions require ML engine - check availability early
        if not _check_ml_engine():
            return {
                "status": "error",
                "message": (
                    "Semantic features require ML dependencies. "
                    "Install with: pip install llm-tldr[ml]"
                ),
                "ml_available": False,
            }

        try:
            if action == "index":
                # For indexing, use ml_engine.build_index which uses ModelManager
                from tldr.ml_engine import build_index

                language = command.get("language", "python")

                # Ensure model is loaded first (warm up)
                self._ensure_model_loaded()

                # Build index using persistent model
                count = build_index(str(self.project), lang=language)
                return {"status": "ok", "indexed": count}

            elif action == "search":
                # For search, use ml_engine.search which uses ModelManager + IndexManager
                from tldr.ml_engine import search

                query = command.get("query")
                if not query:
                    return {"status": "error", "message": "Missing required parameter: query"}

                # Ensure model is loaded
                self._ensure_model_loaded()

                k = command.get("k", 10)
                expand_graph = command.get("expand_graph", False)
                results = search(str(self.project), query, k=k, expand_graph=expand_graph)
                return {"status": "ok", "results": results}

            elif action == "warmup":
                # Explicitly warm up the model (useful for pre-loading on daemon start)
                mm = self._ensure_model_loaded()
                stats = mm.memory_stats()
                return {
                    "status": "ok",
                    "model": stats.get("model"),
                    "device": stats.get("device"),
                    "backend": mm.backend,
                }

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception("Semantic operation failed")
            return {"status": "error", "message": str(e)}

    def _handle_tree(self, command: dict) -> dict:
        """Handle file tree command."""
        try:
            extensions = command.get("extensions")
            ext_tuple = tuple(extensions) if extensions else ()
            exclude_hidden = command.get("exclude_hidden", True)
            return self.salsa_db.query(
                cached_tree,
                self.salsa_db,
                str(self.project),
                ext_tuple,
                exclude_hidden,
            )
        except Exception as e:
            logger.exception("File tree failed")
            return {"status": "error", "message": str(e)}

    def _handle_structure(self, command: dict) -> dict:
        """Handle code structure command."""
        try:
            language = command.get("language", "python")
            max_results = command.get("max_results", 100)
            return self.salsa_db.query(
                cached_structure,
                self.salsa_db,
                str(self.project),
                language,
                max_results,
            )
        except Exception as e:
            logger.exception("Code structure failed")
            return {"status": "error", "message": str(e)}

    def _handle_context(self, command: dict) -> dict:
        """Handle relevant context command."""
        entry = command.get("entry")
        if not entry:
            return {"status": "error", "message": "Missing required parameter: entry"}

        try:
            language = command.get("language", "python")
            depth = command.get("depth", 2)
            return self.salsa_db.query(
                cached_context,
                self.salsa_db,
                str(self.project),
                entry,
                language,
                depth,
            )
        except Exception as e:
            logger.exception("Relevant context failed")
            return {"status": "error", "message": str(e)}

    def _handle_imports(self, command: dict) -> dict:
        """Handle imports extraction command."""
        file_path = command.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing required parameter: file"}

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_imports,
                self.salsa_db,
                file_path,
                language,
            )
        except Exception as e:
            logger.exception("Imports extraction failed")
            return {"status": "error", "message": str(e)}

    def _handle_importers(self, command: dict) -> dict:
        """Handle reverse import lookup command."""
        module = command.get("module")
        if not module:
            return {"status": "error", "message": "Missing required parameter: module"}

        try:
            language = command.get("language", "python")
            return self.salsa_db.query(
                cached_importers,
                self.salsa_db,
                str(self.project),
                module,
                language,
            )
        except Exception as e:
            logger.exception("Importers lookup failed")
            return {"status": "error", "message": str(e)}

    def _ensure_dedup_index_loaded(self):
        """Load or create ContentHashedIndex for file deduplication."""
        if self.dedup_index is not None:
            return

        self.dedup_index = ContentHashedIndex(str(self.project))

        # Try to load persisted index
        if self.dedup_index.load():
            logger.info("Loaded content-hash index from disk")
        else:
            logger.info("Created new content-hash index")

        # Index all Python files in project
        for py_file in self.project.rglob("*.py"):
            if ".venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            try:
                self.dedup_index.get_or_create_edges(str(py_file), lang="python")
            except Exception as e:
                logger.debug(f"Could not index {py_file}: {e}")

    def _save_dedup_index(self):
        """Persist ContentHashedIndex to disk."""
        if self.dedup_index:
            try:
                self.dedup_index.save()
                logger.info("Saved content-hash index to disk")
            except Exception as e:
                logger.error(f"Failed to save dedup index: {e}")

    def _handle_notify(self, command: dict) -> dict:
        """Handle file change notification from hooks.

        Tracks dirty files and triggers background semantic re-indexing
        when threshold is reached.

        Args:
            command: Dict with 'file' (path to changed file)

        Returns:
            Response with dirty count and reindex status
        """
        file_path = command.get("file")
        if not file_path:
            return {"status": "error", "message": "Missing required parameter: file"}

        # Check if semantic search is enabled
        if not self._semantic_config.get("enabled", True):
            # Still notify for Salsa cache invalidation
            self.notify_file_changed(file_path)
            return {"status": "ok", "semantic_enabled": False}

        # Track dirty file and check reindex threshold under lock
        # Lock protects _dirty_files, _dirty_count, _reindex_in_progress from
        # concurrent access by main thread and background reindex thread
        threshold = self._semantic_config.get("auto_reindex_threshold", 20)
        with self._dirty_lock:
            if file_path not in self._dirty_files:
                self._dirty_files.add(file_path)
                self._dirty_count += 1
                logger.info(
                    f"Dirty file tracked: {file_path} (count: {self._dirty_count})"
                )

            # Check if we should trigger background re-indexing
            should_reindex = (
                self._dirty_count >= threshold and not self._reindex_in_progress
            )
            # Capture current count for response while holding lock
            current_dirty_count = self._dirty_count

        # Notify Salsa for cache invalidation (outside lock - no shared state)
        self.notify_file_changed(file_path)

        if should_reindex:
            self._trigger_background_reindex()

        return {
            "status": "ok",
            "dirty_count": current_dirty_count,
            "threshold": threshold,
            "reindex_triggered": should_reindex,
        }

    def _trigger_background_reindex(self):
        """Trigger background semantic re-indexing.

        Spawns a subprocess to rebuild the semantic index,
        allowing the daemon to continue serving requests.

        Thread-safe: Uses _dirty_lock to prevent TOCTOU race on _reindex_in_progress.
        """
        import subprocess

        # Atomic check-and-set under lock prevents TOCTOU race where two threads
        # both pass the check before either sets the flag
        with self._dirty_lock:
            if self._reindex_in_progress:
                logger.info("Re-index already in progress, skipping")
                return
            self._reindex_in_progress = True
            dirty_files = list(self._dirty_files)

        logger.info(
            f"Triggering background semantic re-index for {len(dirty_files)} files"
        )

        def do_reindex():
            try:
                # Run semantic index command
                cmd = [
                    sys.executable,
                    "-m",
                    "tldr.cli",
                    "semantic",
                    "index",
                    str(self.project),
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 min max
                )

                if result.returncode == 0:
                    logger.info("Background semantic re-index completed successfully")
                else:
                    logger.error(
                        f"Background semantic re-index failed: {result.stderr}"
                    )

            except Exception as e:
                logger.exception(f"Background semantic re-index error: {e}")
            finally:
                # Reset dirty tracking under lock
                with self._dirty_lock:
                    self._dirty_files.clear()
                    self._dirty_count = 0
                    self._reindex_in_progress = False

        # Run in thread to not block daemon
        # MED-025 FIX: Store thread reference for graceful shutdown tracking
        self._reindex_thread = threading.Thread(target=do_reindex, daemon=True)
        self._reindex_thread.start()

    def _handle_diagnostics(self, command: dict) -> dict:
        """Handle diagnostics command - type check + lint.

        Runs pyright for type checking and ruff for linting.
        Returns structured errors for pre-test validation.

        Args:
            command: Dict with optional:
                - file: Single file to check
                - project: If True, check whole project
                - no_lint: If True, skip ruff (type check only)

        Returns:
            Response with errors list and summary
        """
        import subprocess

        file_path = command.get("file")
        check_project = command.get("project", False)
        no_lint = command.get("no_lint", False)

        target = str(self.project) if check_project else file_path
        if not target:
            return {"status": "error", "message": "Missing required parameter: file or project"}

        errors = []

        # Run pyright for type checking
        try:
            pyright_cmd = ["pyright", "--outputjson", target]
            result = subprocess.run(
                pyright_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.project),
            )
            if result.stdout:
                try:
                    pyright_output = json.loads(result.stdout)
                    for diag in pyright_output.get("generalDiagnostics", []):
                        errors.append({
                            "type": "type",
                            "severity": diag.get("severity", "error"),
                            "file": diag.get("file", ""),
                            "line": diag.get("range", {}).get("start", {}).get("line", 0),
                            "message": diag.get("message", ""),
                            "rule": diag.get("rule", "pyright"),
                        })
                except json.JSONDecodeError:
                    pass
        except FileNotFoundError:
            logger.debug("pyright not found, skipping type check")
        except subprocess.TimeoutExpired:
            logger.warning("pyright timed out")
        except Exception as e:
            logger.debug(f"pyright error: {e}")

        # Run ruff for linting (unless disabled)
        if not no_lint:
            try:
                ruff_cmd = ["ruff", "check", "--output-format=json", target]
                result = subprocess.run(
                    ruff_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(self.project),
                )
                if result.stdout:
                    try:
                        ruff_output = json.loads(result.stdout)
                        for diag in ruff_output:
                            errors.append({
                                "type": "lint",
                                "severity": "warning" if diag.get("fix") else "error",
                                "file": diag.get("filename", ""),
                                "line": diag.get("location", {}).get("row", 0),
                                "message": diag.get("message", ""),
                                "rule": diag.get("code", "ruff"),
                            })
                    except json.JSONDecodeError:
                        pass
            except FileNotFoundError:
                logger.debug("ruff not found, skipping lint")
            except subprocess.TimeoutExpired:
                logger.warning("ruff timed out")
            except Exception as e:
                logger.debug(f"ruff error: {e}")

        type_errors = len([e for e in errors if e["type"] == "type"])
        lint_errors = len([e for e in errors if e["type"] == "lint"])

        return {
            "status": "ok",
            "errors": errors,
            "summary": {
                "total": len(errors),
                "type_errors": type_errors,
                "lint_errors": lint_errors,
            },
        }

    def _handle_change_impact(self, command: dict) -> dict:
        """Handle change-impact command - find affected tests.

        Uses call graph to find what tests are affected by changed files.
        Two-method discovery:
        1. Call graph traversal: tests that call changed functions
        2. Import analysis: tests that import changed modules

        Args:
            command: Dict with optional:
                - files: List of changed file paths
                - session: If True, use session's dirty files
                - git: If True, use git diff to find changed files

        Returns:
            Response with affected tests list
        """
        import subprocess

        files = command.get("files", [])
        use_session = command.get("session", False)
        use_git = command.get("git", False)

        # Get changed files from various sources
        # Protect read of _dirty_files from concurrent modification by reindex thread
        if use_session:
            with self._dirty_lock:
                if self._dirty_files:
                    files = list(self._dirty_files)
        if not files and use_git:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(self.project),
                )
                if result.returncode == 0:
                    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            except Exception as e:
                logger.debug(f"git diff failed: {e}")

        if not files:
            return {"status": "ok", "affected_tests": [], "message": "No changed files"}

        affected_tests = set()
        changed_functions = set()

        # Extract functions from changed files
        for file_path in files:
            if not file_path.endswith(".py"):
                continue
            # Security: validate path is within project
            try:
                full_path = self._validate_path_in_project(file_path)
            except ValueError:
                continue  # Skip paths outside project
            if not full_path.exists():
                continue

            try:
                from tldr.ast_extractor import extract_file
                info = extract_file(str(full_path))
                for func in info.functions:
                    changed_functions.add(func.name)
            except Exception as e:
                logger.debug(f"Could not extract {file_path}: {e}")

        # Method 1: Call graph traversal - find tests that call changed functions
        if changed_functions and self.call_graph:
            for func_name in changed_functions:
                # Find callers of this function
                for edge in self.call_graph.get("edges", []):
                    if edge.get("to_func") == func_name:
                        caller_file = edge.get("from_file", "")
                        if "test" in caller_file.lower():
                            affected_tests.add(caller_file)

        # Method 2: Import analysis - find test files that import changed modules
        for file_path in files:
            if not file_path.endswith(".py"):
                continue
            module_name = Path(file_path).stem

            # Search for imports of this module in test files
            try:
                from tldr.cross_file_calls import scan_project
                test_files = [f for f in scan_project(self.project) if "test" in f.lower()]

                for test_file in test_files:
                    try:
                        with open(self.project / test_file) as f:
                            content = f.read()
                            if f"import {module_name}" in content or f"from {module_name}" in content:
                                affected_tests.add(test_file)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Import analysis failed: {e}")

        return {
            "status": "ok",
            "affected_tests": sorted(list(affected_tests)),
            "changed_files": files,
            "changed_functions": sorted(list(changed_functions)),
            "summary": {
                "files_changed": len(files),
                "functions_changed": len(changed_functions),
                "tests_affected": len(affected_tests),
            },
        }

    def notify_file_changed(self, file_path: str):
        """Notify daemon that a file has changed.

        This invalidates cached queries that depend on this file.

        Args:
            file_path: Absolute path to the changed file
        """
        logger.debug(f"File change notification: {file_path}")

        # Invalidate SalsaDB cache entries for this file
        self.salsa_db.set_file(file_path, "changed")  # Triggers invalidation

        # Update dedup index if loaded
        if self.dedup_index:
            # Re-extract edges for the changed file
            try:
                # Detect language from extension
                lang = "python"
                if file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
                    lang = "typescript"
                elif file_path.endswith(".go"):
                    lang = "go"
                elif file_path.endswith(".rs"):
                    lang = "rust"

                self.dedup_index.get_or_create_edges(file_path, lang=lang)
            except Exception as e:
                logger.debug(f"Could not re-index {file_path}: {e}")

    def _get_tmp_pid_path(self) -> Path:
        """Get PID file path in /tmp (matches socket path pattern)."""
        hash_val = hashlib.sha256(str(self.project).encode()).hexdigest()[:16]
        return Path(f"/tmp/tldr-{hash_val}.pid")

    def write_pid_file(self):
        """Write daemon PID to .tldr/daemon.pid (and /tmp if not already done).

        If _pidfile is set, startup.py already wrote and locked /tmp/tldr-{hash}.pid.
        We only write to .tldr/daemon.pid for backwards compatibility.
        """
        pid = str(os.getpid())

        # Write to .tldr/daemon.pid (backwards compat)
        self.tldr_dir.mkdir(parents=True, exist_ok=True)
        pid_file = self.tldr_dir / "daemon.pid"
        pid_file.write_text(pid)

        # Only write to /tmp if startup.py didn't already (legacy path)
        if self._pidfile is None:
            tmp_pid_file = self._get_tmp_pid_path()
            tmp_pid_file.write_text(pid)
            logger.info(f"Wrote PID {pid} to {pid_file} and {tmp_pid_file}")
        else:
            logger.info(f"Wrote PID {pid} to {pid_file} (lock held by startup)")

    def remove_pid_file(self):
        """Remove PID files and release lock."""
        # Remove .tldr/daemon.pid
        pid_file = self.tldr_dir / "daemon.pid"
        if pid_file.exists():
            try:
                pid_file.unlink()
            except OSError:
                pass

        # Close and remove /tmp/tldr-{hash}.pid
        # If _pidfile is set, closing it releases the flock
        if self._pidfile is not None:
            try:
                self._pidfile.close()  # This releases the flock
            except Exception:
                pass
            self._pidfile = None
            logger.info("Released PID file lock")

        # Also try to remove the /tmp file (in case it exists)
        tmp_pid_file = self._get_tmp_pid_path()
        if tmp_pid_file.exists():
            try:
                tmp_pid_file.unlink()
            except OSError:
                pass

        logger.info("Removed PID files")

    def write_status(self, status: str):
        """Write status to .tldr/status file."""
        self.tldr_dir.mkdir(parents=True, exist_ok=True)
        status_file = self.tldr_dir / "status"
        status_file.write_text(status)
        self._status = status
        logger.info(f"Status: {status}")

    def read_status(self) -> str:
        """Read status from .tldr/status file."""
        status_file = self.tldr_dir / "status"
        if status_file.exists():
            return status_file.read_text().strip()
        return "unknown"

    def _create_server_socket(self) -> socket.socket:
        """Create appropriate socket for platform.

        On Windows, creates a TCP socket bound to localhost.
        On Unix, creates a Unix domain socket.

        Returns:
            Configured and bound socket ready for listening.
        """
        import errno

        if sys.platform == "win32":
            # TCP on localhost for Windows
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            addr, port = self._get_connection_info()
            sock.bind((addr, port))
            sock.listen(5)
            sock.settimeout(1.0)
            logger.info(f"Listening on {addr}:{port}")
        else:
            # Unix socket for Linux/macOS
            # Try to bind without deleting existing socket - if bind fails,
            # another daemon is running. This prevents race conditions.
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # Don't use SO_REUSEADDR for Unix sockets - it allows multiple binds
            try:
                sock.bind(str(self.socket_path))
            except OSError as e:
                # Socket exists and is in use - clean up and retry once
                # EADDRINUSE is 48 on macOS, 98 on Linux
                if e.errno == errno.EADDRINUSE or "Address already in use" in str(e):
                    # Check if existing daemon is responsive
                    if self.socket_path.exists():
                        try:
                            test_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            test_sock.connect(str(self.socket_path))
                            test_sock.close()
                            # Another daemon is running - exit
                            sock.close()
                            raise RuntimeError("Another daemon is already running")
                        except ConnectionRefusedError:
                            # Stale socket - remove and retry
                            self.socket_path.unlink()
                            sock.bind(str(self.socket_path))
                        except FileNotFoundError:
                            # Socket was removed between check and connect
                            sock.bind(str(self.socket_path))
                else:
                    raise
            sock.listen(5)
            sock.settimeout(1.0)
            logger.info(f"Listening on {self.socket_path}")

        return sock

    def _cleanup_socket(self):
        """Clean up the socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self.socket_path.exists():
            self.socket_path.unlink()
        logger.info("Socket cleaned up")

    def _handle_one_connection(self):
        """Handle a single client connection with connection limit enforcement.

        DA-7 FIX: Uses semaphore to enforce MAX_CONNECTIONS limit.
        If limit is reached, new connections are immediately rejected with
        a 503 Service Unavailable-style error instead of hanging.
        """
        if not self._socket:
            return

        try:
            conn, _ = self._socket.accept()
        except socket.timeout:
            return
        except OSError:
            return

        # DA-7 FIX: Try to acquire semaphore with timeout
        # Non-blocking acquire to immediately reject when at limit
        acquired = self._connection_semaphore.acquire(blocking=False)
        if not acquired:
            # At connection limit - reject immediately with helpful error
            try:
                response = {
                    "status": "error",
                    "message": f"Server busy: maximum {MAX_CONNECTIONS} concurrent connections reached",
                    "code": "CONNECTION_LIMIT",
                }
                conn.settimeout(1.0)
                conn.sendall(json.dumps(response).encode() + b"\n")
            except Exception:
                pass
            finally:
                conn.close()
            logger.warning(f"Connection rejected: at {MAX_CONNECTIONS} connection limit")
            return

        # Track active connections for status reporting
        with self._connection_count_lock:
            self._active_connections += 1

        try:
            conn.settimeout(5.0)
            data = b""
            size_exceeded = False
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                # Security: Prevent OOM from malicious clients sending infinite data
                if len(data) > MAX_REQUEST_SIZE:
                    size_exceeded = True
                    logger.warning(
                        f"Request size exceeded {MAX_REQUEST_SIZE} bytes, rejecting connection"
                    )
                    break
                if b"\n" in data:
                    break

            if size_exceeded:
                response = {
                    "status": "error",
                    "message": f"Request too large: exceeds {MAX_REQUEST_SIZE} byte limit",
                }
                conn.sendall(json.dumps(response).encode() + b"\n")
            elif data:
                try:
                    # Use errors='replace' to handle malformed UTF-8 gracefully
                    command = json.loads(data.decode('utf-8', errors='replace').strip())
                    response = self.handle_command(command)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    response = {"status": "error", "message": f"Invalid request: {e}"}

                conn.sendall(json.dumps(response).encode() + b"\n")
        except BrokenPipeError:
            # Client disconnected before receiving response - normal occurrence
            logger.debug("Client disconnected before receiving response")
        except Exception:
            logger.exception("Error handling connection")
        finally:
            conn.close()
            # DA-7 FIX: Release semaphore and update counter
            with self._connection_count_lock:
                self._active_connections -= 1
            self._connection_semaphore.release()

    def run(self):
        """Run the daemon main loop."""
        self.write_pid_file()
        self.write_status("indexing")

        # Cross-platform signal handling for graceful shutdown
        # Signal handlers just set the flag - actual cleanup happens in finally block
        #
        # CRIT-009 FIX: Removed logger.info() from signal handler because logging
        # is NOT async-signal-safe. If signal arrives while logging system holds a
        # lock, calling logger.info() deadlocks waiting for the same lock.
        # Only set the flag here; log the signal AFTER exiting the handler context.
        _pending_signal: list[int] = []  # Mutable container to capture signal number

        def _signal_handler(signum: int, frame: Any) -> None:
            # CRIT-009: NO LOGGING HERE - not async-signal-safe!
            # Only safe operations: setting flags, writing to pipes, sig_atomic ops
            _pending_signal.append(signum)
            self._shutdown_requested = True

        # SIGINT works on all platforms (Ctrl+C)
        signal.signal(signal.SIGINT, _signal_handler)

        # SIGTERM only on Unix/Mac (Windows ignores it but doesn't raise)
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, _signal_handler)

        try:
            self._socket = self._create_server_socket()
            self.write_status("ready")

            logger.info(f"TLDR daemon started for {self.project}")

            while not self._shutdown_requested:
                self._handle_one_connection()

                # Check for idle timeout
                if self.is_idle():
                    logger.info("Idle timeout reached, shutting down")
                    break

            # CRIT-009 FIX: Log signal info AFTER exiting handler context (now safe)
            if _pending_signal:
                signum = _pending_signal[0]
                signame = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
                logger.info(f"Received {signame}, initiating graceful shutdown")

        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down")
        except Exception:
            logger.exception("Daemon error")
        finally:
            # DA-4 FIX: Wait for background reindex thread before cleanup
            # This ensures index consistency - must complete before we unload ML model
            try:
                self._wait_for_reindex_thread()
            except Exception as e:
                logger.error(f"Failed to wait for reindex thread: {e}")

            # Persist stats before cleanup (graceful shutdown)
            try:
                self._persist_all_stats()
                logger.info("Stats persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist stats on shutdown: {e}")

            # Save dedup index to avoid losing content-hash cache
            try:
                self._save_dedup_index()
            except Exception as e:
                logger.error(f"Failed to save dedup index on shutdown: {e}")

            # Unload ML model to free GPU/memory (P9)
            try:
                self._unload_model()
            except Exception as e:
                logger.error(f"Failed to unload ML model on shutdown: {e}")

            self._cleanup_socket()
            self.remove_pid_file()
            self.write_status("stopped")
            logger.info("Daemon stopped")
