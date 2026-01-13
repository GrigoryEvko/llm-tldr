"""
Production ML inference engine with Qwen3-Embedding + TEI.

Features:
- Qwen3-Embedding-0.6B: 32K context, instruction-aware, MRL support
- TEI backend: High-throughput Rust server (text-embeddings-inference)
- Fallback to SentenceTransformers if TEI unavailable
- bf16 on GPU, fp32/bf16 on CPU
- torch.compile() for non-TEI path
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np
    import torch


# ============================================================================
# Model Configuration
# ============================================================================

# Default model - Qwen3-Embedding-0.6B
# - 0.6B params, 1024 dims, 32K context
# - Instruction-aware, MRL support
# - Apache 2.0 license
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Model registry with characteristics
SUPPORTED_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "dimension": 1024,
        "max_seq_len": 32768,
        "size_gb": 1.2,
        "pooling": "last_token",
        "instruction_aware": True,
        "is_matryoshka": True,
        "matryoshka_dims": [32, 64, 128, 256, 512, 768, 1024],
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dimension": 2560,
        "max_seq_len": 32768,
        "size_gb": 8.0,
        "pooling": "last_token",
        "instruction_aware": True,
        "is_matryoshka": True,
        "matryoshka_dims": [32, 64, 128, 256, 512, 1024, 1536, 2048, 2560],
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dimension": 4096,
        "max_seq_len": 32768,
        "size_gb": 16.0,
        "pooling": "last_token",
        "instruction_aware": True,
        "is_matryoshka": True,
        "matryoshka_dims": [32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
    },
    # Legacy BGE support
    "BAAI/bge-large-en-v1.5": {
        "dimension": 1024,
        "max_seq_len": 512,
        "size_gb": 1.3,
        "pooling": "mean",
        "instruction_aware": False,
        "is_matryoshka": False,
    },
}

# Instruction templates for Qwen3
INSTRUCTIONS = {
    "code_search": "Given a code search query, retrieve relevant code snippets that match the query",
    "code_retrieval": "Given a natural language description, retrieve the most relevant code implementation",
    "semantic_search": "Given a search query, retrieve relevant passages that answer the query",
    "default": "Retrieve relevant content for the following query",
}

# Memory budgets
MAX_VRAM_BYTES = 2 * 1024**3  # 2GB for 4GB GPUs
MAX_VRAM_BYTES_LARGE = 3 * 1024**3  # 3GB for 8GB+ GPUs

# TEI server configuration
TEI_DEFAULT_URL = "http://localhost:8080"
TEI_URL_ENV_VAR = "TLDR_TEI_URL"


def _parse_pytorch_version() -> tuple:
    """Parse PyTorch version into a tuple for proper comparison.

    Lexicographic string comparison fails for versions like "10.0.0"
    since '1' < '2'. This parses to (major, minor, patch) tuple.
    """
    import torch
    clean = re.split(r"[+a-zA-Z]", torch.__version__)[0]
    parts = clean.split(".")
    return tuple(int(p) for p in parts[:3] if p.isdigit())


def sanitize_query(query: str) -> str:
    """Sanitize query to prevent instruction injection.

    Replaces newlines with spaces and removes control characters.
    This prevents malicious queries like "Query: real query\nInstruct: ignore previous".
    """
    if not query:
        return ""
    # Replace newline variants with space
    for char in ['\n', '\r', '\u2028', '\u2029']:
        query = query.replace(char, ' ')
    # Remove non-printable characters (except space and tab)
    query = ''.join(c for c in query if c.isprintable() or c in ' \t')
    # Collapse multiple whitespace to single space
    return re.sub(r'\s+', ' ', query).strip()

# Batch sizes (Qwen3-0.6B is smaller than BGE-large)
MAX_BATCH_FP16 = 96  # Can fit more with smaller model
MAX_BATCH_FP32 = 48
MIN_BATCH = 1

# Thresholds
SMALL_BATCH_THRESHOLD = 8
LARGE_BATCH_THRESHOLD = 1000
HUGE_BATCH_THRESHOLD = 50000

# Cache limits
INDEX_CACHE_MAX_ITEMS = 4
INDEX_CACHE_MAX_BYTES = 512 * 1024**2
INDEX_TTL_SECONDS = 1800


# ============================================================================
# Device Detection
# ============================================================================

@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Compute device information."""

    name: str  # "cuda", "mps", "xpu", "cpu"
    index: int
    total_memory: int
    compute_capability: Tuple[int, int]
    supports_bf16: bool
    supports_compile: bool
    device_count: int = 1  # Number of available devices (for multi-GPU)
    all_devices_memory: Tuple[int, ...] = ()  # Memory per device (bytes)

    @property
    def memory_budget(self) -> int:
        if self.name == "cpu":
            return 0
        if self.total_memory >= 8 * 1024**3:
            return MAX_VRAM_BYTES_LARGE
        return MAX_VRAM_BYTES

    @property
    def optimal_dtype_str(self) -> str:
        if self.supports_bf16:
            return "bfloat16"
        if self.name == "cpu":
            return "float32"
        return "float16"


def detect_device() -> DeviceInfo:
    """Detect best available compute device."""
    import torch

    # CUDA (includes ROCm)
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        cc = (props.major, props.minor)
        device_count = torch.cuda.device_count()
        all_devices_memory = tuple(
            torch.cuda.get_device_properties(i).total_memory
            for i in range(device_count)
        )
        return DeviceInfo(
            name="cuda",
            index=idx,
            total_memory=props.total_memory,
            compute_capability=cc,
            supports_bf16=cc >= (8, 0),
            supports_compile=cc >= (7, 0),
            device_count=device_count,
            all_devices_memory=all_devices_memory,
        )

    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            name="mps",
            index=0,
            total_memory=0,
            compute_capability=(0, 0),
            supports_bf16=True,
            supports_compile=False,
        )

    # Intel XPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return DeviceInfo(
            name="xpu",
            index=0,
            total_memory=0,
            compute_capability=(0, 0),
            supports_bf16=True,
            supports_compile=False,
        )

    # CPU - bf16 supported in PyTorch 2.0+
    # Use proper version tuple comparison (not lexicographic string)
    cpu_bf16 = _parse_pytorch_version() >= (2, 0, 0)
    return DeviceInfo(
        name="cpu",
        index=0,
        total_memory=0,
        compute_capability=(0, 0),
        supports_bf16=cpu_bf16,
        supports_compile=True,
    )


def get_dtype(device: DeviceInfo) -> "torch.dtype":
    """Get optimal dtype for device."""
    import torch

    if device.supports_bf16:
        return torch.bfloat16
    if device.name == "cpu":
        return torch.float32
    return torch.float16


# ============================================================================
# TEI Backend (text-embeddings-inference)
# ============================================================================

def check_tei_available(
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: float = 5.0,
) -> bool:
    """Check if TEI gRPC server is available.

    Args:
        host: TEI server host. Defaults to TLDR_TEI_HOST env var or localhost.
        port: TEI server port. Defaults to TLDR_TEI_PORT env var or 18080.
        timeout: Connection timeout in seconds. Defaults to 5.0s (not 120s).

    Returns:
        True if TEI gRPC server is reachable.
    """
    try:
        from .tei_client import is_available
        return is_available(host=host, port=port, timeout=timeout)
    except ImportError:
        return False


def check_vllm_available() -> bool:
    """Check if vLLM is available (deprecated - kept for backwards compat)."""
    # TEI replaces vLLM - this now checks TEI
    return check_tei_available()


class TEIEmbedder:
    """TEI-based embedder using text-embeddings-inference gRPC server.

    High-throughput Rust server for embeddings. Requires TEI gRPC server running:
        docker run --gpus all -p 18080:80 \\
            ghcr.io/huggingface/text-embeddings-inference:89-latest-grpc \\
            --model-id Qwen/Qwen3-Embedding-0.6B --pooling last-token

    Features:
    - 2.4x faster than vLLM on short sequences
    - Native MRL dimension truncation via 'dimensions' parameter
    - No Python distributed state to manage (stateless gRPC)
    - Works with any Python version (Rust binary)
    - Fast tokenizer for accurate token counting
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """Initialize TEI gRPC embedder.

        Args:
            model_name: Model name (must match TEI server's --model-id).
            host: TEI server host. Defaults to TLDR_TEI_HOST env var or localhost.
            port: TEI server port. Defaults to TLDR_TEI_PORT env var or 18080.
        """
        from .tei_client import TEIClient

        self.model_name = model_name
        self.model_info = SUPPORTED_MODELS.get(model_name, SUPPORTED_MODELS[DEFAULT_MODEL])

        # Create client
        self._client = TEIClient(host=host, port=port)

        # Get server info
        try:
            info = self._client.info()
            self._server_model = info.model_id
            self.dimension = self.model_info["dimension"]
            self._max_input_length = info.max_input_length or 32768
        except Exception as e:
            self._client.close()
            raise RuntimeError(
                f"Failed to connect to TEI gRPC server. "
                f"Start TEI with: docker run --gpus all -p 18080:80 "
                f"ghcr.io/huggingface/text-embeddings-inference:89-latest-grpc "
                f"--model-id {model_name} --pooling last-token. Error: {e}"
            ) from e

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        dimension: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode texts to embeddings using gRPC streaming.

        Args:
            texts: List of strings to encode.
            normalize: If True, L2-normalize the embeddings (TEI handles this server-side).
            dimension: Optional output dimension for MRL truncation.
                       If specified, must be <= model's native dimension.

        Returns:
            np.ndarray of shape (len(texts), dimension or self.dimension).
        """
        import numpy as np

        # Normalize invalid dimensions early - treat 0 and negative as None
        # This prevents dimension=0 from being passed to TEI
        if dimension is not None and dimension <= 0:
            dimension = None

        output_dim = dimension if dimension is not None else self.dimension

        if not texts:
            return np.empty((0, output_dim), dtype=np.float32)

        # Validate dimension exceeds model capacity
        if dimension is not None and dimension > self.dimension:
            raise ValueError(
                f"Requested dimension {dimension} exceeds model's native dimension {self.dimension}"
            )

        return self._client.embed(texts, normalize=normalize, dimensions=dimension)

    def encode_with_instruction(
        self,
        texts: List[str],
        instruction: str,
        normalize: bool = True,
        dimension: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode texts with instruction prefix (for queries).

        Args:
            texts: List of texts to encode.
            instruction: Instruction prefix for the query.
            normalize: Whether to L2-normalize embeddings.
            dimension: Target dimension for MRL truncation.

        Returns:
            Embeddings array of shape (len(texts), dimension or self.dimension).
        """
        # Sanitize inputs to prevent instruction injection
        sanitized = [sanitize_query(text) for text in texts]
        # Note: No space after "Query:" per Qwen3-Embedding spec
        prefixed = [f"Instruct: {instruction}\nQuery:{text}" for text in sanitized]
        return self.encode(prefixed, normalize=normalize, dimension=dimension)

    def count_tokens(self, text: str) -> int:
        """Count tokens using TEI's fast tokenizer.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        return self._client.count_tokens(text)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts efficiently.

        Args:
            texts: List of texts.

        Returns:
            List of token counts.
        """
        return self._client.count_tokens_batch(texts)

    def cleanup(self) -> None:
        """Close gRPC channel.

        Unlike vLLM, TEI is stateless - no distributed state to clean up.
        """
        if hasattr(self, "_client") and self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass


# ============================================================================
# SentenceTransformers Backend (Fallback)
# ============================================================================

class SentenceTransformersEmbedder:
    """SentenceTransformers-based embedder (fallback when vLLM unavailable)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        compile_model: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_info = SUPPORTED_MODELS.get(model_name, SUPPORTED_MODELS[DEFAULT_MODEL])

        # Detect device - cache device info BEFORE mutating device variable
        if device == "auto":
            self._device_info = detect_device()
            device = self._device_info.name
        else:
            self._device_info = None

        # Load model with optimal settings
        model_kwargs = {}
        tokenizer_kwargs = {}

        # Qwen3 needs left padding
        if "Qwen3" in model_name:
            tokenizer_kwargs["padding_side"] = "left"
            # Check if flash_attn is actually importable before setting attn_implementation
            # Setting dict key never throws, but using flash_attention_2 without
            # flash_attn installed will crash during model load
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass  # Flash attention not installed, use default attention

        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            trust_remote_code=True,
        )
        self.model.eval()

        # Apply dtype
        dev = detect_device()
        dtype = get_dtype(dev)
        if dtype != __import__("torch").float32:
            self.model = self.model.to(dtype)

        # torch.compile for faster inference
        self._compiled = False
        if compile_model and dev.supports_compile:
            try:
                import torch
                backbone = self.model._modules["0"].auto_model
                compiled = torch.compile(backbone, mode="reduce-overhead", fullgraph=False)
                self.model._modules["0"].auto_model = compiled
                self._compiled = True
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}", file=sys.stderr)

        self.dimension = self.model_info["dimension"]

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False,
        dimension: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode.
            batch_size: Batch size for encoding. Auto-detected if None.
            normalize: Whether to L2-normalize embeddings.
            show_progress: Whether to show progress bar for large batches.
            dimension: Target dimension for MRL truncation. Must be <= model dimension.
                       Truncation happens BEFORE normalization per MRL spec.

        Returns:
            Embeddings array of shape (len(texts), dimension or self.dimension).
        """
        import numpy as np
        import torch

        # Validate dimension parameter
        if dimension is not None:
            if dimension > self.dimension:
                raise ValueError(
                    f"dimension={dimension} exceeds model dimension={self.dimension}"
                )
            if dimension <= 0:
                raise ValueError(f"dimension must be positive, got {dimension}")

        output_dim = dimension if dimension is not None else self.dimension

        if not texts:
            return np.empty((0, output_dim), dtype=np.float32)

        if batch_size is None:
            dev = detect_device()
            dtype_bytes = 2 if dev.supports_bf16 else 4
            batch_size = MAX_BATCH_FP16 if dtype_bytes == 2 else MAX_BATCH_FP32

        # MRL truncation must happen BEFORE normalization
        # If truncating, we get raw embeddings and normalize after truncation
        needs_manual_normalize = dimension is not None and normalize

        with torch.inference_mode():
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize and not needs_manual_normalize,
                convert_to_numpy=True,
                show_progress_bar=show_progress and len(texts) >= LARGE_BATCH_THRESHOLD,
            )

        # MRL truncation - must happen BEFORE normalization
        if dimension is not None and dimension < embeddings.shape[1]:
            embeddings = embeddings[:, :dimension]

        # Manual normalization after truncation
        if needs_manual_normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero for zero vectors
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def encode_with_instruction(
        self,
        texts: List[str],
        instruction: str,
        normalize: bool = True,
        dimension: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode texts with instruction prefix.

        Args:
            texts: List of texts to encode.
            instruction: Instruction prefix for the query.
            normalize: Whether to L2-normalize embeddings.
            dimension: Target dimension for MRL truncation.

        Returns:
            Embeddings array of shape (len(texts), dimension or self.dimension).
        """
        # Sanitize inputs to prevent instruction injection
        sanitized = [sanitize_query(text) for text in texts]
        # Note: No space after "Query:" per Qwen3-Embedding spec
        prefixed = [f"Instruct: {instruction}\nQuery:{text}" for text in sanitized]
        return self.encode(prefixed, normalize=normalize, dimension=dimension)


# ============================================================================
# Unified Model Manager
# ============================================================================

class ModelManager:
    """Singleton manager for embedding model.

    Automatically selects TEI (preferred) or SentenceTransformers backend.
    Thread-safe singleton with proper initialization under lock.
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        # Double-checked locking pattern with ALL initialization under lock
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    # Initialize ALL state inside lock to prevent race condition
                    # where other threads see partially initialized object
                    obj._embedder: Optional[TEIEmbedder | SentenceTransformersEmbedder] = None
                    obj._model_name: Optional[str] = None
                    obj._backend: Optional[str] = None
                    obj._device: Optional[DeviceInfo] = None
                    obj._model_lock = threading.RLock()
                    obj._initialized = True  # Set LAST, after all state initialized
                    cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        # All initialization done in __new__ - this is now a no-op
        pass

    @property
    def device(self) -> DeviceInfo:
        if self._device is None:
            self._device = detect_device()
        return self._device

    @property
    def backend(self) -> str:
        """Get current backend name."""
        return self._backend or "none"

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._embedder:
            return self._embedder.dimension
        return SUPPORTED_MODELS[DEFAULT_MODEL]["dimension"]

    def load(
        self,
        model_name: str = DEFAULT_MODEL,
        backend: str = "auto",
    ) -> None:
        """Load the embedding model.

        Args:
            model_name: Model name from SUPPORTED_MODELS or HuggingFace path.
            backend: "tei", "sentence_transformers", or "auto" (prefer TEI).
                     "vllm" is accepted for backwards compat but maps to "tei".
        """
        with self._model_lock:
            # Map legacy "vllm" to "tei" for backwards compatibility
            if backend == "vllm":
                backend = "tei"

            # Resolve "auto" backend first so we can compare properly
            resolved_backend = backend
            if backend == "auto":
                if check_tei_available():
                    resolved_backend = "tei"
                else:
                    resolved_backend = "sentence_transformers"

            # Check BOTH model AND backend - prevents silently ignoring backend switch
            if (self._embedder is not None
                and self._model_name == model_name
                and self._backend == resolved_backend):
                return

            # Create new embedder FIRST - keep old one until new succeeds
            # This prevents leaving system with no embedder if load fails
            try:
                if resolved_backend == "tei":
                    new_embedder = TEIEmbedder(model_name)
                else:
                    new_embedder = SentenceTransformersEmbedder(model_name)
            except Exception as e:
                # Keep old embedder functional if new load fails
                raise RuntimeError(
                    f"Failed to load model '{model_name}' with backend "
                    f"'{resolved_backend}': {e}"
                ) from e

            # Only cleanup old embedder after new one succeeds
            if self._embedder is not None:
                if hasattr(self._embedder, "cleanup"):
                    self._embedder.cleanup()
                del self._embedder
                gc.collect()

            self._embedder = new_embedder
            self._model_name = model_name
            self._backend = resolved_backend

    def encode(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = True,
        dimension: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode texts to embeddings.

        Args:
            texts: Texts to encode.
            instruction: Optional instruction for query encoding (Qwen3 style).
            batch_size: Batch size (auto if None).
            show_progress: Show progress bar.
            normalize: L2 normalize embeddings.
            dimension: Target dimension for MRL truncation. Must be <= model dimension.

        Returns:
            numpy array of shape (len(texts), dimension or self.dimension).
        """
        import numpy as np

        output_dim = dimension if dimension is not None else self.dimension
        if not texts:
            return np.empty((0, output_dim), dtype=np.float32)

        # Fix TOCTOU race: capture embedder reference under lock
        # Without this, another thread could call unload() between check and use
        with self._model_lock:
            if self._embedder is None:
                self.load()
            embedder = self._embedder
            if embedder is None:
                raise RuntimeError("Failed to load embedder - check model name and backend")

        # Safe to use - we hold a reference (even if another thread unloads the manager's ref)
        # Use instruction-aware encoding if provided
        if instruction:
            return embedder.encode_with_instruction(
                texts, instruction, normalize=normalize, dimension=dimension
            )

        # Standard encoding
        if isinstance(embedder, TEIEmbedder):
            return embedder.encode(texts, normalize=normalize, dimension=dimension)
        else:
            return embedder.encode(
                texts,
                batch_size=batch_size,
                normalize=normalize,
                show_progress=show_progress,
                dimension=dimension,
            )

    def encode_query(
        self,
        query: str,
        task: str = "code_search",
        dimension: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode a query with appropriate instruction.

        Args:
            query: The query text.
            task: Task type for instruction selection.
            dimension: Target dimension for MRL truncation. Must be <= model dimension.

        Returns:
            1D embedding vector of shape (dimension or self.dimension,).
        """
        instruction = INSTRUCTIONS.get(task, INSTRUCTIONS["default"])
        embeddings = self.encode([query], instruction=instruction, dimension=dimension)
        return embeddings[0]

    def encode_documents(
        self,
        documents: List[str],
        show_progress: bool = False,
        dimension: Optional[int] = None,
    ) -> "np.ndarray":
        """Encode documents (no instruction prefix).

        For Qwen3, documents don't need instruction prefixes.

        Args:
            documents: List of documents to encode.
            show_progress: Show progress bar for large batches.
            dimension: Target dimension for MRL truncation. Must be <= model dimension.

        Returns:
            numpy array of shape (len(documents), dimension or self.dimension).
        """
        return self.encode(
            documents, instruction=None, show_progress=show_progress, dimension=dimension
        )

    def warmup(self) -> Dict[str, Any]:
        """Warm up the model."""
        if self._embedder is None:
            self.load()

        start = time.perf_counter()
        _ = self.encode(["warmup text"] * 4)
        elapsed = time.perf_counter() - start

        return {
            "warmup_time_s": elapsed,
            "device": self.device.name,
            "backend": self.backend,
            "model": self._model_name,
            "dimension": self.dimension,
        }

    def memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        import torch

        stats = {
            "device": self.device.name,
            "backend": self.backend,
            "model": self._model_name,
            "dimension": self.dimension,
        }

        if self.device.name == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            stats["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            stats["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            stats["gpu_total_mb"] = self.device.total_memory / 1024**2

        return stats

    def unload(self) -> None:
        """Unload model and free memory."""
        import torch

        with self._model_lock:
            if self._embedder is not None:
                # Call cleanup for vLLM to release distributed state
                if hasattr(self._embedder, 'cleanup'):
                    self._embedder.cleanup()

                del self._embedder
                self._embedder = None
                self._model_name = None
                self._backend = None

            gc.collect()
            if torch.cuda.is_available():
                # Clear cache on all CUDA devices (not just current)
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete


# ============================================================================
# Index Manager
# ============================================================================

@dataclass
class CachedIndex:
    """Cached usearch index."""

    index: Any  # usearch.Index
    metadata: Dict[str, Any]
    size_bytes: int
    loaded_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()

    @property
    def age_seconds(self) -> float:
        """Time since index was loaded."""
        return time.time() - self.loaded_at

    @property
    def idle_seconds(self) -> float:
        """Time since last access - used for TTL staleness."""
        return time.time() - self.last_access

    @property
    def is_stale(self) -> bool:
        # FIX: Use idle_seconds not age_seconds
        # An index loaded 31 min ago but accessed 1 sec ago should NOT be stale
        return self.idle_seconds > INDEX_TTL_SECONDS


class IndexManager:
    """LRU cache for usearch indexes.

    Thread-safe singleton with proper initialization under lock.
    All state initialized in __new__ to prevent race conditions where
    another thread sees a partially initialized instance.
    """

    _instance: Optional["IndexManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "IndexManager":
        # Double-checked locking with ALL initialization under lock
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    # Initialize ALL state inside lock to prevent race condition
                    # where other threads see partially initialized object
                    obj._cache: Dict[str, CachedIndex] = {}
                    obj._cache_lock = threading.Lock()
                    obj._loading_locks: Dict[str, threading.Lock] = {}  # Per-index locks
                    obj._loading_locks_lock = threading.Lock()  # Guard for _loading_locks dict
                    obj._total_size: int = 0
                    obj._initialized = True  # Set LAST, after all state initialized
                    cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        # All initialization done in __new__ - this is now a no-op
        pass

    def _estimate_index_size(self, num_vectors: int, dim: int) -> int:
        return num_vectors * dim * 4

    def _evict_if_needed(self, required_bytes: int) -> None:
        while len(self._cache) >= INDEX_CACHE_MAX_ITEMS:
            self._evict_lru()
        while self._total_size + required_bytes > INDEX_CACHE_MAX_BYTES:
            if not self._cache:
                break
            self._evict_lru()

    def _evict_lru(self) -> None:
        if not self._cache:
            return
        # FIX: Pick true LRU among stale entries, not arbitrary dict order
        stale = [(k, v.last_access) for k, v in self._cache.items() if v.is_stale]
        if stale:
            # Pick the least recently used among stale entries
            lru_key = min(stale, key=lambda x: x[1])[0]
        else:
            lru_key = min(self._cache, key=lambda k: self._cache[k].last_access)
        entry = self._cache.pop(lru_key)
        self._total_size -= entry.size_bytes
        # Explicit cleanup of native usearch index
        if hasattr(entry.index, 'clear'):
            try:
                entry.index.clear()
            except Exception:
                pass

    def _get_loading_lock(self, cache_key: str) -> threading.Lock:
        """Get or create a loading lock for a specific index.

        Thread-safe: uses separate lock to guard the loading_locks dict.
        """
        with self._loading_locks_lock:
            if cache_key not in self._loading_locks:
                self._loading_locks[cache_key] = threading.Lock()
            return self._loading_locks[cache_key]

    def get(self, project_path: str, force_reload: bool = False) -> CachedIndex:
        """Get or load index for a project.

        Thread-safe with per-index locks: loading index A doesn't block
        threads loading index B. Only threads loading the same index wait.
        """
        from usearch.index import Index

        project = Path(project_path).resolve()
        cache_key = str(project)

        # Fast path: check cache without blocking other loaders
        if not force_reload:
            with self._cache_lock:
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    if not entry.is_stale:
                        entry.touch()
                        return entry
                    # Stale entry - remove it
                    self._cache.pop(cache_key)
                    self._total_size -= entry.size_bytes

        # Get per-index lock - threads loading different indexes don't block each other
        loading_lock = self._get_loading_lock(cache_key)

        with loading_lock:
            # Double-check after acquiring lock - another thread may have loaded it
            with self._cache_lock:
                if cache_key in self._cache:
                    entry = self._cache[cache_key]
                    if not entry.is_stale:
                        entry.touch()
                        return entry
                    # Stale entry - remove it
                    self._cache.pop(cache_key)
                    self._total_size -= entry.size_bytes

            # Load from disk - OUTSIDE cache_lock so other indexes can be accessed
            cache_dir = project / ".tldr" / "cache" / "semantic"
            index_file = cache_dir / "index.usearch"
            metadata_file = cache_dir / "metadata.json"

            # CRIT-001: Detect legacy FAISS index and provide migration guidance
            faiss_file = cache_dir / "index.faiss"
            if faiss_file.exists() and not index_file.exists():
                raise FileNotFoundError(
                    f"Found legacy FAISS index at {faiss_file}. "
                    f"FAISS indexes are no longer supported. "
                    f"Please rebuild with: tldr semantic index {project}"
                )

            if not index_file.exists():
                raise FileNotFoundError(f"Index not found: {index_file}")
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_file}")

            # CRIT-002: Check if index is being rebuilt - warn but don't block
            # This prevents reading mismatched index/metadata during rebuild
            building_lock = cache_dir / ".building"
            if building_lock.exists():
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Index at {cache_dir} is being rebuilt. "
                    "Search results may be inconsistent. "
                    "Wait for rebuild to complete for accurate results."
                )

            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            dim = metadata.get("dimension", 1024)

            index = Index.restore(str(index_file))

            # HIGH-001: Validate index dimension matches metadata
            # Prevents silent corruption from mismatched index/metadata files
            if index.ndim != dim:
                raise ValueError(
                    f"Index dimension mismatch: index has {index.ndim} dims, "
                    f"metadata claims {dim}. Index may be corrupted. "
                    f"Rebuild with: tldr semantic index {project}"
                )

            num_vectors = len(index)
            size_bytes = self._estimate_index_size(num_vectors, dim)

            entry = CachedIndex(index=index, metadata=metadata, size_bytes=size_bytes)

            # Update cache under lock
            with self._cache_lock:
                self._evict_if_needed(size_bytes)
                self._cache[cache_key] = entry
                self._total_size += size_bytes

            return entry

    def invalidate(self, project_path: str) -> None:
        cache_key = str(Path(project_path).resolve())
        entry_to_cleanup = None
        with self._cache_lock:
            if cache_key in self._cache:
                entry_to_cleanup = self._cache.pop(cache_key)
                self._total_size -= entry_to_cleanup.size_bytes
        # Cleanup outside lock to avoid holding lock during potentially slow operation
        if entry_to_cleanup is not None:
            if hasattr(entry_to_cleanup.index, 'clear'):
                try:
                    entry_to_cleanup.index.clear()
                except Exception:
                    pass
            gc.collect()

    def clear(self) -> None:
        """Clear all cached indexes and free native memory."""
        entries_to_cleanup = []
        with self._cache_lock:
            entries_to_cleanup = list(self._cache.values())
            self._cache.clear()
            self._total_size = 0

        # Cleanup native usearch resources outside lock
        for entry in entries_to_cleanup:
            if hasattr(entry.index, 'clear'):
                try:
                    entry.index.clear()
                except Exception:
                    pass

        gc.collect()

    def stats(self) -> Dict[str, Any]:
        with self._cache_lock:
            return {
                "cached_projects": len(self._cache),
                "total_size_mb": self._total_size / 1024**2,
                "max_size_mb": INDEX_CACHE_MAX_BYTES / 1024**2,
            }


# ============================================================================
# High-Level API
# ============================================================================

def get_model_manager() -> ModelManager:
    return ModelManager()


def get_index_manager() -> IndexManager:
    return IndexManager()


def encode_batch(
    texts: List[str],
    instruction: Optional[str] = None,
    show_progress: bool = False,
) -> "np.ndarray":
    """Encode texts to embeddings."""
    return get_model_manager().encode(texts, instruction=instruction, show_progress=show_progress)


def search(
    project_path: str,
    query: str,
    k: int = 5,
    task: str = "code_search",
    expand_graph: bool = False,
    dimension: Optional[int] = None,
    backend: str = "auto",
) -> List[Dict[str, Any]]:
    """Semantic search with Qwen3 embeddings.

    Args:
        project_path: Path to the indexed project.
        query: Search query text.
        k: Number of results to return. Must be positive.
        task: Task type for instruction selection.
        expand_graph: Whether to include call graph info in results.
        dimension: Optional query dimension for MRL. Must match index dimension.
        backend: Inference backend - "vllm", "sentence_transformers", or "auto" (default).

    Returns:
        List of search results with name, file, line, score, etc.

    Raises:
        ValueError: If dimension doesn't match the index dimension.
    """
    import numpy as np

    # Validate k early - prevents useless work and potential issues
    if k <= 0:
        return []

    if not query or not query.strip():
        return []

    cached = get_index_manager().get(project_path)
    index = cached.index
    units = cached.metadata["units"]

    # Validate dimension matches index
    index_dim = cached.metadata.get("dimension")
    if dimension is not None and index_dim and dimension != index_dim:
        raise ValueError(
            f"Query dimension {dimension} != index dimension {index_dim}. "
            f"Rebuild index with matching dimension."
        )

    mm = get_model_manager()
    # Load model with specified backend (uses model from index metadata)
    model_name = cached.metadata.get("model", DEFAULT_MODEL)
    mm.load(model_name, backend=backend)
    query_embedding = mm.encode_query(query, task=task, dimension=dimension)

    k = min(k, len(units))
    # usearch returns (keys, distances) - 1D arrays for single query
    matches = index.search(query_embedding, k)
    indices = matches.keys  # Shape: (k,)
    # Convert distance to similarity, clamp to [0, 1]
    # IP metric: distance = 1 - inner_product, so score = 1 - dist = IP
    # For normalized vectors, IP ranges [-1, 1], clamp negatives to 0
    raw_scores = 1.0 - matches.distances
    scores = np.clip(raw_scores, 0.0, 1.0)

    results = []
    for score, idx in zip(scores, indices):  # Direct iteration, no [0]
        if int(idx) >= len(units):  # uint64 can't be < 0
            continue
        unit = units[idx]
        result = {
            "name": unit["name"],
            "qualified_name": unit["qualified_name"],
            "file": unit["file"],
            "line": unit["line"],
            "unit_type": unit["unit_type"],
            "signature": unit["signature"],
            "score": float(score),
        }
        if expand_graph:
            result["calls"] = unit.get("calls", [])
            result["called_by"] = unit.get("called_by", [])
        results.append(result)

    return results


def build_index(
    project_path: str,
    lang: str = "python",
    model_name: str = DEFAULT_MODEL,
    show_progress: bool = True,
    respect_ignore: bool = True,
    dimension: Optional[int] = None,
    backend: str = "auto",
) -> int:
    """Build semantic index with Qwen3 embeddings and usearch.

    Args:
        project_path: Path to project root.
        lang: Language to extract units from (python, typescript, go, rust).
        model_name: Embedding model name from SUPPORTED_MODELS.
        show_progress: Show progress bar during encoding.
        respect_ignore: Honor .tldrignore patterns.
        dimension: Target dimension for MRL (Matryoshka Representation Learning) truncation.
                   Must be <= model's native dimension. Smaller dimensions reduce index size
                   and search latency with minimal accuracy loss for supported models.
                   If None, uses the model's full native dimension.
        backend: Inference backend - "vllm", "sentence_transformers", or "auto" (default).
                 "auto" prefers vLLM if available, falls back to SentenceTransformers.

    Returns:
        Number of units indexed.
    """
    import shutil
    import uuid
    import numpy as np
    from usearch.index import Index, MetricKind
    from tldr.semantic import build_embedding_text, extract_units_from_project
    from tldr.tldrignore import ensure_tldrignore

    project = Path(project_path).resolve()
    ensure_tldrignore(project)

    units = extract_units_from_project(str(project), lang=lang, respect_ignore=respect_ignore)
    if not units:
        return 0

    texts = [build_embedding_text(u) for u in units]
    n = len(texts)

    mm = get_model_manager()
    mm.load(model_name, backend=backend)
    embeddings = mm.encode_documents(texts, show_progress=show_progress, dimension=dimension)

    actual_dim = embeddings.shape[1]
    native_dim = SUPPORTED_MODELS.get(model_name, {}).get("dimension", actual_dim)

    # usearch with cosine similarity (IP on normalized vectors)
    index = Index(ndim=actual_dim, metric=MetricKind.IP, dtype="f32")
    # Add vectors with sequential keys (0, 1, 2, ...)
    keys = np.arange(n, dtype=np.uint64)
    index.add(keys, embeddings)

    metadata = {
        "version": 2,
        "units": [u.to_dict() for u in units],
        "model": model_name,
        "dimension": actual_dim,
        "native_dimension": native_dim,
        "is_mrl_truncated": dimension is not None and actual_dim < native_dim,
        "count": n,
        "index_type": "usearch",
        "languages": [lang],
        "created_at": datetime.now().isoformat(),
    }

    cache_dir = project / ".tldr" / "cache" / "semantic"
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_file = cache_dir / "index.usearch"
    metadata_file = cache_dir / "metadata.json"
    # CRIT-002: Lock file signals rebuild in progress to concurrent readers
    building_lock = cache_dir / ".building"
    # Use unique temp file names to prevent conflicts from stale files or concurrent builds
    unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    temp_index = cache_dir / f"index.{unique_id}.tmp"
    temp_metadata = cache_dir / f"metadata.{unique_id}.tmp"

    try:
        # Signal rebuild in progress - readers will see warning
        building_lock.touch()

        index.save(str(temp_index))
        temp_metadata.write_text(json.dumps(metadata, indent=2))
        # Use shutil.move instead of Path.rename - works across filesystems
        # Atomic on POSIX: both files are moved to final location
        shutil.move(str(temp_index), str(index_file))
        shutil.move(str(temp_metadata), str(metadata_file))
    finally:
        # Cleanup temp files if they still exist (partial failure case)
        for temp_file in [temp_index, temp_metadata]:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
        # Remove lock file - rebuild complete (or failed)
        try:
            if building_lock.exists():
                building_lock.unlink()
        except Exception:
            pass

    get_index_manager().invalidate(project_path)
    return n


# ============================================================================
# Diagnostics
# ============================================================================

def print_device_info() -> None:
    """Print device and backend information."""
    mm = get_model_manager()
    device = mm.device

    print(f"Device: {device.name}")
    if device.name == "cuda":
        print(f"  Compute capability: {device.compute_capability}")
        print(f"  Total VRAM: {device.total_memory / 1024**3:.1f} GB")
    print(f"  bf16 support: {device.supports_bf16}")
    print(f"  torch.compile: {device.supports_compile}")
    print(f"  Optimal dtype: {device.optimal_dtype_str}")
    print(f"TEI server available: {check_tei_available()}")


def benchmark(num_texts: int = 100) -> Dict[str, Any]:
    """Benchmark encoding performance."""
    mm = get_model_manager()
    warmup_stats = mm.warmup()

    texts = [f"def function_{i}(x): return x * {i}" for i in range(num_texts)]

    start = time.perf_counter()
    _ = mm.encode_documents(texts)
    elapsed = time.perf_counter() - start

    return {
        **warmup_stats,
        "num_texts": num_texts,
        "encode_time_s": elapsed,
        "texts_per_second": num_texts / elapsed,
        "memory": mm.memory_stats(),
    }
