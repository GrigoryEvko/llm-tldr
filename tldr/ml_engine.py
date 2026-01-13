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

import fcntl
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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from tldr.tokenizer import estimate_tokens_fallback

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
# Architecture params added for VRAM prediction (hidden_dim, num_layers, num_heads, vocab_size, intermediate_size)
SUPPORTED_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "dimension": 1024,
        "max_seq_len": 32768,
        "size_gb": 1.2,
        "pooling": "last_token",
        "instruction_aware": True,
        "is_matryoshka": True,
        "matryoshka_dims": [32, 64, 128, 256, 512, 768, 1024],
        # Architecture for VRAM prediction (verified from HuggingFace config.json)
        "hidden_dim": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "vocab_size": 151669,
        "intermediate_size": 3072,  # SwiGLU intermediate
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dimension": 2560,
        "max_seq_len": 32768,
        "size_gb": 8.0,
        "pooling": "last_token",
        "instruction_aware": True,
        "is_matryoshka": True,
        "matryoshka_dims": [32, 64, 128, 256, 512, 1024, 1536, 2048, 2560],
        # Architecture for VRAM prediction
        "hidden_dim": 2560,
        "num_layers": 36,
        "num_heads": 20,
        "vocab_size": 151936,
        "intermediate_size": 9216,
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dimension": 4096,
        "max_seq_len": 32768,
        "size_gb": 16.0,
        "pooling": "last_token",
        "instruction_aware": True,
        "is_matryoshka": True,
        "matryoshka_dims": [32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096],
        # Architecture for VRAM prediction
        "hidden_dim": 4096,
        "num_layers": 36,
        "num_heads": 32,
        "vocab_size": 151936,
        "intermediate_size": 12288,
    },
    # Legacy BGE support (BERT-based architecture)
    "BAAI/bge-large-en-v1.5": {
        "dimension": 1024,
        "max_seq_len": 512,
        "size_gb": 1.3,
        "pooling": "mean",
        "instruction_aware": False,
        "is_matryoshka": False,
        # Architecture for VRAM prediction
        "hidden_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "vocab_size": 30522,
        "intermediate_size": 4096,
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

# TEI server baseline memory footprint (empirically measured)
# This is the constant overhead when TEI is loaded, before any inference
TEI_BASELINE_MB = 1284
TEI_BASELINE_BYTES = TEI_BASELINE_MB * 1024**2  # ~1.25 GB

# Minimum batch memory to prevent single-item batches on low-VRAM GPUs
# Even with limited VRAM, we need enough memory to batch efficiently
MIN_BATCH_MEMORY_MB = 50  # 50MB minimum
MIN_BATCH_MEMORY_BYTES = MIN_BATCH_MEMORY_MB * 1024**2

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


# Cache for flash attention detection (computed once per process)
_flash_attention_available: Optional[bool] = None


def _has_flash_attention() -> bool:
    """Detect if Flash Attention is available for memory-efficient attention.

    Flash Attention uses O(n) memory instead of O(n^2) for attention scores,
    dramatically reducing VRAM usage for long sequences.

    Detection priority:
    1. flash_attn package (FlashAttention-2): Full FA2 implementation
    2. PyTorch 2.0+ scaled_dot_product_attention (SDPA): Built-in efficient attention

    Returns:
        True if either flash_attn or PyTorch SDPA is available on CUDA.

    Note:
        Result is cached per-process since hardware/software don't change at runtime.
    """
    global _flash_attention_available

    # Return cached result if available
    if _flash_attention_available is not None:
        return _flash_attention_available

    _flash_attention_available = False

    try:
        import torch
        if not torch.cuda.is_available():
            return _flash_attention_available

        # Check for flash_attn package (FlashAttention-2)
        try:
            import flash_attn  # noqa: F401
            _flash_attention_available = True
            return _flash_attention_available
        except ImportError:
            pass

        # Check for PyTorch 2.0+ SDPA (built-in efficient attention)
        # SDPA provides memory-efficient attention on supported hardware
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            version = _parse_pytorch_version()
            if version >= (2, 0, 0):
                _flash_attention_available = True

    except ImportError:
        pass

    return _flash_attention_available


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
# VRAM Prediction for Smart Batching
# ============================================================================

# CUDA allocator overhead and fragmentation factor (empirical)
VRAM_OVERHEAD_FACTOR = 1.15  # 15% overhead for CUDA allocator fragmentation

# Default architecture fallback for unknown models (matches Qwen3-Embedding-0.6B)
DEFAULT_ARCHITECTURE = {
    "hidden_dim": 1024,
    "num_layers": 28,
    "num_heads": 16,
    "vocab_size": 151669,
    "intermediate_size": 3072,
}


def get_available_device_memory(device: str = "auto") -> int:
    """Get available memory in bytes for device.

    For CUDA: Returns free VRAM (total - currently allocated).
    For MPS: Returns estimated available memory (system memory based).
    For CPU: Returns available system RAM.

    Args:
        device: Device identifier - "cuda", "cuda:0", "mps", "cpu", or "auto".
                "auto" selects best available accelerator.

    Returns:
        Available memory in bytes. Returns 0 if device unavailable.

    Raises:
        ValueError: If device string is invalid or device index out of range.
        RuntimeError: If requesting CUDA/MPS memory without PyTorch installed.

    Examples:
        >>> mem = get_available_device_memory("cuda")  # Free VRAM on default GPU
        >>> mem = get_available_device_memory("cuda:1")  # Specific GPU
        >>> mem = get_available_device_memory("auto")  # Best available device
    """
    # Parse device string early (e.g., "cuda:1" -> ("cuda", 1))
    # VP-10 FIX: Consistent ValueError handling for invalid device strings
    if ":" in device:
        device_type, device_idx_str = device.split(":", 1)
        try:
            device_idx = int(device_idx_str)
        except ValueError:
            raise ValueError(f"Invalid device format: {device!r}. Expected format like 'cuda:0' or 'cuda'.")
        if device_idx < 0:
            raise ValueError(f"Device index must be non-negative, got {device_idx}")
    else:
        device_type = device
        device_idx = 0

    # Handle CPU without requiring torch - psutil is sufficient
    if device_type == "cpu":
        try:
            import psutil
            return int(psutil.virtual_memory().available * 0.8)
        except ImportError:
            # Fallback: assume 16GB available for CPU
            return 16 * 1024**3

    # For GPU devices (cuda, mps, auto), we need torch
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            f"Cannot detect {device} memory without PyTorch. "
            "Install torch or specify memory manually."
        )

    # Resolve "auto" to actual device
    if device_type == "auto":
        device_info = detect_device()
        device_type = device_info.name

    if device_type == "cuda":
        if not torch.cuda.is_available():
            return 0
        # VP-10 FIX: Validate device index before accessing
        device_count = torch.cuda.device_count()
        if device_idx >= device_count:
            raise ValueError(
                f"CUDA device {device_idx} not found. "
                f"Available devices: 0-{device_count - 1}" if device_count > 0 else "No CUDA devices available"
            )
        try:
            torch.cuda.synchronize(device_idx)
            free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
            return free_mem
        except Exception:
            # Fallback: estimate from total - allocated
            props = torch.cuda.get_device_properties(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            return max(0, props.total_memory - allocated)

    elif device_type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return 0
        # MPS doesn't expose memory info directly
        # Estimate: use 75% of system RAM as upper bound for unified memory
        try:
            import psutil
            return int(psutil.virtual_memory().available * 0.75)
        except ImportError:
            # Fallback: assume 8GB available
            return 8 * 1024**3

    elif device_type == "cpu":
        # CPU after auto-detection resolved to cpu
        try:
            import psutil
            return int(psutil.virtual_memory().available * 0.8)
        except ImportError:
            return 16 * 1024**3

    return 0


def predict_vram_bytes(
    batch_size: int,
    max_tokens: int,
    output_dim: int = 1024,
    dtype_bytes: int = 2,
    model_name: str = DEFAULT_MODEL,
    flash_attention: Optional[bool] = None,
) -> int:
    """Predict VRAM usage in bytes for embedding inference batch.

    This estimates the dynamic memory required for a batch of texts during
    forward pass. It accounts for:
    - Input embeddings
    - Attention score matrices (O(seq_len^2) standard, O(seq_len) with Flash Attention)
    - FFN intermediate activations
    - Layer outputs and residuals
    - Output embeddings
    - CUDA allocator overhead (~15%)

    Note: This does NOT include model weights (static, loaded once).
    Use estimate_model_memory() for model weight memory.

    Args:
        batch_size: Number of texts in the batch. Must be non-negative.
        max_tokens: Maximum sequence length in the batch. Must be positive.
        output_dim: Output embedding dimension. Must be positive. Default 1024.
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32). Must be positive. Default 2.
        model_name: Model name from SUPPORTED_MODELS for architecture lookup.
        flash_attention: Override Flash Attention detection.
            - None (default): Auto-detect (flash_attn package or PyTorch 2.0+ SDPA)
            - True: Force use of Flash Attention O(n) formula
            - False: Force use of standard O(n^2) formula

    Returns:
        Estimated VRAM usage in bytes for the batch's activation memory.

    Raises:
        ValueError: If batch_size is negative, or if max_tokens, dtype_bytes,
                   or output_dim are non-positive.

    Memory Formula (standard attention):
        Total = (attention + input_emb + ffn_activations + output) * overhead

        Where:
        - attention = batch * num_heads * seq_len^2 * dtype (O(n^2))
        - input_emb = batch * seq_len * hidden_dim * dtype
        - ffn_activations = batch * seq_len * intermediate_size * num_layers * dtype
        - output = batch * output_dim * dtype
        - overhead = 1.15 (CUDA allocator fragmentation)

    Memory Formula (Flash Attention):
        - attention = batch * num_heads * seq_len * head_dim * 2 * dtype (O(n))
        Flash Attention computes attention in blocks, never materializing
        the full O(n^2) attention matrix.

    Examples:
        >>> # Small batch, short sequences
        >>> predict_vram_bytes(8, 512)  # ~50MB
        >>> # Large batch, long sequences (standard attention)
        >>> predict_vram_bytes(32, 2048, flash_attention=False)  # ~1.5GB
        >>> # Large batch, long sequences (Flash Attention)
        >>> predict_vram_bytes(32, 2048, flash_attention=True)  # ~200MB
    """
    # Validate inputs - catch programming errors early with clear messages
    if batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {batch_size}")
    if batch_size == 0:
        return 0  # Valid case: zero items need zero memory
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")

    # Get architecture parameters
    model_config = SUPPORTED_MODELS.get(model_name, SUPPORTED_MODELS.get(DEFAULT_MODEL, {}))
    hidden_dim = model_config.get("hidden_dim", DEFAULT_ARCHITECTURE["hidden_dim"])
    num_layers = model_config.get("num_layers", DEFAULT_ARCHITECTURE["num_layers"])
    num_heads = model_config.get("num_heads", DEFAULT_ARCHITECTURE["num_heads"])
    intermediate_size = model_config.get("intermediate_size", DEFAULT_ARCHITECTURE["intermediate_size"])

    # Calculate head dimension for Flash Attention formula
    head_dim = hidden_dim // num_heads

    # Determine if Flash Attention is being used
    use_flash = flash_attention if flash_attention is not None else _has_flash_attention()

    # 1. Attention memory calculation
    if use_flash:
        # Flash Attention: O(n) memory - never materializes full attention matrix
        # Memory for Q, K, V projections and output accumulator per block
        # Formula: batch * num_heads * seq_len * head_dim * 2 (for Q/O buffers)
        attention_bytes = batch_size * num_heads * max_tokens * head_dim * dtype_bytes * 2
    else:
        # Standard attention: O(n^2) for attention score matrices
        # batch * num_heads * seq_len * seq_len
        attention_bytes = batch_size * num_heads * max_tokens * max_tokens * dtype_bytes

    # 2. Input embeddings: batch * seq_len * hidden_dim
    input_emb_bytes = batch_size * max_tokens * hidden_dim * dtype_bytes

    # 3. FFN intermediate activations (per layer, but reused across layers)
    # For SwiGLU: gate * up activations stored temporarily
    # Conservative: assume peak of 2x intermediate_size per layer (gate + up)
    # Only need to store activations for current layer during forward pass
    ffn_bytes = batch_size * max_tokens * intermediate_size * 2 * dtype_bytes

    # 4. Layer outputs / residuals: batch * seq_len * hidden_dim
    # Need at most 2 copies (current layer output + residual)
    residual_bytes = batch_size * max_tokens * hidden_dim * 2 * dtype_bytes

    # 5. Output embeddings: batch * output_dim
    output_bytes = batch_size * output_dim * dtype_bytes

    # 6. Attention K, V projections for current layer: 2 * batch * seq_len * hidden_dim
    kv_bytes = 2 * batch_size * max_tokens * hidden_dim * dtype_bytes

    # Total with overhead factor for CUDA allocator fragmentation
    total = attention_bytes + input_emb_bytes + ffn_bytes + residual_bytes + output_bytes + kv_bytes
    return int(total * VRAM_OVERHEAD_FACTOR)


def estimate_model_memory(model_name: str = DEFAULT_MODEL, dtype_bytes: int = 2) -> int:
    """Estimate memory required for model weights.

    This is a one-time constant overhead when the model is loaded.

    Args:
        model_name: Model name from SUPPORTED_MODELS.
        dtype_bytes: Bytes per parameter (2 for fp16/bf16, 4 for fp32). Must be positive.

    Returns:
        Estimated model weight memory in bytes.

    Raises:
        ValueError: If dtype_bytes is non-positive.
    """
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")

    model_config = SUPPORTED_MODELS.get(model_name, SUPPORTED_MODELS.get(DEFAULT_MODEL, {}))

    # Use known model size if available
    size_gb = model_config.get("size_gb", 1.2)
    base_bytes = int(size_gb * 1024**3)

    # Scale by dtype (models typically reported in fp16/bf16 = 2 bytes)
    # Mapping: dtype_bytes -> multiplier relative to fp16 baseline
    dtype_scale = {
        1: 0.5,   # int8: half the size
        2: 1.0,   # fp16/bf16: baseline
        4: 2.0,   # fp32: double
        8: 4.0,   # fp64: quadruple
    }.get(dtype_bytes, dtype_bytes / 2)  # Fallback: linear scaling

    return int(base_bytes * dtype_scale)


def calculate_optimal_batch_size(
    token_counts: List[int],
    available_memory_bytes: int,
    target_utilization: float = 0.9,
    output_dim: int = 1024,
    dtype_bytes: int = 2,
    model_name: str = DEFAULT_MODEL,
    min_batch_size: int = 1,
    max_batch_size: int = 256,
    flash_attention: Optional[bool] = None,
) -> int:
    """Calculate optimal batch size to use target percentage of available memory.

    Uses binary search to find the largest batch size that fits within memory budget.
    For variable-length inputs, uses the maximum token count to ensure all sequences fit.

    Args:
        token_counts: List of token counts for texts to be batched.
        available_memory_bytes: Available device memory in bytes.
        target_utilization: Target memory utilization (0.0 < x <= 1.0). Default 0.9 (90%).
        output_dim: Output embedding dimension. Default 1024.
        dtype_bytes: Bytes per element. Default 2 (fp16/bf16).
        model_name: Model name for architecture lookup.
        min_batch_size: Minimum batch size to return. Default 1.
        max_batch_size: Maximum batch size to consider. Default 256.
        flash_attention: Override Flash Attention detection. None=auto, True=force, False=disable.

    Returns:
        Optimal batch size that fits within memory budget.

    Examples:
        >>> tokens = [512, 1024, 768, 256]  # 4 texts with varying lengths
        >>> available = 4 * 1024**3  # 4GB
        >>> batch_size = calculate_optimal_batch_size(tokens, available)
        >>> print(f"Optimal batch: {batch_size}")
    """
    # VP-5 FIX: Validate target_utilization is in valid range (0, 1.0]
    if target_utilization <= 0 or target_utilization > 1.0:
        raise ValueError(
            f"target_utilization must be in (0, 1.0], got {target_utilization}"
        )

    # Validate min_batch_size is positive
    if min_batch_size < 1:
        raise ValueError(f"min_batch_size must be >= 1, got {min_batch_size}")

    # Empty input returns 0 (no batches needed)
    if not token_counts:
        return 0

    # VP-4 FIX: Clamp batch size bounds to actual data size
    data_size = len(token_counts)
    min_batch_size = min(min_batch_size, data_size)
    max_batch_size = min(max_batch_size, data_size)

    # Memory budget
    budget = int(available_memory_bytes * target_utilization)

    # For padded batching, use max sequence length
    max_tokens = max(token_counts)

    # Binary search for largest batch that fits
    low, high = min_batch_size, max_batch_size
    result = min_batch_size

    while low <= high:
        mid = (low + high) // 2
        predicted = predict_vram_bytes(
            batch_size=mid,
            max_tokens=max_tokens,
            output_dim=output_dim,
            dtype_bytes=dtype_bytes,
            model_name=model_name,
            flash_attention=flash_attention,
        )

        if predicted <= budget:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result


def split_into_optimal_batches(
    texts: List[str],
    available_memory_bytes: int,
    target_utilization: float = 0.9,
    output_dim: int = 1024,
    dtype_bytes: int = 2,
    model_name: str = DEFAULT_MODEL,
    count_tokens_fn: Optional[Callable[[str], int]] = None,
    flash_attention: Optional[bool] = None,
) -> List[List[str]]:
    """Split texts into optimally-sized batches for processing.

    Groups texts into batches that maximize memory utilization while staying
    within budget. Uses greedy bin-packing: starts a new batch when adding
    the next text would exceed memory budget.

    Args:
        texts: List of texts to batch.
        available_memory_bytes: Available device memory in bytes.
        target_utilization: Target memory utilization (0.0-1.0). Default 0.9.
        output_dim: Output embedding dimension. Default 1024.
        dtype_bytes: Bytes per element. Default 2.
        model_name: Model name for architecture lookup.
        count_tokens_fn: Optional token counting function. If None, uses
                        character estimation (~4 chars per token).
        flash_attention: Override Flash Attention detection. None=auto, True=force, False=disable.

    Returns:
        List of text batches, each batch sized to fit memory budget.

    Examples:
        >>> texts = ["short", "medium length text", "very long document..."]
        >>> batches = split_into_optimal_batches(texts, 2 * 1024**3)
        >>> for i, batch in enumerate(batches):
        ...     embeddings = model.encode(batch)
    """
    if not texts:
        return []

    # Default token counter: character-type aware estimation
    if count_tokens_fn is None:
        count_tokens_fn = estimate_tokens_fallback

    budget = int(available_memory_bytes * target_utilization)

    batches = []
    current_batch = []
    current_max_tokens = 0

    for text in texts:
        tokens = count_tokens_fn(text)

        # Predict memory if we add this text to current batch
        new_max_tokens = max(current_max_tokens, tokens)
        new_batch_size = len(current_batch) + 1

        predicted = predict_vram_bytes(
            batch_size=new_batch_size,
            max_tokens=new_max_tokens,
            output_dim=output_dim,
            dtype_bytes=dtype_bytes,
            model_name=model_name,
            flash_attention=flash_attention,
        )

        if predicted > budget and current_batch:
            # Current batch is full, start new batch
            batches.append(current_batch)
            current_batch = [text]
            current_max_tokens = tokens
        else:
            # Add to current batch
            current_batch.append(text)
            current_max_tokens = new_max_tokens

    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)

    return batches


class VRAMPredictor:
    """VRAM prediction utility with cached model configuration.

    Provides convenient methods for predicting memory usage and calculating
    optimal batch sizes. Caches model architecture parameters for efficiency.

    Attributes:
        model_name: Name of the model for architecture lookup.
        output_dim: Output embedding dimension.
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32).
        device: Device string for memory queries.

    Examples:
        >>> predictor = VRAMPredictor()
        >>> available = predictor.get_available_memory()
        >>> batch_size = predictor.optimal_batch_size(token_counts, available)
        >>> batches = predictor.split_texts(texts)
    """

    __slots__ = (
        "model_name",
        "output_dim",
        "dtype_bytes",
        "device",
        "flash_attention",
        "_model_config",
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        output_dim: int = 1024,
        dtype_bytes: int = 2,
        device: str = "auto",
        flash_attention: Optional[bool] = None,
    ):
        """Initialize VRAM predictor.

        Args:
            model_name: Model name from SUPPORTED_MODELS.
            output_dim: Output embedding dimension.
            dtype_bytes: Bytes per element. Default 2 (fp16/bf16).
            device: Device for memory queries. Default "auto".
            flash_attention: Override Flash Attention detection. None=auto, True=force, False=disable.
        """
        self.model_name = model_name
        self.output_dim = output_dim
        self.dtype_bytes = dtype_bytes
        self.device = device
        self.flash_attention = flash_attention

        # Cache model config for repeated predictions
        self._model_config = SUPPORTED_MODELS.get(
            model_name, SUPPORTED_MODELS.get(DEFAULT_MODEL, {})
        )

    def get_available_memory(self) -> int:
        """Get available device memory in bytes."""
        return get_available_device_memory(self.device)

    def predict_batch_memory(self, batch_size: int, max_tokens: int) -> int:
        """Predict VRAM for a batch.

        Args:
            batch_size: Number of texts.
            max_tokens: Maximum sequence length.

        Returns:
            Estimated VRAM in bytes.
        """
        return predict_vram_bytes(
            batch_size=batch_size,
            max_tokens=max_tokens,
            output_dim=self.output_dim,
            dtype_bytes=self.dtype_bytes,
            model_name=self.model_name,
            flash_attention=self.flash_attention,
        )

    def optimal_batch_size(
        self,
        token_counts: List[int],
        available_memory: Optional[int] = None,
        target_utilization: float = 0.9,
    ) -> int:
        """Calculate optimal batch size.

        Args:
            token_counts: Token counts for each text.
            available_memory: Available memory in bytes. Auto-detected if None.
            target_utilization: Target memory utilization. Default 0.9.

        Returns:
            Optimal batch size.
        """
        if available_memory is None:
            available_memory = self.get_available_memory()

        return calculate_optimal_batch_size(
            token_counts=token_counts,
            available_memory_bytes=available_memory,
            target_utilization=target_utilization,
            output_dim=self.output_dim,
            dtype_bytes=self.dtype_bytes,
            model_name=self.model_name,
            flash_attention=self.flash_attention,
        )

    def split_texts(
        self,
        texts: List[str],
        available_memory: Optional[int] = None,
        target_utilization: float = 0.9,
        count_tokens_fn: Optional[Callable[[str], int]] = None,
    ) -> List[List[str]]:
        """Split texts into optimal batches.

        Args:
            texts: Texts to batch.
            available_memory: Available memory in bytes. Auto-detected if None.
            target_utilization: Target memory utilization. Default 0.9.
            count_tokens_fn: Token counting function. Uses estimation if None.

        Returns:
            List of text batches.
        """
        if available_memory is None:
            available_memory = self.get_available_memory()

        return split_into_optimal_batches(
            texts=texts,
            available_memory_bytes=available_memory,
            target_utilization=target_utilization,
            output_dim=self.output_dim,
            dtype_bytes=self.dtype_bytes,
            model_name=self.model_name,
            count_tokens_fn=count_tokens_fn,
            flash_attention=self.flash_attention,
        )

    def model_memory(self) -> int:
        """Get estimated model weight memory."""
        return estimate_model_memory(self.model_name, self.dtype_bytes)

    def memory_report(self, batch_size: int, max_tokens: int) -> Dict[str, Any]:
        """Generate detailed memory report for a batch configuration.

        Args:
            batch_size: Number of texts.
            max_tokens: Maximum sequence length.

        Returns:
            Dictionary with memory breakdown.
        """
        available = self.get_available_memory()
        batch_mem = self.predict_batch_memory(batch_size, max_tokens)
        model_mem = self.model_memory()

        # Cap utilization at 1000% to prevent absurd values in edge cases
        if available > 0:
            utilization = min((batch_mem + model_mem) / available, 10.0)
        else:
            utilization = 0.0

        return {
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "device": self.device,
            "model_name": self.model_name,
            "dtype_bytes": self.dtype_bytes,
            "output_dim": self.output_dim,
            "available_memory_mb": available / 1024**2,
            "batch_memory_mb": batch_mem / 1024**2,
            "model_memory_mb": model_mem / 1024**2,
            "total_predicted_mb": (batch_mem + model_mem) / 1024**2,
            "utilization_percent": utilization * 100,
        }


# ============================================================================
# Inference Queue with Bin-Packing
# ============================================================================

class InferenceQueue:
    """Queue that batches variable-length texts optimally for inference.

    Uses bin-packing to minimize inference calls by densely packing
    sequences of different lengths into batches that fit in available VRAM.

    The key insight is that memory usage is O(batch_size * max_seq_len^2) due
    to attention padding. Grouping similar-length sequences is more efficient
    than arbitrary batching.

    Uses First-Fit Decreasing (FFD) algorithm: sort by token count descending,
    greedily add items that fit. This groups long sequences together and short
    sequences together, reducing padding waste.

    Usage:
        queue = InferenceQueue(embedder, target_utilization=0.9)

        # Add texts (queued until batch is optimal)
        for unit in code_units:
            result = queue.add(unit.embedding_text, unit.id)
            if result:
                # Batch was processed, results available
                handle_results(result)

        # Process remaining texts
        final_results = queue.flush()

    Attributes:
        target_utilization: Target memory utilization (0.0-1.0).
        min_batch_size: Minimum texts before processing.
        stats: Dictionary with processing statistics.
    """

    __slots__ = (
        "_embedder",
        "_target_utilization",
        "_min_batch_size",
        "_queue",
        "_predictor",
        "_available_bytes",
        "_target_bytes",
        "_tokenizer",
        "_stats",
        "_lock",
        "_dimension",  # BUG IQ-8 FIX: MRL dimension support
    )

    def __init__(
        self,
        embedder: "TEIEmbedder | SentenceTransformersEmbedder",
        target_utilization: float = 0.9,
        min_batch_size: int = 1,
        tokenizer: Optional[Any] = None,
        subtract_tei_baseline: bool = True,
        dimension: Optional[int] = None,
    ) -> None:
        """Initialize inference queue.

        Args:
            embedder: TEIEmbedder or SentenceTransformersEmbedder instance.
            target_utilization: Target memory utilization (0.0-1.0). Default 0.9.
                               Higher values pack batches more densely but risk OOM.
            min_batch_size: Minimum texts before processing a batch. Default 1.
            tokenizer: Optional tokenizer for token counting. If None, auto-detects
                      from embedder (TEIEmbedder has count_tokens method).
            subtract_tei_baseline: If True, subtract TEI baseline memory from available.
                                  Set to True when using TEI backend. Default True.
            dimension: Target dimension for MRL (Matryoshka Representation Learning)
                      truncation. If specified, embeddings will be truncated to this
                      dimension. Must be <= model's native dimension. Default None
                      (use model's full dimension).

        Raises:
            ValueError: If target_utilization not in (0, 1].
        """
        if not 0 < target_utilization <= 1.0:
            raise ValueError(f"target_utilization must be in (0, 1], got {target_utilization}")

        self._embedder = embedder
        self._dimension = dimension
        self._target_utilization = target_utilization
        self._min_batch_size = max(1, min_batch_size)
        self._queue: List[Tuple[str, Any, int]] = []  # (text, id, token_count)
        self._predictor = VRAMPredictor()
        self._tokenizer = tokenizer

        # Calculate available memory minus TEI baseline
        raw_available = self._predictor.get_available_memory()
        if subtract_tei_baseline:
            self._available_bytes = max(0, raw_available - TEI_BASELINE_BYTES)
        else:
            self._available_bytes = raw_available

        # BUG EO-1 FIX: Ensure minimum batch memory to prevent single-item batches
        # When available GPU memory < TEI baseline, target_bytes would be 0,
        # causing every text to exceed budget and resulting in 1000x slower batching
        self._target_bytes = max(
            MIN_BATCH_MEMORY_BYTES,
            int(self._available_bytes * target_utilization)
        )

        # Statistics for monitoring
        self._stats = {
            "batches_processed": 0,
            "texts_processed": 0,
            "total_tokens": 0,
            "avg_batch_size": 0.0,
            "avg_utilization": 0.0,
        }

        # Thread safety: protects _queue and _stats from concurrent access
        # BUG IQ-5 FIX: Add lock for thread-safe queue operations
        self._lock = threading.Lock()

    @property
    def target_utilization(self) -> float:
        """Target memory utilization (0.0-1.0)."""
        return self._target_utilization

    @property
    def min_batch_size(self) -> int:
        """Minimum texts before processing."""
        return self._min_batch_size

    @property
    def stats(self) -> Dict[str, Any]:
        """Processing statistics (thread-safe copy)."""
        with self._lock:
            return self._stats.copy()

    @property
    def queue_size(self) -> int:
        """Number of texts currently queued (thread-safe)."""
        with self._lock:
            return len(self._queue)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using available tokenizer.

        Priority:
        1. Custom tokenizer if provided
        2. TEIEmbedder.count_tokens if embedder is TEI
        3. Character estimation (~4 chars per token)

        BUG IQ-3 FIX: Tokenizer failures now fall back gracefully instead of propagating.
        """
        # Try custom tokenizer first
        if self._tokenizer is not None:
            try:
                if callable(self._tokenizer):
                    return self._tokenizer(text)
                elif hasattr(self._tokenizer, "encode"):
                    return len(self._tokenizer.encode(text))
            except Exception:
                pass  # Fall through to next method

        # Try embedder's count_tokens method
        if self._embedder is not None and hasattr(self._embedder, "count_tokens"):
            try:
                return self._embedder.count_tokens(text)
            except Exception:
                pass  # Fall through to character estimation

        # Fallback: character-type aware estimation (handles CJK, emoji, etc.)
        return estimate_tokens_fallback(text)

    def add(self, text: str, id: Any = None) -> Optional[Dict[Any, "np.ndarray"]]:
        """Add text to queue. Returns results if batch was processed.

        Thread-safe: Multiple threads can call add() concurrently.

        Args:
            text: Text to add for embedding.
            id: Optional identifier for the text. If None, uses queue index.

        Returns:
            Dictionary mapping ids to embeddings if batch was processed,
            None if text was queued for later processing.

        Raises:
            ValueError: If text is None.
            TypeError: If text is not a string.
        """
        # BUG IQ-2 FIX: Validate text input to prevent TypeError in _count_tokens
        if text is None:
            raise ValueError("text cannot be None")
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        # Token counting outside lock (no shared state access)
        token_count = self._count_tokens(text)

        # BUG IQ-5 FIX: Protect queue operations with lock
        with self._lock:
            effective_id = id if id is not None else len(self._queue)
            self._queue.append((text, effective_id, token_count))

            # Check if we should process a batch (caller holds lock)
            if self._should_process():
                return self._process_optimal_batch()
        return None

    def add_batch(
        self,
        texts: List[str],
        ids: Optional[List[Any]] = None,
    ) -> Dict[Any, "np.ndarray"]:
        """Add multiple texts, process optimal batches, return all results.

        Args:
            texts: List of texts to add.
            ids: Optional list of identifiers (must match texts length if provided).

        Returns:
            Dictionary mapping ids to embeddings for all processed batches.
            Note: Some texts may still be queued - call flush() to process remaining.

        Raises:
            ValueError: If ids provided but length doesn't match texts.
        """
        if ids is not None and len(ids) != len(texts):
            raise ValueError(f"ids length {len(ids)} != texts length {len(texts)}")

        results: Dict[Any, "np.ndarray"] = {}
        for i, text in enumerate(texts):
            effective_id = ids[i] if ids is not None else i
            batch_results = self.add(text, effective_id)
            if batch_results:
                results.update(batch_results)
        return results

    def flush(self) -> Dict[Any, "np.ndarray"]:
        """Process all remaining texts in queue.

        Thread-safe: Can be called concurrently with add().
        Note: add() calls will block until flush() completes.

        Returns:
            Dictionary mapping ids to embeddings for all remaining texts.
        """
        import numpy as np

        results: Dict[Any, np.ndarray] = {}
        # BUG IQ-5 FIX: Hold lock during entire flush to prevent races
        with self._lock:
            while self._queue:
                batch_results = self._process_optimal_batch()
                results.update(batch_results)
        return results

    def _should_process(self) -> bool:
        """Check if current queue can form an optimal batch.

        Note: Caller must hold self._lock.

        Returns True when:
        1. Queue has at least min_batch_size texts, AND
        2. Predicted memory for queue >= 95% of target (near-optimal)
        """
        if len(self._queue) < self._min_batch_size:
            return False

        # Calculate memory for current queue
        max_tokens = max(tc for _, _, tc in self._queue)
        batch_size = len(self._queue)
        predicted = self._predictor.predict_batch_memory(batch_size, max_tokens)

        # Process if we're at or above 95% of target utilization
        return predicted >= self._target_bytes * 0.95

    def _process_optimal_batch(self) -> Dict[Any, "np.ndarray"]:
        """Extract and process optimal batch from queue using bin-packing.

        Uses First-Fit Decreasing (FFD) algorithm to select texts that
        maximize memory utilization while staying within budget.

        Note: Caller must hold self._lock.

        Returns:
            Dictionary mapping ids to embeddings for the processed batch.

        Note:
            Queue is only modified AFTER successful encoding to prevent
            data loss if encoder fails (OOM, network error, etc.).
        """
        import numpy as np

        if not self._queue:
            return {}

        # Bin-packing: select texts that maximize utilization
        # Note: Don't modify _queue yet - items must stay safe until encode succeeds
        # BUG IQ-6 FIX: _pack_optimal_batch now returns batch_tokens to avoid redundant counting
        batch_texts, batch_ids, batch_tokens, remaining = self._pack_optimal_batch()

        if not batch_texts:
            return {}

        # Run inference - if this fails, items remain in _queue for retry
        # BUG IQ-8 FIX: Pass dimension for MRL truncation support
        embeddings = self._embedder.encode(batch_texts, dimension=self._dimension)

        # BUG IQ-4 FIX: Validate embedder output to prevent silent data loss from zip()
        if len(embeddings) != len(batch_texts):
            raise RuntimeError(
                f"Embedder returned {len(embeddings)} embeddings for "
                f"{len(batch_texts)} texts. This indicates a bug in the embedder."
            )

        # Only modify queue AFTER successful encoding (prevents data loss on failure)
        self._queue = remaining

        # Update statistics using pre-computed token counts (BUG IQ-6 FIX)
        batch_size = len(batch_texts)
        total_tokens = sum(batch_tokens)
        max_tokens = max(batch_tokens) if batch_tokens else 0
        predicted_mem = self._predictor.predict_batch_memory(batch_size, max_tokens)
        utilization = predicted_mem / self._available_bytes if self._available_bytes > 0 else 0

        self._stats["batches_processed"] += 1
        self._stats["texts_processed"] += batch_size
        self._stats["total_tokens"] += total_tokens
        n = self._stats["batches_processed"]
        self._stats["avg_batch_size"] = (
            (self._stats["avg_batch_size"] * (n - 1) + batch_size) / n
        )
        self._stats["avg_utilization"] = (
            (self._stats["avg_utilization"] * (n - 1) + utilization) / n
        )

        return {id_: emb for id_, emb in zip(batch_ids, embeddings)}

    def _pack_optimal_batch(
        self,
    ) -> Tuple[List[str], List[Any], List[int], List[Tuple[str, Any, int]]]:
        """Bin-packing algorithm to select optimal batch.

        Strategy: First-Fit Decreasing (FFD) - sort by token count descending,
        greedily add items that fit. This groups similar-length sequences
        together, reducing padding waste from attention matrices.

        Note: Caller must hold self._lock.

        Returns:
            Tuple of (batch_texts, batch_ids, batch_tokens, remaining_queue).
            batch_tokens contains pre-computed token counts to avoid redundant counting.
        """
        if not self._queue:
            return [], [], [], []

        # Sort by token count descending (FFD heuristic)
        # Items with most tokens first - they define the batch memory ceiling
        sorted_queue = sorted(self._queue, key=lambda x: x[2], reverse=True)

        batch_texts: List[str] = []
        batch_ids: List[Any] = []
        batch_tokens: List[int] = []
        remaining: List[Tuple[str, Any, int]] = []

        for text, id_, tokens in sorted_queue:
            if batch_texts:
                # Max tokens in batch determines memory (due to padding)
                new_max_tokens = max(max(batch_tokens), tokens)
                new_batch_size = len(batch_texts) + 1
                predicted = self._predictor.predict_batch_memory(
                    new_batch_size, new_max_tokens
                )

                if predicted > self._target_bytes:
                    remaining.append((text, id_, tokens))
                    continue

            batch_texts.append(text)
            batch_ids.append(id_)
            batch_tokens.append(tokens)

        # Note: The first item is always added unconditionally (when batch_texts is empty,
        # the budget check is skipped). This means batch_texts can never be empty after
        # the loop if queue was non-empty, so no "safety net" fallback is needed.

        return batch_texts, batch_ids, batch_tokens, remaining

    def clear(self) -> None:
        """Clear the queue without processing (thread-safe)."""
        with self._lock:
            self._queue.clear()

    def memory_report(self) -> Dict[str, Any]:
        """Get current memory status and queue information (thread-safe).

        Returns:
            Dictionary with memory breakdown and queue status.
        """
        # BUG IQ-5 FIX: Protect queue reads with lock
        with self._lock:
            current_tokens = [tc for _, _, tc in self._queue]
            max_tokens = max(current_tokens) if current_tokens else 0
            queue_size = len(self._queue)
            predicted = self._predictor.predict_batch_memory(queue_size, max_tokens)
            would_process = self._should_process()
            stats_copy = self._stats.copy()

        return {
            "queue_size": queue_size,
            "queue_tokens": sum(current_tokens),
            "max_tokens_in_queue": max_tokens,
            "available_memory_mb": self._available_bytes / 1024**2,
            "target_memory_mb": self._target_bytes / 1024**2,
            "predicted_batch_mb": predicted / 1024**2,
            "tei_baseline_mb": TEI_BASELINE_MB,
            "target_utilization": self._target_utilization,
            "would_process": would_process,
            "stats": stats_copy,
        }


def create_inference_queue(
    target_utilization: float = 0.9,
    backend: str = "auto",
    min_batch_size: int = 1,
) -> InferenceQueue:
    """Create an optimally-batching inference queue.

    Convenience function that initializes the queue with the appropriate
    embedder backend.

    Args:
        target_utilization: Target memory utilization (0.0-1.0). Default 0.9.
        backend: Inference backend - "tei", "sentence_transformers", or "auto".
                 "auto" prefers TEI if available.
        min_batch_size: Minimum texts before processing. Default 1.

    Returns:
        InferenceQueue configured with the selected embedder.

    Example:
        >>> queue = create_inference_queue(target_utilization=0.85)
        >>> for text, id in code_units:
        ...     result = queue.add(text, id)
        ...     if result:
        ...         save_embeddings(result)
        >>> final = queue.flush()
        >>> save_embeddings(final)
        >>> print(queue.stats)
    """
    mm = get_model_manager()
    embedder = mm.get_embedder(backend=backend)

    # Determine if TEI baseline should be subtracted
    # TEIEmbedder is defined later in file, use string check for safety
    subtract_tei = type(embedder).__name__ == "TEIEmbedder"

    return InferenceQueue(
        embedder=embedder,
        target_utilization=target_utilization,
        min_batch_size=min_batch_size,
        subtract_tei_baseline=subtract_tei,
    )


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

    def get_embedder(
        self,
        backend: str = "auto",
    ) -> "TEIEmbedder | SentenceTransformersEmbedder":
        """Get the embedder instance, loading if necessary.

        Args:
            backend: Inference backend - "tei", "sentence_transformers", or "auto".

        Returns:
            The embedder instance (TEIEmbedder or SentenceTransformersEmbedder).

        Raises:
            RuntimeError: If embedder fails to load.
        """
        with self._model_lock:
            if self._embedder is None:
                self.load(backend=backend)
            if self._embedder is None:
                raise RuntimeError("Failed to load embedder")
            return self._embedder

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

            metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
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


def get_vram_predictor(
    model_name: str = DEFAULT_MODEL,
    output_dim: int = 1024,
    dtype_bytes: int = 2,
    device: str = "auto",
) -> VRAMPredictor:
    """Get a VRAM predictor for smart batching.

    Args:
        model_name: Model name from SUPPORTED_MODELS.
        output_dim: Output embedding dimension.
        dtype_bytes: Bytes per element (2 for fp16/bf16, 4 for fp32).
        device: Device for memory queries.

    Returns:
        VRAMPredictor instance.
    """
    return VRAMPredictor(
        model_name=model_name,
        output_dim=output_dim,
        dtype_bytes=dtype_bytes,
        device=device,
    )


def encode_batch(
    texts: List[str],
    instruction: Optional[str] = None,
    show_progress: bool = False,
) -> "np.ndarray":
    """Encode texts to embeddings."""
    return get_model_manager().encode(texts, instruction=instruction, show_progress=show_progress)


def encode_optimally(
    texts: List[str],
    ids: Optional[List[Any]] = None,
    target_utilization: float = 0.9,
    show_progress: bool = True,
    dimension: Optional[int] = None,
) -> Dict[Any, "np.ndarray"]:
    """Encode texts with optimal batching for maximum throughput.

    High-level API that handles all batching internally using bin-packing
    to minimize inference calls while maximizing GPU utilization.

    Just drop all your texts in - the function figures out the most
    efficient way to batch them based on available VRAM.

    Args:
        texts: List of texts to encode.
        ids: Optional IDs for each text. If None, uses indices 0, 1, 2...
        target_utilization: Target VRAM utilization (0.0-1.0). Default 0.9.
        show_progress: Show progress bar. Default True.
        dimension: Target dimension for MRL truncation. Must be <= model dimension.
                   If None, uses the model's full native dimension.

    Returns:
        Dictionary mapping id -> embedding (numpy array).

    Note:
        This function is NOT thread-safe. Do not call from multiple
        threads simultaneously. For thread-safe encoding, create
        separate ModelManager instances per thread.

    Example:
        >>> # Encode code units optimally
        >>> texts = [unit.embedding_text for unit in units]
        >>> ids = [unit.qualified_name for unit in units]
        >>> embeddings = encode_optimally(texts, ids)
        >>>
        >>> # Or just use indices
        >>> embeddings = encode_optimally(texts)
        >>> first_embedding = embeddings[0]
    """
    import numpy as np

    if not texts:
        return {}

    # EO-3 FIX: Validate dimension upfront for consistent behavior across backends
    # TEIEmbedder silently converts invalid dimensions to None, while
    # SentenceTransformersEmbedder raises ValueError. This ensures consistent error.
    if dimension is not None:
        if not isinstance(dimension, int):
            raise TypeError(f"dimension must be int, got {type(dimension).__name__}")
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

    if ids is None:
        ids = list(range(len(texts)))

    if len(ids) != len(texts):
        raise ValueError(f"ids length ({len(ids)}) must match texts length ({len(texts)})")

    # BUG EO-8 FIX: Warn on duplicate IDs to prevent silent data loss
    # Later embeddings overwrite earlier ones, which may not be user's intent
    if len(set(ids)) != len(ids):
        import logging
        _logger = logging.getLogger(__name__)
        seen: set = set()
        duplicates = [id_ for id_ in ids if id_ in seen or seen.add(id_)]  # type: ignore[func-returns-value]
        _logger.warning(
            f"Duplicate IDs detected: {duplicates[:5]!r}"
            f"{'...' if len(duplicates) > 5 else ''}. "
            f"Later embeddings will overwrite earlier ones."
        )

    # Get embedder and create queue
    manager = get_model_manager()
    embedder = manager.get_embedder()

    # Detect backend type for batching strategy
    # TEI uses token-budget batching (handled server-side), not VRAM-based
    is_tei_backend = type(embedder).__name__ == "TEIEmbedder"

    # TEI default max_batch_tokens - match server config for optimal throughput
    # TEI handles memory management internally via Rust/candle with flash attention
    # Default 65536 works well for 6GB+ GPUs; override via TEI_MAX_BATCH_TOKENS env var
    TEI_MAX_BATCH_TOKENS = int(os.environ.get("TEI_MAX_BATCH_TOKENS", "65536"))

    queue = InferenceQueue(embedder, target_utilization=target_utilization)

    # Count tokens for all texts upfront
    token_counts = []
    for text in texts:
        tc = queue._count_tokens(text)
        token_counts.append(tc)

    # Sort by token count for optimal bin-packing (group similar lengths)
    indexed = list(zip(range(len(texts)), texts, ids, token_counts))
    indexed.sort(key=lambda x: x[3], reverse=True)  # Longest first

    # Plan optimal batches using bin-packing
    batches = []  # List of (batch_texts, batch_ids)
    current_batch_texts = []
    current_batch_ids = []
    current_batch_tokens = 0  # Sum of tokens in current batch (for TEI)
    current_max_tokens = 0    # Max tokens in current batch (for VRAM prediction)

    if is_tei_backend:
        # TEI backend: Use token-budget batching (matching TEI's internal logic)
        # TEI handles memory via max_batch_tokens, not VRAM calculations
        for _, text, id_, tokens in indexed:
            if current_batch_texts:
                # Check if adding this would exceed TEI token budget
                new_total = current_batch_tokens + tokens
                if new_total > TEI_MAX_BATCH_TOKENS:
                    # Finalize current batch, start new one
                    batches.append((current_batch_texts, current_batch_ids))
                    current_batch_texts = [text]
                    current_batch_ids = [id_]
                    current_batch_tokens = tokens
                    continue

            current_batch_texts.append(text)
            current_batch_ids.append(id_)
            current_batch_tokens += tokens
    else:
        # SentenceTransformers backend: Use VRAM-based predictions (PyTorch)
        # BUG EO-7 FIX: Use correct output dimension for memory prediction
        if dimension is not None:
            predictor = VRAMPredictor(output_dim=dimension)
        else:
            predictor = queue._predictor
        target_bytes = queue._target_bytes

        for _, text, id_, tokens in indexed:  # BUG EO-6 FIX: use _ for unused index
            if current_batch_texts:
                # Check if adding this would exceed target
                new_max = max(current_max_tokens, tokens)
                new_size = len(current_batch_texts) + 1
                predicted = predictor.predict_batch_memory(new_size, new_max)

                if predicted > target_bytes:
                    # Finalize current batch, start new one
                    batches.append((current_batch_texts, current_batch_ids))
                    current_batch_texts = [text]
                    current_batch_ids = [id_]
                    current_max_tokens = tokens

                    # BUG EO-2 FIX: Warn if single text exceeds memory budget
                    single_mem = predictor.predict_batch_memory(1, tokens)
                    if single_mem > target_bytes:
                        import logging
                        _logger = logging.getLogger(__name__)
                        _logger.warning(
                            f"Single text ({tokens} tokens) requires "
                            f"{single_mem / 1024**2:.0f}MB but budget is "
                            f"{target_bytes / 1024**2:.0f}MB. "
                            f"May cause OOM. Consider chunking this text or "
                            f"using a smaller model."
                        )
                    continue

            current_batch_texts.append(text)
            current_batch_ids.append(id_)
            current_max_tokens = max(current_max_tokens, tokens)

            # BUG EO-2 FIX: Also check first text in batch (when batch was empty)
            if len(current_batch_texts) == 1:
                single_mem = predictor.predict_batch_memory(1, tokens)
                if single_mem > target_bytes:
                    import logging
                    _logger = logging.getLogger(__name__)
                    _logger.warning(
                        f"Single text ({tokens} tokens) requires "
                        f"{single_mem / 1024**2:.0f}MB but budget is "
                        f"{target_bytes / 1024**2:.0f}MB. "
                        f"May cause OOM. Consider chunking this text or "
                        f"using a smaller model."
                    )

    # Don't forget the last batch
    if current_batch_texts:
        batches.append((current_batch_texts, current_batch_ids))

    # Process all batches
    results: Dict[Any, np.ndarray] = {}

    if show_progress:
        try:
            from tqdm import tqdm
            batch_iter = tqdm(batches, desc="Encoding", unit="batch")
        except ImportError:
            batch_iter = batches
    else:
        batch_iter = batches

    total_batches = 0
    for batch_texts, batch_ids in batch_iter:
        embeddings = embedder.encode(batch_texts, dimension=dimension)
        for id_, emb in zip(batch_ids, embeddings):
            results[id_] = emb
        total_batches += 1

    # Log batch statistics for observability
    if total_batches > 0:
        import logging
        logger = logging.getLogger(__name__)
        avg_batch = len(texts) / total_batches
        logger.info(
            f"Encoded {len(texts)} texts in {total_batches} batches "
            f"(avg {avg_batch:.1f} texts/batch)"
        )

    return results


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

    # BI-2 FIX: Validate MRL truncation before expensive embedding operation
    # Non-Matryoshka models produce semantically meaningless vectors when truncated
    if dimension is not None:
        model_info = SUPPORTED_MODELS.get(model_name, {})
        native_dim = model_info.get("dimension", 1024)
        is_matryoshka = model_info.get("is_matryoshka", False)

        if dimension < native_dim and not is_matryoshka:
            raise ValueError(
                f"Model {model_name} does not support MRL truncation "
                f"(is_matryoshka=False). Cannot truncate {native_dim}D to {dimension}D. "
                f"Use a Matryoshka-trained model (e.g., Qwen/Qwen3-Embedding-0.6B) "
                f"or remove the dimension parameter."
            )

        valid_dims = model_info.get("matryoshka_dims", [])
        if valid_dims and dimension not in valid_dims:
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                f"Dimension {dimension} not in recommended MRL dimensions {valid_dims}. "
                f"Results may be suboptimal."
            )

    units = extract_units_from_project(str(project), lang=lang, respect_ignore=respect_ignore)
    if not units:
        return 0

    texts = [build_embedding_text(u) for u in units]
    n = len(texts)

    # Load model first (required for encode_optimally to use correct backend)
    mm = get_model_manager()
    mm.load(model_name, backend=backend)

    # Use optimal batching for maximum throughput
    # encode_optimally handles bin-packing to minimize inference calls
    embeddings_dict = encode_optimally(
        texts,
        ids=None,  # Use indices 0, 1, 2, ...
        show_progress=show_progress,
        dimension=dimension,
    )
    # Convert dict to array in order (indices 0 to n-1)
    embeddings = np.stack([embeddings_dict[i] for i in range(n)])

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

    # BI-3 FIX: Use proper file locking to prevent concurrent builds
    # Advisory lock files (.building) don't prevent race conditions - multiple
    # processes can touch() and proceed concurrently, corrupting the index.
    # fcntl.flock() provides actual OS-level exclusive locking.
    lock_file = cache_dir / ".build_lock"
    lock_fd = open(lock_file, "w")

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fd.close()
        raise RuntimeError(
            f"Another process is building the index at {cache_dir}. "
            "Wait for it to complete or remove .build_lock if stale."
        )

    # BI-1 FIX: Use staging directory for atomic two-file save
    # Problem: Sequential moves of index + metadata are non-atomic. If first move
    # succeeds but second fails, we have new embeddings with old/missing metadata.
    # Solution: Stage both files to temp directory, then backup-and-swap atomically.
    unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    temp_dir = cache_dir / f".build_{unique_id}"
    backup_dir = cache_dir / ".backup"

    try:
        # Stage: Write both files to temp directory first
        # This ensures both files are valid before any swap operation
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_index = temp_dir / "index.usearch"
        temp_metadata = temp_dir / "metadata.json"

        index.save(str(temp_index))
        temp_metadata.write_text(json.dumps(metadata, indent=2))

        # Swap: Backup existing files, then move new files into place
        # Clean any stale backup from previous failed attempt
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(exist_ok=True)

        # Move current files to backup (if they exist)
        if index_file.exists():
            shutil.move(str(index_file), str(backup_dir / "index.usearch"))
        if metadata_file.exists():
            shutil.move(str(metadata_file), str(backup_dir / "metadata.json"))

        # Move new files into place
        # If this fails mid-way, the backup contains valid consistent state
        shutil.move(str(temp_index), str(index_file))
        shutil.move(str(temp_metadata), str(metadata_file))

        # Success - clean up backup
        shutil.rmtree(backup_dir, ignore_errors=True)

    except Exception:
        # Rollback: Restore from backup if swap failed
        # This handles the case where new index was moved but metadata move failed
        if backup_dir.exists():
            # Remove any partially-moved new files
            if index_file.exists():
                try:
                    index_file.unlink()
                except OSError:
                    pass
            if metadata_file.exists():
                try:
                    metadata_file.unlink()
                except OSError:
                    pass
            # Restore from backup
            backup_index = backup_dir / "index.usearch"
            backup_metadata = backup_dir / "metadata.json"
            if backup_index.exists():
                try:
                    shutil.move(str(backup_index), str(index_file))
                except OSError:
                    pass
            if backup_metadata.exists():
                try:
                    shutil.move(str(backup_metadata), str(metadata_file))
                except OSError:
                    pass
        raise

    finally:
        # Always clean up temp directory and release lock
        shutil.rmtree(temp_dir, ignore_errors=True)
        # BI-3 FIX: Release fcntl lock - allows other processes to proceed
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        except OSError:
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
