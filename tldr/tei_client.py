"""
Clean Python client for TEI (text-embeddings-inference) gRPC server.

Wraps the generated protobuf stubs with a Pythonic interface.

Usage:
    client = TEIClient("localhost", 18080)

    # Embeddings
    embeddings = client.embed(["text1", "text2"], normalize=True)

    # Tokenization
    tokens = client.tokenize("Hello world")
    token_count = client.count_tokens("Hello world")

    # Server info
    info = client.info()
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Iterator

import functools

import grpc
import numpy as np

from . import tei_pb2
from . import tei_pb2_grpc


class TEIError(Exception):
    """TEI service error with context.

    Wraps gRPC errors with meaningful messages including status code and details.
    """
    pass


def _wrap_grpc_error(func):
    """Decorator to translate gRPC errors into TEIError with context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except grpc.RpcError as e:
            status = e.code()
            details = e.details()
            raise TEIError(f"TEI call failed: {status.name} - {details}") from e
    return wrapper


# Default connection settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 18080
ENV_HOST = "TLDR_TEI_HOST"
ENV_PORT = "TLDR_TEI_PORT"


@dataclass(slots=True, frozen=True)
class ServerInfo:
    """TEI server information."""
    model_id: str
    model_type: str  # "embedding", "classifier", "reranker"
    max_input_length: int
    max_batch_tokens: int
    max_batch_requests: int
    max_client_batch_size: int
    tokenization_workers: int


@dataclass(slots=True, frozen=True)
class Token:
    """A single token from tokenization."""
    id: int
    text: str
    special: bool
    start: Optional[int] = None
    stop: Optional[int] = None


@dataclass(slots=True, frozen=True)
class EmbedMetadata:
    """Metadata from embedding request."""
    compute_chars: int
    compute_tokens: int
    total_time_ns: int
    tokenization_time_ns: int
    queue_time_ns: int
    inference_time_ns: int


class TEIClient:
    """High-level client for TEI gRPC server.

    Provides clean interface for:
    - embed(): Dense embeddings with MRL support
    - embed_sparse(): Sparse embeddings (SPLADE-style)
    - tokenize(): Get tokens with offsets
    - count_tokens(): Fast token counting
    - info(): Server information
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 120.0,
    ):
        """Initialize TEI client.

        Args:
            host: Server host. Defaults to TLDR_TEI_HOST or localhost.
            port: Server port. Defaults to TLDR_TEI_PORT or 18080.
            timeout: Default timeout for requests in seconds.
        """
        if host is None:
            host = os.environ.get(ENV_HOST, DEFAULT_HOST)
        if port is None:
            port = int(os.environ.get(ENV_PORT, str(DEFAULT_PORT)))

        self.host = host
        self.port = port
        self.timeout = timeout
        self._address = f"{host}:{port}"

        # Create channel with optimized settings
        self._channel = grpc.insecure_channel(
            self._address,
            options=[
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ],
        )

        # Create stubs for all services
        self._info_stub = tei_pb2_grpc.InfoStub(self._channel)
        self._embed_stub = tei_pb2_grpc.EmbedStub(self._channel)
        self._tokenize_stub = tei_pb2_grpc.TokenizeStub(self._channel)
        self._rerank_stub = tei_pb2_grpc.RerankStub(self._channel)

        # Thread safety for close operation
        self._close_lock = threading.Lock()
        self._closed = False

    @_wrap_grpc_error
    def info(self) -> ServerInfo:
        """Get server information."""
        resp = self._info_stub.Info(tei_pb2.InfoRequest(), timeout=self.timeout)

        model_types = {0: "embedding", 1: "classifier", 2: "reranker"}

        return ServerInfo(
            model_id=resp.model_id,
            model_type=model_types.get(resp.model_type, "unknown"),
            max_input_length=resp.max_input_length,
            max_batch_tokens=resp.max_batch_tokens,
            max_batch_requests=resp.max_batch_requests,
            max_client_batch_size=resp.max_client_batch_size,
            tokenization_workers=resp.tokenization_workers,
        )

    @_wrap_grpc_error
    def embed(
        self,
        texts: List[str],
        normalize: bool = True,
        truncate: bool = True,
        dimensions: Optional[int] = None,
    ) -> np.ndarray:
        """Embed texts to dense vectors.

        Args:
            texts: List of texts to embed.
            normalize: L2-normalize embeddings (default True).
            truncate: Truncate texts exceeding max length (default True).
            dimensions: Output dimensions for MRL (Matryoshka). None = full.

        Returns:
            np.ndarray of shape (len(texts), dimensions or model_dim)
        """
        if not texts:
            return np.empty((0, dimensions or 1024), dtype=np.float32)

        def gen_requests():
            for text in texts:
                req = tei_pb2.EmbedRequest(
                    inputs=text,
                    truncate=truncate,
                    normalize=normalize,
                    truncation_direction=tei_pb2.TRUNCATION_DIRECTION_RIGHT,
                )
                if dimensions is not None:
                    req.dimensions = dimensions
                yield req

        results = []
        for resp in self._embed_stub.EmbedStream(gen_requests(), timeout=self.timeout):
            results.append(list(resp.embeddings))

        return np.array(results, dtype=np.float32)

    @_wrap_grpc_error
    def embed_single(
        self,
        text: str,
        normalize: bool = True,
        truncate: bool = True,
        dimensions: Optional[int] = None,
    ) -> tuple[np.ndarray, EmbedMetadata]:
        """Embed single text with metadata.

        Returns:
            Tuple of (embedding array, metadata)
        """
        req = tei_pb2.EmbedRequest(
            inputs=text,
            truncate=truncate,
            normalize=normalize,
            truncation_direction=tei_pb2.TRUNCATION_DIRECTION_RIGHT,
        )
        if dimensions is not None:
            req.dimensions = dimensions

        resp = self._embed_stub.Embed(req, timeout=self.timeout)

        meta = EmbedMetadata(
            compute_chars=resp.metadata.compute_chars,
            compute_tokens=resp.metadata.compute_tokens,
            total_time_ns=resp.metadata.total_time_ns,
            tokenization_time_ns=resp.metadata.tokenization_time_ns,
            queue_time_ns=resp.metadata.queue_time_ns,
            inference_time_ns=resp.metadata.inference_time_ns,
        )

        return np.array(resp.embeddings, dtype=np.float32), meta

    @_wrap_grpc_error
    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[Token]:
        """Tokenize text and return tokens with offsets.

        Args:
            text: Text to tokenize.
            add_special_tokens: Add model's special tokens (default True).

        Returns:
            List of Token objects with id, text, special flag, and offsets.
        """
        req = tei_pb2.EncodeRequest(
            inputs=text,
            add_special_tokens=add_special_tokens,
        )
        resp = self._tokenize_stub.Tokenize(req, timeout=self.timeout)

        return [
            Token(
                id=t.id,
                text=t.text,
                special=t.special,
                start=t.start if t.HasField("start") else None,
                stop=t.stop if t.HasField("stop") else None,
            )
            for t in resp.tokens
        ]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (fast path).

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens (excluding special tokens).
        """
        tokens = self.tokenize(text, add_special_tokens=False)
        return len(tokens)

    @_wrap_grpc_error
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts efficiently.

        Args:
            texts: List of texts to count.

        Returns:
            List of token counts.
        """
        def gen_requests():
            for text in texts:
                yield tei_pb2.EncodeRequest(inputs=text, add_special_tokens=False)

        counts = []
        for resp in self._tokenize_stub.TokenizeStream(gen_requests(), timeout=self.timeout):
            counts.append(len(resp.tokens))

        return counts

    @_wrap_grpc_error
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text.
        """
        req = tei_pb2.DecodeRequest(ids=token_ids)
        resp = self._tokenize_stub.Decode(req, timeout=self.timeout)
        return resp.text

    @_wrap_grpc_error
    def rerank(
        self,
        query: str,
        texts: List[str],
        truncate: bool = True,
        return_text: bool = False,
    ) -> List[tuple[int, float, Optional[str]]]:
        """Rerank texts against a query.

        Args:
            query: Query text.
            texts: List of texts to rerank.
            truncate: Truncate texts exceeding max length.
            return_text: Include text in results.

        Returns:
            List of (index, score, text) tuples sorted by score descending.

        Raises:
            ValueError: If query is empty.
            TEIError: If gRPC call fails.
        """
        if not texts:
            return []
        if not query:
            raise ValueError("query cannot be empty")

        req = tei_pb2.RerankRequest(
            query=query,
            texts=texts,
            truncate=truncate,
            return_text=return_text,
        )
        resp = self._rerank_stub.Rerank(req, timeout=self.timeout)

        return [
            (r.index, r.score, r.text if return_text else None)
            for r in resp.ranks
        ]

    def close(self):
        """Close the gRPC channel and release stub references.

        Thread-safe: uses lock to prevent double-close race condition.
        """
        with self._close_lock:
            if self._closed:
                return
            if self._channel is not None:
                self._channel.close()
                self._channel = None
            # Clear stubs to prevent use-after-close errors
            self._info_stub = None
            self._embed_stub = None
            self._tokenize_stub = None
            self._rerank_stub = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def get_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> TEIClient:
    """Get a TEI client instance.

    Args:
        host: Server host. Defaults to TLDR_TEI_HOST or localhost.
        port: Server port. Defaults to TLDR_TEI_PORT or 18080.

    Returns:
        TEIClient instance.
    """
    return TEIClient(host=host, port=port)


def is_available(
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: float = 5.0,
) -> bool:
    """Check if TEI server is available.

    Args:
        host: Server host.
        port: Server port.
        timeout: Connection timeout in seconds. Defaults to 5.0s for health checks.

    Returns:
        True if server is reachable.
    """
    try:
        with TEIClient(host=host, port=port, timeout=timeout) as client:
            client.info()
            return True
    except Exception:
        return False
