"""Persistent, incremental Qwen text-embedding cache.

The cache stores resource-document embeddings, not query embeddings.  A
document is addressed by a SHA-256 fingerprint, so adding a tool or changing
its description only computes the new/changed document.  The cache is shared
by A1 and the MCP search gateways through ``BIOMNI_EMBEDDING_CACHE_PATH``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_EMBEDDING_CACHE_FILENAME = "qwen3_resource_embeddings.pt"


def default_cache_path() -> Path:
    """Return the repository-local cache path unless explicitly overridden."""
    configured = os.getenv("BIOMNI_EMBEDDING_CACHE_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / DEFAULT_EMBEDDING_CACHE_FILENAME


class QwenEmbeddingCache:
    """Compute and persist Qwen embedding vectors with incremental updates."""

    CACHE_VERSION = 2

    def __init__(
        self,
        *,
        cache_path: str | os.PathLike[str] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser() if cache_path else default_cache_path()
        self.model = model or os.getenv("BIOMNI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.base_url = (base_url or os.getenv("BIOMNI_EMBEDDING_BASE_URL") or DEFAULT_EMBEDDING_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        self.batch_size = max(1, int(batch_size or os.getenv("BIOMNI_EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE)))
        self.timeout_seconds = float(timeout_seconds or os.getenv("BIOMNI_EMBEDDING_TIMEOUT_SECONDS", "60"))
        self.last_stats: dict[str, Any] = {"hits": 0, "misses": 0, "api_calls": 0, "api_tokens": 0}

    @staticmethod
    def document_key(document: str) -> str:
        return hashlib.sha256(document.encode("utf-8")).hexdigest()

    def _read_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        credential_path = Path("qwen_apikey.txt")
        if credential_path.is_file():
            for raw_line in credential_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" not in line and not line.startswith(("http://", "https://")):
                    return line
                if line.startswith(("DASHSCOPE_API_KEY=", "QWEN_API_KEY=", "API_KEY=")):
                    return line.split("=", 1)[1].strip()
        raise ValueError("Qwen embedding API key not found; set DASHSCOPE_API_KEY or QWEN_API_KEY")

    @contextmanager
    def _locked(self):
        """Serialize cache read/compute/write across A1 and MCP workers."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_path.with_name(self.cache_path.name + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass

    def _load(self) -> dict[str, Any]:
        if not self.cache_path.is_file():
            return {"metadata": {}, "entries": {}}
        try:
            import torch

            try:
                payload = torch.load(self.cache_path, map_location="cpu", weights_only=False)
            except TypeError:  # torch<2.6 does not expose weights_only
                payload = torch.load(self.cache_path, map_location="cpu")
        except Exception:
            return {"metadata": {}, "entries": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
            return {"metadata": {}, "entries": {}}
        metadata = payload.get("metadata") or {}
        # Older local Qwen3-Embedding-4B caches use category-level matrices and
        # 2560 dimensions. They are deliberately not mixed with API vectors.
        if metadata.get("version") != self.CACHE_VERSION or metadata.get("model") != self.model:
            return {"metadata": {}, "entries": {}}
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        import torch

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=self.cache_path.name + ".", dir=self.cache_path.parent)
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(self.cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _embed_api(self, texts: list[str]) -> tuple[list[list[float]], int]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._read_api_key()}", "Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            detail = response.text[:500]
            raise RuntimeError(f"Qwen embedding API request failed: {detail}") from exc
        data = sorted(body.get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in data]
        if len(vectors) != len(texts) or any(not isinstance(vector, list) or not vector for vector in vectors):
            raise RuntimeError("Qwen embedding API returned an invalid vector list")
        usage = body.get("usage") or {}
        return vectors, int(usage.get("total_tokens", usage.get("prompt_tokens", 0)) or 0)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed query text without persisting it as a resource document."""
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch_vectors, _tokens = self._embed_api(texts[start : start + self.batch_size])
            vectors.extend(batch_vectors)
        return vectors

    def ensure_documents(self, documents: list[str]) -> list[list[float]]:
        """Return vectors in input order, computing only cache misses."""
        if not documents:
            self.last_stats = {"hits": 0, "misses": 0, "api_calls": 0, "api_tokens": 0}
            return []
        keys = [self.document_key(document) for document in documents]
        with self._locked():
            payload = self._load()
            entries = payload["entries"]
            missing_indices = [index for index, key in enumerate(keys) if key not in entries]
            api_calls = 0
            api_tokens = 0
            for start in range(0, len(missing_indices), self.batch_size):
                batch_indices = missing_indices[start : start + self.batch_size]
                batch_vectors, tokens = self._embed_api([documents[index] for index in batch_indices])
                api_calls += 1
                api_tokens += tokens
                for index, vector in zip(batch_indices, batch_vectors, strict=True):
                    entries[keys[index]] = {"document": documents[index], "embedding": vector}

            if missing_indices:
                dimensions = len(next(iter(entries.values()))["embedding"])
                payload["metadata"] = {
                    "version": self.CACHE_VERSION,
                    "provider": "Qwen",
                    "model": self.model,
                    "dimension": dimensions,
                    "updated_at": time.time(),
                }
                self._save(payload)

            self.last_stats = {
                "hits": len(documents) - len(missing_indices),
                "misses": len(missing_indices),
                "api_calls": api_calls,
                "api_tokens": api_tokens,
                "cache_path": str(self.cache_path),
            }
            return [entries[key]["embedding"] for key in keys]
