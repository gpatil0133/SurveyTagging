"""Embedding model + canon-stage similarity scoring.

The embedding model (default: BAAI/bge-small-en-v1.5, 384-dim) is loaded
lazily on first use and shared across the process via a module-level
singleton. Vectors are L2-normalized at encode time so cosine similarity
reduces to a single dot product.

`CanonEmbeddingIndex` bundles the per-stage vectors with their canon. It is
serialized alongside the canon JSON as a numpy `.npz` so the embedding cost
is paid once per tenant (a few hundred ms for a 14-stage canon on CPU).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

import sharefs
from models.tenant_canon import CanonStage, TenantCanon

logger = logging.getLogger(__name__)


# ---------- Embedding model singleton ----------


class EmbeddingModel:
    """Process-wide cached sentence-transformers model.

    Lazy import keeps `sentence_transformers` an optional install — only the
    first call to `encode()` actually pulls the dependency.
    """

    _instance: "EmbeddingModel | None" = None
    _lock = threading.Lock()

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None  # populated on first encode()

    @classmethod
    def get(cls, model_name: str) -> "EmbeddingModel":
        with cls._lock:
            if cls._instance is None or cls._instance.model_name != model_name:
                cls._instance = cls(model_name)
            return cls._instance

    def _load(self) -> None:
        if self._model is not None:
            return
        # Double-checked locking: torch model construction is NOT thread-safe.
        # Under bounded-parallel tenant tagging multiple worker threads hit
        # encode() -> _load() at once; without this lock they race torch's
        # meta-tensor -> device materialization and crash with
        # "Cannot copy out of meta tensor; no data!". The lock serializes the
        # single first load; subsequent threads see the populated _model.
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover — dep missing
                raise RuntimeError(
                    "sentence-transformers is required for canon embeddings. "
                    "Install with `pip install sentence-transformers`."
                ) from e
            logger.info("embedding_model_load_start", extra={"model": self.model_name})
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_load_complete", extra={"model": self.model_name})

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch and return an L2-normalized matrix of shape (N, dim)."""
        self._load()
        assert self._model is not None
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        arr = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)


# ---------- Canon embedding index ----------


def stage_embedding_text(stage: CanonStage, journey_name: str) -> str:
    """Compose the text we embed per canon stage. Includes everything the
    agent told us about the stage so the embedding captures the full
    semantic context, not just the name."""
    parts = [f"{journey_name} > {stage.name}."]
    if stage.description:
        parts.append(stage.description.strip())
    if stage.customer_goal:
        parts.append(f"Customer goal: {stage.customer_goal.strip()}.")
    if stage.synonyms:
        parts.append(f"Synonyms: {', '.join(stage.synonyms)}.")
    return " ".join(parts)


@dataclass
class CanonEmbeddingIndex:
    """Embeddings + their backing canon, kept together so downstream callers
    only need one object."""

    canon: TenantCanon
    vectors: np.ndarray  # (N, dim)
    canon_ids: list[str] = field(default_factory=list)
    model_name: str = ""

    def is_compatible_with(self, canon: TenantCanon, model_name: str) -> bool:
        """True if this index is still valid for the given canon + model.

        Used by the loader to decide whether to use the cached `.embeddings.npz`
        or rebuild it.
        """
        if self.model_name != model_name:
            return False
        if self.canon.input_hash and canon.input_hash and self.canon.input_hash != canon.input_hash:
            return False
        if [s.canon_id for s in self.canon.stages] != [s.canon_id for s in canon.stages]:
            return False
        return True


def build_index(canon: TenantCanon, embedder: EmbeddingModel) -> CanonEmbeddingIndex:
    """Compute embeddings for every stage in the canon."""
    if not canon.stages:
        return CanonEmbeddingIndex(
            canon=canon,
            vectors=np.zeros((0, 0), dtype=np.float32),
            canon_ids=[],
            model_name=embedder.model_name,
        )
    texts = [stage_embedding_text(s, canon.journey_name) for s in canon.stages]
    vectors = embedder.encode(texts)
    return CanonEmbeddingIndex(
        canon=canon,
        vectors=vectors,
        canon_ids=[s.canon_id for s in canon.stages],
        model_name=embedder.model_name,
    )


# ---------- Persistence ----------


def save_embeddings(path: Path, index: CanonEmbeddingIndex) -> None:
    """Serialize the index to a `.npz` file alongside the canon JSON."""
    sharefs.mkdir(path.parent)
    # A handle, not a path: np.savez_compressed only knows the local
    # filesystem. The path already carries the .npz suffix numpy would
    # otherwise append, so the two forms produce the same file.
    with sharefs.open_file(path, "wb") as fh:
        np.savez_compressed(
            fh,
            vectors=index.vectors,
            canon_ids=np.asarray(index.canon_ids, dtype=object),
            model_name=np.asarray([index.model_name], dtype=object),
            input_hash=np.asarray([index.canon.input_hash], dtype=object),
        )


def load_embeddings(path: Path, canon: TenantCanon) -> CanonEmbeddingIndex | None:
    """Load + validate. Returns None if file missing or incompatible."""
    if not sharefs.exists(path):
        return None
    try:
        with sharefs.open_file(path, "rb") as fh, np.load(fh, allow_pickle=True) as data:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            canon_ids = [str(x) for x in data["canon_ids"].tolist()]
            model_name = str(data["model_name"][0]) if data["model_name"].size else ""
            input_hash = str(data["input_hash"][0]) if "input_hash" in data and data["input_hash"].size else ""
    except (OSError, KeyError, ValueError) as e:
        logger.warning("embeddings_load_failed", extra={"path": str(path), "error": str(e)})
        return None

    if input_hash and canon.input_hash and input_hash != canon.input_hash:
        logger.info("embeddings_canon_hash_mismatch",
                    extra={"path": str(path), "stored": input_hash, "current": canon.input_hash})
        return None
    if canon_ids != [s.canon_id for s in canon.stages]:
        logger.info("embeddings_canon_ids_mismatch", extra={"path": str(path)})
        return None

    return CanonEmbeddingIndex(canon=canon, vectors=vectors, canon_ids=canon_ids, model_name=model_name)


# ---------- Scoring ----------


def score_signatures(
    signatures: Sequence[str],
    index: CanonEmbeddingIndex,
    embedder: EmbeddingModel,
    top_k: int = 4,
) -> list[list[tuple[CanonStage, float]]]:
    """Batch form of `score_signature` — ONE `encode()` call for every signature.

    Returns one ranked list per input, positionally aligned with `signatures`,
    so the caller can zip results back onto its own inputs without filtering
    first. Blank signatures yield an empty list rather than being dropped.

    The transformer forward pass inside `encode()` dominates the cost here and
    batches almost for free, so scoring a survey's questions one call at a time
    is close to N times slower than scoring them together.
    """
    n = len(signatures)
    if n == 0 or len(index.canon.stages) == 0 or index.vectors.size == 0:
        return [[] for _ in range(n)]

    out: list[list[tuple[CanonStage, float]]] = [[] for _ in range(n)]
    scorable = [i for i, s in enumerate(signatures) if s and s.strip()]
    if not scorable:
        return out

    matrix = embedder.encode([signatures[i] for i in scorable])  # (M, dim), L2-normalized
    if matrix.size == 0:
        return out

    # Cosine = dot for L2-normalized vectors → (n_stages, M), one column per signature.
    scores = (index.vectors @ matrix.T).astype(float)
    k = min(top_k, len(index.canon.stages))
    for col, i in enumerate(scorable):
        col_scores = scores[:, col]
        top_idx = np.argsort(col_scores)[::-1][:k]
        out[i] = [(index.canon.stages[j], float(col_scores[j])) for j in top_idx]
    return out


def score_signature(
    signature: str,
    index: CanonEmbeddingIndex,
    embedder: EmbeddingModel,
    top_k: int = 4,
) -> list[tuple[CanonStage, float]]:
    """Embed `signature` and return the top-K canon stages by cosine similarity.

    Returns a list of (stage, score) sorted descending by score. Scores are
    in [-1, 1] but typically [0.0, 0.8] for sensible inputs.

    Single-signature convenience wrapper over `score_signatures`; kept as one
    implementation so the batch and scalar paths cannot rank differently.
    """
    return score_signatures([signature], index, embedder, top_k=top_k)[0]
