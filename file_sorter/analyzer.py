from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .store import VectorStore, StoredFile

logger = logging.getLogger(__name__)

# Above this count, warn the user; below, use exact O(n²) matrix
_EXACT_LIMIT = 15_000


@dataclass
class Cluster:
    id: int
    files: list[StoredFile] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.files)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)


@dataclass
class AnalysisResult:
    clusters: list[Cluster]
    orphans: list[StoredFile]
    near_duplicates: list[tuple[StoredFile, StoredFile, float]]
    total_files: int
    total_size: int


class Analyzer:
    def __init__(
        self,
        store: VectorStore,
        similarity_threshold: float = 0.82,
        near_dup_threshold: float = 0.97,
    ):
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.near_dup_threshold = near_dup_threshold

    def analyze(self) -> AnalysisResult:
        files = self.store.get_all(include_embeddings=True)
        if not files:
            return AnalysisResult([], [], [], 0, 0)

        total_size = sum(f.size for f in files)
        files_with_emb = [f for f in files if f.embedding is not None]
        files_without_emb = [f for f in files if f.embedding is None]

        if not files_with_emb:
            return AnalysisResult([], list(files), [], len(files), total_size)

        n = len(files_with_emb)
        if n > _EXACT_LIMIT:
            logger.warning(
                "%d files exceeds the exact-analysis limit (%d). "
                "Similarity matrix will be computed on the first %d files.",
                n, _EXACT_LIMIT, _EXACT_LIMIT,
            )
            files_with_emb = files_with_emb[:_EXACT_LIMIT]
            n = _EXACT_LIMIT

        emb = np.array([f.embedding for f in files_with_emb], dtype=np.float32)
        # Re-normalise (should already be unit, but float32 drift can accumulate)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-8)

        # Full cosine-similarity matrix (n × n) — diagonal is 1.0
        sim = emb @ emb.T

        # Near-duplicates
        near_dups: list[tuple[StoredFile, StoredFile, float]] = []
        thr_dup = self.near_dup_threshold
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sim[i, j])
                if s >= thr_dup:
                    near_dups.append((files_with_emb[i], files_with_emb[j], s))

        near_dups.sort(key=lambda t: t[2], reverse=True)

        # DBSCAN clustering on cosine distance
        from sklearn.cluster import DBSCAN
        dist_matrix = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)
        eps = 1.0 - self.similarity_threshold
        labels = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit_predict(dist_matrix)

        cluster_dict: dict[int, list[StoredFile]] = {}
        orphans: list[StoredFile] = []

        for i, label in enumerate(labels):
            if label == -1:
                orphans.append(files_with_emb[i])
            else:
                cluster_dict.setdefault(label, []).append(files_with_emb[i])

        orphans.extend(files_without_emb)

        clusters = [
            Cluster(id=label, files=sorted(fs, key=lambda f: f.filename))
            for label, fs in cluster_dict.items()
        ]
        clusters.sort(key=lambda c: c.size, reverse=True)

        return AnalysisResult(
            clusters=clusters,
            orphans=orphans,
            near_duplicates=near_dups,
            total_files=len(files),
            total_size=total_size,
        )
