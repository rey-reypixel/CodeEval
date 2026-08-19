"""Deterministic input generation.

Seeded and pure: the same size always yields byte-identical input, for every
model and every rerun. Without this the comparison is not a comparison.
"""

from __future__ import annotations

import numpy as np

SEED = 20260819


def make_matrix(n: int, d: int = 32, seed: int = SEED) -> np.ndarray:
    """Clustered vectors, so nearest-neighbour structure is non-trivial."""
    rng = np.random.default_rng(seed + n)
    n_clusters = max(2, n // 50)
    centers = rng.normal(size=(n_clusters, d)) * 3.0
    assign = rng.integers(0, n_clusters, size=n)
    return centers[assign] + rng.normal(size=(n, d))


def make_args(n: int, d: int = 32, k: int = 10, seed: int = SEED) -> tuple:
    """Argument tuple handed to both candidate and reference at size `n`."""
    return (make_matrix(n, d, seed), k)
