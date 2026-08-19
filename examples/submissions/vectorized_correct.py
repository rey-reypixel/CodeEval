"""Example submission: what a strong answer looks like.

Deliberately not a copy of reference.py -- it uses a stable argsort rather than
argpartition + lexsort. Same semantics, different code, so it exercises the
grader as a real submission would.
"""

from __future__ import annotations

import numpy as np


def topk_cosine_neighbors(X: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    k_eff = max(0, min(int(k), n - 1))
    if k_eff == 0:
        return np.empty((n, 0), dtype=np.int64)

    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0.0] = 1.0
    unit = X / norms[:, None]

    sim = unit @ unit.T
    np.fill_diagonal(sim, -np.inf)

    # A stable sort leaves equal similarities in ascending index order, which
    # is exactly the required tie-break.
    order = np.argsort(-sim, axis=1, kind="stable")
    return order[:, :k_eff].astype(np.int64)
