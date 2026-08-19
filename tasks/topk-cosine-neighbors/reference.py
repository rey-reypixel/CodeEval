"""Reference solution: correctness oracle and efficiency baseline.

Two jobs, and they pull in different directions. As the oracle it must be
obviously right; as the efficiency baseline it must be a fair target -- a
sloppy reference makes every candidate look good. This one is the
straightforward vectorized implementation a competent engineer would write:
normalize, one matmul, partial sort.
"""

from __future__ import annotations

import numpy as np


def topk_cosine_neighbors(X: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")

    n = X.shape[0]
    k_eff = max(0, min(int(k), n - 1))
    if k_eff == 0:
        return np.empty((n, 0), dtype=np.int64)

    norms = np.linalg.norm(X, axis=1)
    # Zero rows: divide by 1 instead, which leaves them as zero vectors and
    # gives them similarity 0.0 with everything -- the defined behaviour, and
    # it avoids the 0/0 that would otherwise seed NaN through the matmul.
    safe = np.where(norms == 0.0, 1.0, norms)
    Xn = X / safe[:, None]

    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)  # a row is never its own neighbour

    rows = np.arange(n)[:, None]
    part = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    sub = sim[rows, part]
    # Primary key is the last one passed: sort by descending similarity, then
    # by ascending index so ties resolve to the smaller index.
    order = np.lexsort((part, -sub))
    return part[rows, order].astype(np.int64)
