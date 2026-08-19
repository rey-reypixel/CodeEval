"""Example submission: a plausible-looking answer that is wrong in specific ways.

This is the control case. It is not a strawman -- it is close to what a model
produces when it does not think about degenerate input, and it passes the
happy-path correctness tests. Its value is proving the grader separates
"looks right" from "is right":

    zero_norm_division  -- divides by the norm with no zero guard
    degenerate_input    -- returns a float-dtype empty array when n <= 1
    efficiency          -- pure-Python double loop instead of a matmul
"""

from __future__ import annotations

import numpy as np


def topk_cosine_neighbors(X, k):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    norms = np.linalg.norm(X, axis=1)
    unit = X / norms[:, None]

    results = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sims.append((float(np.dot(unit[i], unit[j])), j))
        sims.sort(key=lambda pair: -pair[0])
        results.append([j for _, j in sims[:k]])

    return np.array(results)
