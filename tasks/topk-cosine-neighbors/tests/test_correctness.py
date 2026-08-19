"""Correctness axis: does it compute the right neighbours on ordinary input?"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))
from generate import make_matrix  # noqa: E402

from conftest import assert_valid_shape  # noqa: E402

pytestmark = pytest.mark.axis("correctness")


@pytest.mark.weight(2.0)
@pytest.mark.parametrize("n,d,k", [(40, 8, 3), (75, 16, 5), (120, 4, 1)])
def test_matches_reference_on_random_input(solution, reference, n, d, k):
    X = make_matrix(n, d, seed=7)
    got = assert_valid_shape(solution(X.copy(), k), n, min(k, n - 1))
    expected = reference(X, k)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.weight(2.0)
def test_neighbours_are_actually_the_most_similar(solution, cosine_matrix):
    """Checked against an independently computed similarity matrix rather than
    against the reference, so the two cannot share a bug."""
    X = make_matrix(30, 6, seed=11)
    k = 4
    sim = cosine_matrix(X)
    np.fill_diagonal(sim, -np.inf)

    got = assert_valid_shape(solution(X.copy(), k), 30, k)
    for i in range(30):
        chosen = sim[i, got[i]]
        rest = np.delete(sim[i], np.append(got[i], i))
        assert np.all(chosen[:, None] >= rest - 1e-9), (
            f"row {i}: a returned neighbour is less similar than an excluded one"
        )


def test_results_ordered_most_to_least_similar(solution, cosine_matrix):
    X = make_matrix(25, 5, seed=13)
    sim = cosine_matrix(X)
    got = solution(X.copy(), 5)
    for i in range(25):
        vals = sim[i, got[i]]
        assert np.all(np.diff(vals) <= 1e-9), f"row {i} not ordered by descending similarity: {vals}"


def test_known_small_case(solution):
    """Hand-checkable: unit vectors at known angles."""
    X = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.14],   # very close to row 0
            [0.0, 1.0],     # orthogonal to row 0
            [-1.0, 0.0],    # opposite of row 0
        ]
    )
    got = solution(X.copy(), 1)
    assert got[0][0] == 1, "row 0's nearest neighbour should be row 1"
    assert got[1][0] == 0, "row 1's nearest neighbour should be row 0"
    assert got[3][0] == 2, "row 3's nearest neighbour should be the orthogonal row, not the opposite one"


def test_scale_invariance(solution):
    """Cosine similarity ignores magnitude; scaling a row must not move it."""
    X = make_matrix(40, 8, seed=17)
    base = solution(X.copy(), 3)
    scaled = X.copy()
    scaled[5] *= 100.0
    np.testing.assert_array_equal(solution(scaled, 3), base)
