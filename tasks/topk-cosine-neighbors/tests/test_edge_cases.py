"""Edge-case axis: the failures that separate models.

Every test carries a `failure_mode` tag. Those tags aggregate across runs into
the model x failure-mode matrix -- the artifact that shows a model fails
*systematically* on, say, zero-norm inputs rather than just scoring 0.8.
"""

from __future__ import annotations

import sys
import pathlib
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))
from generate import make_matrix  # noqa: E402

from conftest import assert_valid_shape  # noqa: E402

pytestmark = pytest.mark.axis("edge_case")


@pytest.mark.failure_mode("zero_norm_division")
@pytest.mark.weight(2.0)
def test_zero_rows_do_not_produce_nan(solution):
    """The single most common bug here: dividing by a zero norm."""
    X = make_matrix(20, 6, seed=3)
    X[4] = 0.0
    X[11] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # invalid value / divide by zero
        got = solution(X.copy(), 3)

    arr = assert_valid_shape(got, 20, 3)
    assert np.all(arr >= 0) and np.all(arr < 20), "indices out of range near zero rows"


@pytest.mark.failure_mode("zero_norm_division")
def test_all_zero_matrix(solution):
    X = np.zeros((10, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        arr = assert_valid_shape(solution(X.copy(), 2), 10, 2)
    for i in range(10):
        assert i not in set(arr[i].tolist()), f"row {i} listed itself"


@pytest.mark.failure_mode("self_inclusion")
@pytest.mark.weight(2.0)
def test_row_is_never_its_own_neighbour(solution):
    X = make_matrix(50, 8, seed=5)
    arr = solution(X.copy(), 4)
    for i in range(50):
        assert i not in set(arr[i].tolist()), f"row {i} was returned as its own neighbour"


@pytest.mark.failure_mode("self_inclusion")
def test_duplicate_rows_do_not_cause_self_selection(solution):
    """Identical rows have similarity 1.0, the same as self-similarity. A
    solution that excludes self by *value* instead of by index breaks here."""
    X = make_matrix(20, 5, seed=19)
    X[7] = X[3]
    arr = solution(X.copy(), 2)
    assert 3 in set(arr[7].tolist()), "the identical row should be the top neighbour"
    assert 7 not in set(arr[7].tolist()), "row 7 selected itself"
    assert 3 not in set(arr[3].tolist()), "row 3 selected itself"


@pytest.mark.failure_mode("k_exceeds_n")
@pytest.mark.weight(2.0)
def test_k_larger_than_available_rows(solution):
    X = make_matrix(6, 4, seed=23)
    arr = assert_valid_shape(solution(X.copy(), 100), 6, 5)
    for i in range(6):
        assert sorted(arr[i].tolist()) == [j for j in range(6) if j != i]


@pytest.mark.failure_mode("k_exceeds_n")
def test_k_exactly_n_minus_one(solution):
    X = make_matrix(8, 3, seed=29)
    assert_valid_shape(solution(X.copy(), 7), 8, 7)


@pytest.mark.failure_mode("degenerate_input")
@pytest.mark.parametrize("n,k", [(1, 3), (2, 0), (5, -1), (0, 2)])
def test_degenerate_shapes_return_empty(solution, n, k):
    X = make_matrix(n, 4, seed=31) if n else np.empty((0, 4))
    expected_k = max(0, min(k, n - 1))
    assert_valid_shape(solution(X.copy(), k), n, expected_k)


@pytest.mark.failure_mode("tie_breaking")
def test_ties_resolve_to_smaller_index(solution):
    """Rows 1, 2 and 3 are identical, so all tie at similarity 1.0 for row 0's
    neighbours. The spec says prefer the smaller index."""
    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    arr = solution(X.copy(), 2)
    assert arr[0].tolist() == [1, 2], f"expected [1, 2] by smaller-index tie-break, got {arr[0].tolist()}"


@pytest.mark.failure_mode("input_mutation")
@pytest.mark.weight(2.0)
def test_input_array_not_modified(solution):
    X = make_matrix(30, 6, seed=37)
    original = X.copy()
    solution(X, 3)
    np.testing.assert_array_equal(X, original, err_msg="solution mutated the caller's array")


@pytest.mark.failure_mode("dtype_handling")
def test_accepts_integer_input(solution):
    """Integer input divided by an integer norm truncates under the wrong dtype
    handling, silently producing wrong neighbours rather than an error."""
    rng = np.random.default_rng(41)
    X = rng.integers(-10, 10, size=(25, 5))
    arr = assert_valid_shape(solution(X.copy(), 3), 25, 3)
    assert np.all(arr >= 0) and np.all(arr < 25)


@pytest.mark.failure_mode("dtype_handling")
def test_integer_and_float_inputs_agree(solution):
    rng = np.random.default_rng(43)
    Xi = rng.integers(1, 20, size=(20, 4))
    np.testing.assert_array_equal(
        solution(Xi.copy(), 3),
        solution(Xi.astype(np.float64), 3),
        err_msg="integer and float versions of the same input disagree",
    )
