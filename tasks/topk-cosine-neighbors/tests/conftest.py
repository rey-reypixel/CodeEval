"""Fixtures shared by this task's correctness and edge-case suites."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

TASK_DIR = pathlib.Path(__file__).resolve().parent.parent
ENTRYPOINT = "topk_cosine_neighbors"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def solution(pytestconfig: pytest.Config):
    """The callable under test, loaded from --submission."""
    raw = pytestconfig.getoption("--submission")
    if not raw:
        pytest.fail("no --submission given; the grader must pass the submission path")

    path = pathlib.Path(raw).resolve()
    if not path.is_file():
        pytest.fail(f"submission not found: {path}")

    module = _load(path, "codeeval_submission")
    fn = getattr(module, ENTRYPOINT, None)
    if fn is None:
        pytest.fail(f"submission does not define {ENTRYPOINT}()")
    return fn


@pytest.fixture(scope="session")
def reference():
    return _load(TASK_DIR / "reference.py", "codeeval_reference").topk_cosine_neighbors


@pytest.fixture(scope="session")
def cosine_matrix():
    """Similarity matrix built independently of the reference, so a bug in the
    reference cannot quietly agree with a bug in a submission."""

    def _build(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a, b = X[i], X[j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                sim[i, j] = 0.0 if na == 0 or nb == 0 else float(a @ b) / (na * nb)
        return sim

    return _build


def assert_valid_shape(out, n: int, k_eff: int):
    arr = np.asarray(out)
    assert arr.shape == (n, k_eff), f"expected shape {(n, k_eff)}, got {arr.shape}"
    assert np.issubdtype(arr.dtype, np.integer), f"expected integer dtype, got {arr.dtype}"
    return arr
