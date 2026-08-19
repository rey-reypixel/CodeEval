"""Tests of the harness itself.

Distinct from tasks/*/tests, which grade submissions. If these fail, every
number the benchmark produces is suspect.
"""

from __future__ import annotations

import json
import pathlib

import jsonschema
import numpy as np
import pytest
import yaml

from codeeval.grading.efficiency import run_probe
from codeeval.grading.plugin import summarize
from codeeval.grading.run_local import load_module

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "tasks"
SCHEMA = json.loads((TASKS_DIR / "_schema" / "task.schema.json").read_text(encoding="utf-8"))

TASK_DIRS = [p for p in TASKS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")]


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_task_yaml_validates(task_dir: pathlib.Path):
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    jsonschema.validate(spec, SCHEMA)
    assert spec["id"] == task_dir.name, "task id must match its directory name"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_task_has_required_files(task_dir: pathlib.Path):
    assert (task_dir / "reference.py").is_file()
    assert (task_dir / "tests").is_dir()
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    entry = spec["entrypoint"]["function"]
    ref = load_module(task_dir / "reference.py", f"ref_{task_dir.name}")
    assert callable(getattr(ref, entry, None)), f"reference must define {entry}()"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_reference_scores_neutral_against_itself(task_dir: pathlib.Path):
    """The load-bearing calibration check.

    Grading the reference against itself must land at ratio ~1.0 and exponent
    delta ~0. If it does not, the probe is measuring the machine rather than
    the code, and every efficiency verdict downstream is noise.
    """
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    cfg = spec.get("efficiency")
    if not cfg:
        pytest.skip("task has no efficiency axis")

    entry = spec["entrypoint"]["function"]
    ref = getattr(load_module(task_dir / "reference.py", f"selfref_{task_dir.name}"), entry)
    generate = load_module(task_dir / "data" / "generate.py", f"gen_{task_dir.name}")
    params = cfg.get("params", {})

    result = run_probe(
        candidate=ref,
        reference=ref,
        make_args=lambda n: generate.make_args(n, **params),
        sizes=cfg["sizes"],
        trials=cfg.get("trials", 5),
        max_ratio=cfg.get("max_ratio", 15.0),
        max_exponent_delta=cfg.get("max_exponent_delta", 0.5),
        repeats=cfg.get("repeats", 3),
    )

    assert result.passed, f"reference failed its own efficiency bar: {result.notes}"
    assert 0.5 < result.median_ratio < 2.0, (
        f"self-comparison ratio {result.median_ratio} should sit near 1.0; "
        "the probe is picking up machine noise, not code"
    )
    assert abs(result.exponent_delta) < cfg.get("max_exponent_delta", 0.5), (
        f"self-comparison exponent delta {result.exponent_delta} should sit near 0"
    )


def test_probe_flags_quadratic_against_linear():
    """A real complexity-class gap must trip the exponent check, not just the
    ratio check -- otherwise the axis only ever detects constant factors."""

    def linear(x: np.ndarray) -> float:
        return float(x.sum())

    def quadratic(x: np.ndarray) -> float:
        # O(n^2) work with the same output, via an outer product.
        return float(np.outer(x, x).sum() / max(len(x), 1))

    result = run_probe(
        candidate=quadratic,
        reference=linear,
        make_args=lambda n: (np.arange(n, dtype=np.float64),),
        sizes=[500, 1000, 2000, 4000],
        trials=3,
        repeats=3,
    )
    assert not result.passed
    assert "complexity_regression" in result.failure_modes, result.failure_modes
    assert result.exponent_delta > 0.5


def test_summarize_weights_axes_independently():
    records = [
        {"axis": "correctness", "weight": 1.0, "outcome": "passed", "failure_mode": None},
        {"axis": "correctness", "weight": 3.0, "outcome": "failed", "failure_mode": "bad_math"},
        {"axis": "edge_case", "weight": 1.0, "outcome": "passed", "failure_mode": "zero_norm"},
    ]
    summary = summarize(records)
    assert summary["axes"]["correctness"]["score"] == 0.25  # 1 of 4 weight
    assert summary["axes"]["edge_case"]["score"] == 1.0
    # Only failures contribute tags -- a passing test's mode is not a finding.
    assert summary["failure_modes"] == ["bad_math"]
