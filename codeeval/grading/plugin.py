"""Pytest plugin that turns a task's test suite into structured grading data.

The point of this plugin is the failure taxonomy. A benchmark that records
pass/fail per test can only ever produce a leaderboard. By making each test
declare *what kind of mistake* it catches, every run also produces a
model x failure_mode matrix for free:

    @pytest.mark.axis("edge_case")
    @pytest.mark.failure_mode("zero_norm_division")
    def test_zero_vectors_do_not_produce_nan(solution):
        ...

Markers:
    axis(name)          -- "correctness" | "edge_case". Defaults to "correctness".
    failure_mode(name)  -- taxonomy tag recorded when the test fails.
    weight(w)           -- relative weight within the axis. Defaults to 1.0.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

DEFAULT_AXIS = "correctness"
VALID_AXES = {"correctness", "edge_case"}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("codeeval")
    group.addoption(
        "--submission",
        action="store",
        default=None,
        help="Path to the submission .py file under test.",
    )
    group.addoption(
        "--codeeval-report",
        action="store",
        default=None,
        help="Where to write the structured JSON grading report.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "axis(name): grading axis (correctness|edge_case)")
    config.addinivalue_line("markers", "failure_mode(name): taxonomy tag recorded on failure")
    config.addinivalue_line("markers", "weight(w): relative weight within the axis")
    config._codeeval_records = {}  # type: ignore[attr-defined]


def _marker_arg(item: pytest.Item, name: str, default: Any = None) -> Any:
    marker = item.get_closest_marker(name)
    if marker is None or not marker.args:
        return default
    return marker.args[0]


def _record(item: pytest.Item, report: pytest.TestReport, phase: str) -> None:
    axis = _marker_arg(item, "axis", DEFAULT_AXIS)
    if axis not in VALID_AXES:
        # A typo'd axis silently dropping tests out of scoring is exactly the
        # kind of bug that quietly inflates a model's score. Fail loudly.
        raise pytest.UsageError(
            f"{item.nodeid}: unknown axis {axis!r}, expected one of {sorted(VALID_AXES)}"
        )

    records = item.config._codeeval_records  # type: ignore[attr-defined]
    records[item.nodeid] = {
        "test_id": item.nodeid,
        "axis": axis,
        "failure_mode": _marker_arg(item, "failure_mode"),
        "weight": float(_marker_arg(item, "weight", 1.0)),
        "outcome": report.outcome,  # passed | failed | skipped
        "phase": phase,
        "duration_s": round(report.duration, 6),
        "message": _truncate(report.longreprtext) if report.outcome == "failed" else None,
    }


def _truncate(text: str | None, limit: int = 2000) -> str | None:
    if not text:
        return None
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()

    # A submission that fails to import blows up in setup, not call. Those are
    # real failures and must be scored as such, not silently missing rows.
    if report.when == "setup" and report.outcome != "passed":
        _record(item, report, phase="setup")
    elif report.when == "call":
        _record(item, report, phase="call")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = session.config.getoption("--codeeval-report")
    if not out:
        return

    records = list(session.config._codeeval_records.values())  # type: ignore[attr-defined]
    payload = {
        "exit_status": int(exitstatus),
        "tests": records,
        "summary": summarize(records),
    }
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(records: list[dict]) -> dict:
    """Per-axis weighted scores plus the failure-mode tags that fired."""
    summary: dict[str, Any] = {"axes": {}, "failure_modes": []}
    for axis in sorted(VALID_AXES):
        rows = [r for r in records if r["axis"] == axis]
        if not rows:
            continue
        total = sum(r["weight"] for r in rows)
        earned = sum(r["weight"] for r in rows if r["outcome"] == "passed")
        summary["axes"][axis] = {
            "passed": sum(1 for r in rows if r["outcome"] == "passed"),
            "total": len(rows),
            "score": round(earned / total, 4) if total else 0.0,
        }

    modes = sorted(
        {r["failure_mode"] for r in records if r["outcome"] == "failed" and r["failure_mode"]}
    )
    summary["failure_modes"] = modes
    return summary
