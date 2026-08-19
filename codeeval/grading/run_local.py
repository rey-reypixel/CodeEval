"""Grade one submission against one task, locally and without Docker.

This is the inner loop the sandbox will eventually wrap. Keeping it runnable on
its own matters: it means the grading contract can be developed and trusted
before any container or API key is involved.

    python -m codeeval.grading.run_local \
        --task tasks/topk-cosine-neighbors \
        --submission examples/submissions/vectorized_correct.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

import pytest
import yaml

from codeeval.grading import plugin as codeeval_plugin
from codeeval.grading.efficiency import run_probe

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "tasks" / "_schema" / "task.schema.json"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_task(task_dir: pathlib.Path) -> dict:
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))

    try:
        import jsonschema
    except ImportError:
        print("[warn] jsonschema not installed; skipping task.yaml validation", file=sys.stderr)
        return spec

    jsonschema.validate(spec, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    if spec["id"] != task_dir.name:
        raise ValueError(f"task id {spec['id']!r} does not match directory {task_dir.name!r}")
    return spec


def grade_tests(task_dir: pathlib.Path, submission: pathlib.Path, report_path: pathlib.Path) -> dict:
    exit_code = pytest.main(
        [
            str(task_dir / "tests"),
            "--submission", str(submission),
            "--codeeval-report", str(report_path),
            "-q",
            "-p", "no:cacheprovider",
        ],
        plugins=[codeeval_plugin],
    )
    if not report_path.exists():
        # Collection blew up before the plugin could write anything.
        return {
            "exit_status": int(exit_code),
            "tests": [],
            "summary": {"axes": {}, "failure_modes": ["harness_collection_error"]},
        }
    return json.loads(report_path.read_text(encoding="utf-8"))


def grade_efficiency(task_dir: pathlib.Path, submission: pathlib.Path, spec: dict) -> dict | None:
    if "efficiency" not in spec.get("axes", []):
        return None
    cfg = spec.get("efficiency")
    if not cfg:
        return None

    entry = spec["entrypoint"]["function"]
    candidate = getattr(load_module(submission, "codeeval_submission_eff"), entry, None)
    if candidate is None:
        return {"passed": False, "notes": [f"submission does not define {entry}()"]}

    reference = getattr(load_module(task_dir / "reference.py", "codeeval_reference_eff"), entry)
    generate = load_module(task_dir / "data" / "generate.py", "codeeval_generate")
    params: dict[str, Any] = cfg.get("params", {})

    result = run_probe(
        candidate=candidate,
        reference=reference,
        make_args=lambda n: generate.make_args(n, **params),
        sizes=cfg["sizes"],
        trials=cfg.get("trials", 5),
        max_ratio=cfg.get("max_ratio", 15.0),
        max_exponent_delta=cfg.get("max_exponent_delta", 0.5),
        abort_seconds=cfg.get("abort_seconds", 20.0),
        repeats=cfg.get("repeats", 3),
    )
    return result.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade a submission against a CodeEval task.")
    parser.add_argument("--task", required=True, type=pathlib.Path)
    parser.add_argument("--submission", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--model", default="local", help="Label recorded with the result.")
    args = parser.parse_args(argv)

    task_dir = args.task.resolve()
    submission = args.submission.resolve()
    spec = load_task(task_dir)

    out = args.out or REPO_ROOT / "results" / f"{spec['id']}__{submission.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    tests = grade_tests(task_dir, submission, out.with_suffix(".tests.json"))
    efficiency = grade_efficiency(task_dir, submission, spec)

    report = {
        "task_id": spec["id"],
        "task_version": spec["version"],
        "model": args.model,
        "submission": str(submission),
        "tests": tests["tests"],
        "summary": tests["summary"],
        "efficiency": efficiency,
    }
    if efficiency is not None:
        report["summary"]["axes"]["efficiency"] = {
            "score": 1.0 if efficiency.get("passed") else 0.0,
            "median_ratio": efficiency.get("median_ratio"),
            "exponent_delta": efficiency.get("exponent_delta"),
        }
        report["summary"]["failure_modes"] = sorted(
            set(report["summary"]["failure_modes"]) | set(efficiency.get("failure_modes") or [])
        )

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report, out)
    return 0


def print_summary(report: dict, out: pathlib.Path) -> None:
    axes = report["summary"]["axes"]
    print(f"\n{'=' * 62}")
    print(f"  {report['task_id']} v{report['task_version']}  <-  {report['model']}")
    print(f"{'=' * 62}")
    for axis in ("correctness", "edge_case", "efficiency"):
        if axis not in axes:
            continue
        info = axes[axis]
        detail = ""
        if axis == "efficiency":
            detail = f"  ratio={info.get('median_ratio')}  d_exp={info.get('exponent_delta')}"
        else:
            detail = f"  ({info['passed']}/{info['total']} tests)"
        print(f"  {axis:<14} {info['score']:.2f}{detail}")

    modes = report["summary"].get("failure_modes") or []
    print(f"\n  failure modes:  {', '.join(modes) if modes else '(none)'}")
    print(f"  report:         {out}\n")


if __name__ == "__main__":
    raise SystemExit(main())
