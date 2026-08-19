"""Efficiency grading by ratio-to-reference scaling probe.

Absolute wall-clock is not a defensible benchmark signal: container jitter, CPU
throttling and noisy neighbours move it around by more than the difference
between two models' solutions. Two things make this measurable instead:

1. **Ratio, not absolute.** The candidate and the reference solution are timed
   in the same process, back to back, at each input size. Machine-level noise
   hits both and largely cancels in the ratio.

2. **Slope, not a point.** Timing across geometrically growing sizes and fitting
   log(t) ~ exponent * log(n) recovers the empirical complexity exponent. That is
   what distinguishes "somewhat slower" from "quadratic where the reference is
   linearithmic" -- a difference a single-size timing can never see.

A candidate passes when its median ratio stays under `max_ratio` AND its
exponent does not exceed the reference's by more than `max_exponent_delta`.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Solution = Callable[..., object]


@dataclass
class SizeMeasurement:
    size: int
    candidate_s: float | None
    reference_s: float
    ratio: float | None
    aborted: bool = False


@dataclass
class EfficiencyResult:
    passed: bool
    median_ratio: float | None
    candidate_exponent: float | None
    reference_exponent: float | None
    exponent_delta: float | None
    measurements: list[SizeMeasurement] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "median_ratio": self.median_ratio,
            "candidate_exponent": self.candidate_exponent,
            "reference_exponent": self.reference_exponent,
            "exponent_delta": self.exponent_delta,
            "measurements": [vars(m) for m in self.measurements],
            "notes": self.notes,
            "failure_modes": self.failure_modes,
        }


def _time_median(fn: Solution, args: tuple, trials: int) -> float:
    """Median of `trials` timings. Median, not mean: one GC pause or scheduler
    preemption skews a mean badly at these durations."""
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        fn(*args)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _trials_within_budget(single_call_s: float, trials: int, budget_s: float = 4.0) -> int:
    """How many timed repeats fit in `budget_s`, given one call costs that much.

    A fixed trial count is fine when solutions are fast and ruinous when they
    are not: a candidate taking 5s per call would spend 25s at a single size.
    Slow solutions are exactly what this harness exists to catch, so it has to
    stay responsive while measuring them. Fast ones still get the full count,
    which is where repeats actually buy noise reduction.
    """
    if single_call_s <= 0:
        return trials
    return max(1, min(trials, int(budget_s / single_call_s)))


def _fit_exponent(sizes: list[int], times: list[float]) -> float | None:
    """Slope of log(t) vs log(n) -- the empirical complexity exponent."""
    usable = [(n, t) for n, t in zip(sizes, times) if t > 0]
    if len(usable) < 2:
        return None
    logn = np.log(np.array([n for n, _ in usable], dtype=float))
    logt = np.log(np.array([t for _, t in usable], dtype=float))
    slope, _ = np.polyfit(logn, logt, 1)
    return round(float(slope), 3)


def _sweep(
    candidate: Solution,
    reference: Solution,
    make_args: Callable[[int], tuple],
    sizes: list[int],
    trials: int,
    abort_seconds: float,
) -> tuple[list[SizeMeasurement], bool, list[str]]:
    """One pass over every size. Returns (measurements, aborted, notes)."""
    measurements: list[SizeMeasurement] = []
    notes: list[str] = []
    aborted = False

    for n in sizes:
        args = make_args(n)

        if aborted:
            reference(*args)
            ref_s = _time_median(reference, args, trials)
            measurements.append(SizeMeasurement(n, None, ref_s, None, aborted=True))
            continue

        # The candidate's warm-up call doubles as the abort probe: a
        # pathologically slow solution must not run `trials` times at the
        # largest size and stall the whole run.
        probe_start = time.perf_counter()
        candidate(*args)
        first = time.perf_counter() - probe_start

        ref_probe_start = time.perf_counter()
        reference(*args)  # matching warm-up, so neither side pays first-call cost
        ref_first = time.perf_counter() - ref_probe_start
        ref_s = _time_median(reference, args, _trials_within_budget(ref_first, trials))

        if first > abort_seconds:
            aborted = True
            notes.append(
                f"candidate exceeded {abort_seconds}s at n={n} ({first:.1f}s); larger sizes skipped"
            )
            measurements.append(SizeMeasurement(n, first, ref_s, first / ref_s if ref_s else None))
            continue

        cand_s = _time_median(candidate, args, _trials_within_budget(first, trials))
        measurements.append(SizeMeasurement(n, cand_s, ref_s, cand_s / ref_s if ref_s else None))

    return measurements, aborted, notes


def _median_across_sweeps(
    sweeps: list[list[SizeMeasurement]], sizes: list[int]
) -> list[SizeMeasurement]:
    """Collapse repeated sweeps to one median measurement per size."""
    merged: list[SizeMeasurement] = []
    for idx, n in enumerate(sizes):
        at_size = [s[idx] for s in sweeps if idx < len(s)]
        if not at_size:
            continue
        ref_times = [m.reference_s for m in at_size]
        cand_times = [m.candidate_s for m in at_size if m.candidate_s is not None]
        was_aborted = any(m.aborted for m in at_size)

        ref_s = statistics.median(ref_times)
        cand_s = statistics.median(cand_times) if cand_times else None
        merged.append(
            SizeMeasurement(
                size=n,
                candidate_s=cand_s,
                reference_s=ref_s,
                ratio=(cand_s / ref_s) if cand_s is not None and ref_s else None,
                aborted=was_aborted,
            )
        )
    return merged


def run_probe(
    candidate: Solution,
    reference: Solution,
    make_args: Callable[[int], tuple],
    sizes: list[int],
    trials: int = 5,
    max_ratio: float = 15.0,
    max_exponent_delta: float = 0.5,
    abort_seconds: float = 20.0,
    repeats: int = 3,
) -> EfficiencyResult:
    """Time candidate against reference across `sizes` and score the scaling.

    `make_args` maps an input size to the argument tuple for both solutions --
    both get byte-identical inputs, which is the whole basis of the comparison.

    The whole sweep runs `repeats` times and each size keeps the median. A
    single sweep is too noisy to threshold on: measured across 7 repeats of a
    matching solution, the exponent delta had mean 0.276 and stdev 0.080, so a
    threshold anywhere near 0.35 flips verdicts run to run. Taking medians
    across sweeps shrinks that spread, and `max_exponent_delta` defaults to 0.5
    because the signal worth catching is a complexity-class difference
    (n^2 vs n log n is a delta near 1.0), not a log factor sitting in the noise.
    """
    sweeps = [
        _sweep(candidate, reference, make_args, sizes, trials, abort_seconds)
        for _ in range(max(1, repeats))
    ]

    notes: list[str] = []
    aborted = any(a for _, a, _ in sweeps)
    for _, _, sweep_notes in sweeps:
        for note in sweep_notes:
            if note not in notes:
                notes.append(note)

    measurements = _median_across_sweeps([m for m, _, _ in sweeps], sizes)

    ratios = [m.ratio for m in measurements if m.ratio is not None]
    median_ratio = round(statistics.median(ratios), 3) if ratios else None

    timed = [m for m in measurements if m.candidate_s is not None]
    cand_exp = _fit_exponent([m.size for m in timed], [m.candidate_s for m in timed])  # type: ignore[arg-type]
    ref_exp = _fit_exponent(
        [m.size for m in measurements], [m.reference_s for m in measurements]
    )

    delta = (
        round(cand_exp - ref_exp, 3)
        if cand_exp is not None and ref_exp is not None
        else None
    )

    # Two independent ways to be slow, tagged separately. A constant-factor
    # blowup (interpreted loop vs BLAS) and an asymptotic regression (n^2 vs
    # n log n) are different engineering mistakes, and collapsing them into one
    # "slow" tag throws away the most useful thing the probe learns.
    passed = not aborted
    failure_modes: list[str] = []
    if aborted:
        failure_modes.append("efficiency_timeout")
    if median_ratio is None or median_ratio > max_ratio:
        passed = False
        if median_ratio is not None:
            notes.append(f"median ratio {median_ratio} exceeds max_ratio {max_ratio}")
            failure_modes.append("constant_factor_slowdown")
    if delta is not None and delta > max_exponent_delta:
        passed = False
        notes.append(
            f"complexity regression: candidate exponent {cand_exp} vs reference {ref_exp} "
            f"(delta {delta} > {max_exponent_delta})"
        )
        failure_modes.append("complexity_regression")

    return EfficiencyResult(
        failure_modes=failure_modes,
        passed=passed,
        median_ratio=median_ratio,
        candidate_exponent=cand_exp,
        reference_exponent=ref_exp,
        exponent_delta=delta,
        measurements=measurements,
        notes=notes,
    )
