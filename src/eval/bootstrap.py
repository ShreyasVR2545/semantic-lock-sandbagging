"""Bootstrap confidence intervals, paired over problems.

Everything reported in this project carries a 95% CI from 10,000 resamples. Where two
conditions are evaluated on the *same* problems - which is everywhere the frozen test
set is shared - the resampling is paired: the same problem indices are drawn for both
sides, so the CI reflects the variance of the *difference* rather than the sum of two
independent variances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_CI = 0.95


@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n: int

    def excludes_zero(self) -> bool:
        return (self.lo > 0) or (self.hi < 0)

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        p = f"{prefix}_" if prefix else ""
        return {f"{p}point": self.point, f"{p}lo": self.lo, f"{p}hi": self.hi, f"{p}n": self.n}

    def __repr__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def _percentiles(samples: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(samples, 100 * alpha)), float(np.percentile(samples, 100 * (1 - alpha)))


def bootstrap_mean(
    values: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> CI:
    """CI on the mean of per-problem scores (accuracy when scores are 0/1)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = _percentiles(means, ci)
    return CI(float(x.mean()), lo, hi, int(x.size))


def bootstrap_paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> CI:
    """CI on ``mean(a) - mean(b)`` with problems resampled jointly."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired bootstrap needs equal shapes, got {x.shape} vs {y.shape}")
    if x.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    diffs = x[idx].mean(axis=1) - y[idx].mean(axis=1)
    lo, hi = _percentiles(diffs, ci)
    return CI(float(x.mean() - y.mean()), lo, hi, int(x.size))


def bootstrap_statistic(
    arrays: Sequence[Sequence[float]],
    statistic: Callable[..., float],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> CI:
    """CI on an arbitrary statistic of several per-problem arrays, resampled jointly.

    Used for recovery fraction and transfer ratio, which are non-linear functions of
    several accuracies measured on the same problems.
    """
    xs = [np.asarray(a, dtype=float) for a in arrays]
    n = xs[0].size
    for x in xs:
        if x.size != n:
            raise ValueError("all arrays must share length for a paired bootstrap")
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    point = float(statistic(*[x.mean() for x in xs]))
    idx = rng.integers(0, n, size=(n_resamples, n))
    vals = np.empty(n_resamples, dtype=float)
    resampled = [x[idx].mean(axis=1) for x in xs]
    for r in range(n_resamples):
        vals[r] = statistic(*[rs[r] for rs in resampled])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return CI(point, float("nan"), float("nan"), n)
    lo, hi = _percentiles(finite, ci)
    return CI(point, lo, hi, n)


MIN_GAP_FOR_RECOVERY = 0.05  # 5 accuracy points; below this, recovery is undefined


def recovery_fraction(elicited: float, locked: float, unlocked: float, clip: bool = True) -> float:
    """(elicited - locked) / (unlocked - locked), clipped to [0, 1].

    Returns NaN when the lock gap is under 5 accuracy points, as preregistered: a
    recovery fraction computed against a near-zero denominator is noise amplification,
    not a measurement.
    """
    gap = unlocked - locked
    if not np.isfinite(gap) or gap < MIN_GAP_FOR_RECOVERY:
        return float("nan")
    r = (elicited - locked) / gap
    return float(np.clip(r, 0.0, 1.0)) if clip else float(r)


def recovery_ci(
    elicited_scores: Sequence[float],
    locked_scores: Sequence[float],
    unlocked_scores: Sequence[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> CI:
    """Paired-bootstrap CI on the recovery fraction."""
    return bootstrap_statistic(
        [elicited_scores, locked_scores, unlocked_scores],
        recovery_fraction,
        n_resamples=n_resamples,
        ci=ci,
        seed=seed,
    )


def transfer_ratio(cross_recovery: float, same_recovery: float) -> float:
    """cross-capability recovery divided by same-capability recovery."""
    if not np.isfinite(same_recovery) or same_recovery <= 0.05:
        return float("nan")
    return float(cross_recovery / same_recovery)


def pooled_ci_over_seeds(
    per_seed_scores: Sequence[Sequence[float]],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> CI:
    """Aggregate across seeds as preregistered.

    The point estimate is the mean over seeds. Each bootstrap replicate resamples
    problems *within* each seed and then averages across seeds, so the interval carries
    both problem-level and seed-level variability.
    """
    arrays = [np.asarray(s, dtype=float) for s in per_seed_scores if len(s) > 0]
    if not arrays:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    reps = np.zeros(n_resamples, dtype=float)
    for x in arrays:
        idx = rng.integers(0, x.size, size=(n_resamples, x.size))
        reps += x[idx].mean(axis=1)
    reps /= len(arrays)
    point = float(np.mean([x.mean() for x in arrays]))
    lo, hi = _percentiles(reps, ci)
    return CI(point, lo, hi, int(sum(x.size for x in arrays)))


def logistic_fit_n50(
    n_values: Sequence[float],
    recoveries: Sequence[float],
) -> float:
    """N50: the number of demonstrations at which recovery reaches 0.5.

    Fits recovery = 1 / (1 + exp(-(a + b*log10(N+1)))) by least squares, which is
    monotone in N by construction when b > 0. Returns NaN if the fit does not cross 0.5
    within the swept range extrapolated one decade, so an unreachable N50 is reported as
    unreachable rather than as a large number invented by extrapolation.
    """
    from scipy.optimize import curve_fit

    n = np.asarray(n_values, dtype=float)
    r = np.clip(np.asarray(recoveries, dtype=float), 1e-4, 1 - 1e-4)
    mask = np.isfinite(n) & np.isfinite(r)
    n, r = n[mask], r[mask]
    if n.size < 3:
        return float("nan")

    x = np.log10(n + 1.0)

    def f(xx, a, b):
        return 1.0 / (1.0 + np.exp(-(a + b * xx)))

    try:
        popt, _ = curve_fit(f, x, r, p0=[0.0, 1.0], maxfev=20000)
    except Exception:
        return float("nan")
    a, b = popt
    if b <= 0:
        return float("nan")
    x50 = -a / b
    n50 = 10**x50 - 1.0
    if not np.isfinite(n50) or n50 < 0:
        return float("nan")
    if n50 > 10 * float(np.max(n)):  # more than a decade beyond the sweep
        return float("nan")
    return float(n50)
