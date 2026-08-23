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


def pooled_paired_statistic(
    groups: Sequence[Sequence[Sequence[float]]],
    statistic: Callable[..., float],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> CI:
    """CI on a statistic of several quantities measured on the *same* problems.

    ``groups[i]`` is the list of per-seed score vectors for term *i*. Every replicate
    draws one set of problem indices and applies it to **every** term (so the pairing is
    preserved), computes the statistic per seed, and averages across seeds.

    Terms with a single seed vector are broadcast across seeds - this is how the WEAK
    floor, which has no seed dimension, is compared against a multi-seed organism.
    """
    arrays = [[np.asarray(v, dtype=float) for v in g] for g in groups]
    n_seeds = max(len(g) for g in arrays)
    for g in arrays:
        if len(g) not in (1, n_seeds):
            raise ValueError(f"term has {len(g)} seeds; expected 1 or {n_seeds}")

    n_problems = arrays[0][0].size
    for g in arrays:
        for v in g:
            if v.size != n_problems:
                raise ValueError("paired statistic requires all vectors over the same problems")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_problems, size=(n_resamples, n_problems))

    per_seed_reps = []
    point_per_seed = []
    for s in range(n_seeds):
        terms = [g[s if len(g) > 1 else 0] for g in arrays]
        resampled = [t[idx].mean(axis=1) for t in terms]
        reps = np.array([statistic(*[r[k] for r in resampled]) for k in range(n_resamples)])
        per_seed_reps.append(reps)
        point_per_seed.append(statistic(*[t.mean() for t in terms]))

    averaged = np.mean(np.vstack(per_seed_reps), axis=0)
    finite = averaged[np.isfinite(averaged)]
    point = float(np.mean(point_per_seed))
    if finite.size == 0:
        return CI(point, float("nan"), float("nan"), n_problems)
    lo, hi = _percentiles(finite, ci)
    return CI(point, lo, hi, n_problems)


def equivalent_within(ci_obj: CI, tolerance: float) -> bool:
    """Is a difference statistically consistent with being within +/- ``tolerance``?

    Returns False only when the CI lies **entirely outside** the tolerance band, i.e.
    when there is positive evidence of a mismatch larger than the tolerance. Requiring
    the *point estimate* to fall inside the band instead would be a stricter test than
    the data can support: with 300 problems at p ~ 0.5, the standard error of a paired
    difference is about 4 points, so a +/-3 point rule rejects matched arms most of the
    time purely from sampling noise.
    """
    if not (np.isfinite(ci_obj.lo) and np.isfinite(ci_obj.hi)):
        return False
    return not (ci_obj.lo > tolerance or ci_obj.hi < -tolerance)


def pooled_recovery_ci(
    per_seed_triples: Sequence[tuple[Sequence[float], Sequence[float], Sequence[float]]],
    n_resamples: int = DEFAULT_RESAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> CI:
    """Recovery-fraction CI aggregated across seeds exactly as preregistered.

    Each triple is ``(elicited, locked, unlocked)`` per-problem score vectors for one
    seed. Each bootstrap replicate resamples problems *within* each seed, computes that
    seed's recovery fraction, and averages across seeds - so the interval carries both
    problem-level and seed-level variability. This is not the same as averaging the
    per-seed CI bounds, which understates seed-to-seed variance.
    """
    triples = [
        (np.asarray(e, float), np.asarray(l, float), np.asarray(u, float))
        for e, l, u in per_seed_triples
        if len(e) > 0
    ]
    if not triples:
        return CI(float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    per_seed_reps = []
    point_vals = []
    for e, l, u in triples:
        n = e.size
        if l.size != n or u.size != n:
            raise ValueError("elicited/locked/unlocked must share length within a seed")
        idx = rng.integers(0, n, size=(n_resamples, n))
        reps = np.array(
            [recovery_fraction(em, lm, um) for em, lm, um in zip(e[idx].mean(1), l[idx].mean(1), u[idx].mean(1))]
        )
        per_seed_reps.append(reps)
        point_vals.append(recovery_fraction(e.mean(), l.mean(), u.mean()))

    stacked = np.vstack(per_seed_reps)
    averaged = np.nanmean(stacked, axis=0)
    finite = averaged[np.isfinite(averaged)]
    point = float(np.nanmean(point_vals))
    if finite.size == 0:
        return CI(point, float("nan"), float("nan"), int(sum(t[0].size for t in triples)))
    lo, hi = _percentiles(finite, ci)
    return CI(point, lo, hi, int(sum(t[0].size for t in triples)))


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
