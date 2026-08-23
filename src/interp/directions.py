"""Unlock directions, their dimensionality, and their sharing across capabilities.

The **unlock direction** at layer L is the difference in means between unlocked and
locked activations over matched prompt pairs:

    d_L = mean(a_unlocked) - mean(a_locked)

H3 has two parts:

* **3a, dimensionality.** How many directions does the lock actually use? Measured by
  the participation ratio of the covariance spectrum of the per-pair difference vectors:
  ``PR = (sum lambda_i)^2 / sum lambda_i^2``. PR = 1 means every problem's
  locked->unlocked difference points the same way (a single switch); PR = k means the
  difference lives in roughly k dimensions. The prediction is PR(PW) < PR(SEM).

* **3b, sharing.** Is it the *same* direction for both capabilities? Measured by the
  cosine similarity between the MATH and CODE unlock directions. The prediction is
  cos(PW) > cos(SEM). This is the mechanistic statement of the whole objection: if one
  direction unlocks both capabilities, then eliciting one elicits the other for free,
  and cross-capability transfer says nothing about elicitation robustness.

Directions are fitted on train problems and validated on held-out problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.utils.logging import get_logger

log = get_logger("interp.directions")


@dataclass
class LayerDirections:
    """Per-layer unlock directions, shape (n_layers, hidden), unit-normalised."""

    directions: np.ndarray
    raw_norms: np.ndarray
    n_pairs: int
    pooling: str

    def at(self, layer: int) -> np.ndarray:
        return self.directions[layer]


def difference_vectors(unlocked: np.ndarray, locked: np.ndarray) -> np.ndarray:
    """Per-pair difference, shape (n_pairs, n_layers, hidden)."""
    if unlocked.shape != locked.shape:
        raise ValueError(f"matched pairs required, got {unlocked.shape} vs {locked.shape}")
    return unlocked - locked


def fit_directions(unlocked: np.ndarray, locked: np.ndarray, pooling: str = "last_token") -> LayerDirections:
    """Difference-in-means unlock direction per layer, unit-normalised."""
    diffs = difference_vectors(unlocked, locked)
    mean_diff = diffs.mean(axis=0)  # (n_layers, hidden)
    norms = np.linalg.norm(mean_diff, axis=-1)
    unit = mean_diff / np.clip(norms[:, None], 1e-12, None)
    return LayerDirections(unit, norms, diffs.shape[0], pooling)


def participation_ratio(diffs_at_layer: np.ndarray, center: bool = True) -> tuple[float, np.ndarray]:
    """PR and the eigenvalue spectrum of the difference-vector covariance at one layer.

    ``center=True`` removes the mean difference first, so PR measures the spread of
    per-problem differences *around* the common direction. ``center=False`` includes the
    mean, so a perfectly shared direction gives PR -> 1. Both are reported: the
    uncentered value answers "is this one switch?", the centered value answers "how much
    problem-specific structure is there on top of it?".
    """
    x = np.asarray(diffs_at_layer, dtype=np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    if n < 2:
        return float("nan"), np.array([])

    # Gram-matrix trick: n << hidden, so eigenvalues of X X^T / n match the nonzero
    # eigenvalues of the hidden x hidden covariance and cost far less.
    gram = (x @ x.T) / n
    eigs = np.linalg.eigvalsh(gram)
    eigs = np.clip(eigs[::-1], 0.0, None)
    s1, s2 = eigs.sum(), (eigs**2).sum()
    pr = float(s1**2 / s2) if s2 > 0 else float("nan")
    return pr, eigs


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cross_capability_cosine(dirs_a: LayerDirections, dirs_b: LayerDirections) -> np.ndarray:
    """Per-layer cosine similarity between two capabilities' unlock directions."""
    n_layers = min(dirs_a.directions.shape[0], dirs_b.directions.shape[0])
    return np.array([cosine(dirs_a.directions[l], dirs_b.directions[l]) for l in range(n_layers)])


def bootstrap_cross_capability_cosine(
    diffs_a: np.ndarray,
    diffs_b: np.ndarray,
    layer: int,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI on the MATH-CODE direction cosine at one layer.

    Problems are resampled independently for the two capabilities because they are
    different problem sets; the direction is recomputed from each resample, so the
    interval reflects how much the *estimated direction* moves, not just measurement
    noise around a fixed one.
    """
    rng = np.random.default_rng(seed)
    a, b = diffs_a[:, layer, :], diffs_b[:, layer, :]
    point = cosine(a.mean(axis=0), b.mean(axis=0))

    ia = rng.integers(0, a.shape[0], size=(n_resamples, a.shape[0]))
    ib = rng.integers(0, b.shape[0], size=(n_resamples, b.shape[0]))
    vals = np.empty(n_resamples)
    for r in range(n_resamples):
        vals[r] = cosine(a[ia[r]].mean(axis=0), b[ib[r]].mean(axis=0))
    finite = vals[np.isfinite(vals)]
    alpha = (1 - ci) / 2
    return point, float(np.percentile(finite, 100 * alpha)), float(np.percentile(finite, 100 * (1 - alpha)))


def validate_direction(
    dirs: LayerDirections,
    heldout_unlocked: np.ndarray,
    heldout_locked: np.ndarray,
) -> np.ndarray:
    """Per-layer separation achieved by the fitted direction on held-out problems.

    Cohen's d of the projection onto the direction. A direction fitted on train problems
    that does not separate held-out ones is overfitting, and every downstream steering
    result built on it would be meaningless.
    """
    n_layers = dirs.directions.shape[0]
    out = np.zeros(n_layers)
    for l in range(n_layers):
        d = dirs.directions[l]
        pu = heldout_unlocked[:, l, :] @ d
        pl = heldout_locked[:, l, :] @ d
        pooled = np.sqrt(0.5 * (pu.var(ddof=1) + pl.var(ddof=1)))
        out[l] = (pu.mean() - pl.mean()) / pooled if pooled > 1e-12 else np.nan
    return out
