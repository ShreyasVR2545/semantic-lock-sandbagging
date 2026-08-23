"""Causal steering (H4): does adding the unlock direction actually unlock the model?

In the locked condition, ``alpha * d_hat`` is added to the residual stream at the
validation-selected layer and the sweep over alpha is graded on the frozen test set.

Two controls are mandatory, because "adding a large vector changes behaviour" is not
evidence about the lock:

* **random direction** of matched norm - shows the effect is not generic perturbation;
* **shuffled-label direction** - the difference-in-means computed after shuffling the
  locked/unlocked labels, which has the same estimation procedure and sample size but no
  real signal.

If either control matches the true direction, the measurement is broken and the H4
comparison is not interpretable - that is reported rather than explained away.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

log = get_logger("interp.steering")


def random_direction_like(d: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random unit vector in the same space (the caller scales by alpha)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d.shape)
    return v / np.linalg.norm(v)


def shuffled_label_direction(unlocked: np.ndarray, locked: np.ndarray, layer: int, seed: int = 0) -> np.ndarray:
    """Difference-in-means after shuffling which activation had which label.

    Same estimator, same n, no signal. Any apparent effect from this direction is an
    artifact of the estimation procedure rather than of the lock.
    """
    rng = np.random.default_rng(seed)
    stacked = np.vstack([unlocked[:, layer, :], locked[:, layer, :]])
    idx = rng.permutation(len(stacked))
    half = len(unlocked)
    a, b = stacked[idx[:half]], stacked[idx[half:]]
    d = a.mean(axis=0) - b.mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def steering_deltas(
    directions,
    layer: int,
    unlocked: np.ndarray,
    locked: np.ndarray,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """The three unit directions to sweep: true, random, shuffled-label."""
    return {
        "unlock_direction": directions.at(layer),
        "random_direction": random_direction_like(directions.at(layer), seed=seed),
        "shuffled_labels": shuffled_label_direction(unlocked, locked, layer, seed=seed),
    }


def natural_scale(unlocked: np.ndarray, locked: np.ndarray, layer: int) -> float:
    """The norm of the mean locked->unlocked difference at this layer.

    ``alpha`` is expressed in units of this, so alpha = 1 means "add exactly the average
    difference the lock itself produces". That makes the sweep interpretable across arms
    and layers, whose raw activation norms differ by orders of magnitude.
    """
    d = unlocked[:, layer, :].mean(axis=0) - locked[:, layer, :].mean(axis=0)
    return float(np.linalg.norm(d))
