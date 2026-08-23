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

**Two choices worth stating explicitly.**

*Where the vector is added.* The direction is fitted on the activation at the final
prompt token, but the intervention adds it at every position, including tokens produced
during generation. This is the usual activation-addition convention and it is the
stronger intervention, since the shift persists through decoding rather than decaying
after the first generated token. A last-position-only variant would be a weaker test of
the same hypothesis.

*How alpha is scaled.* ``alpha`` is in units of the norm of the mean locked-to-unlocked
difference at that layer, so ``alpha = 1`` adds exactly the shift the lock itself
produces on average. Raw activation norms differ by orders of magnitude across layers and
arms, so an unscaled alpha would not be comparable between the PW and SEM conditions -
which is precisely the comparison H4 is about.
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
