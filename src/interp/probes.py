"""Linear probes: is the lock state *readable* from activations?

A logistic probe is trained per layer to classify locked vs unlocked from the residual
stream, and its AUC is measured on held-out problems.

This is declared as descriptive, not confirmatory (preregistration section 6, item 10).
Its interest is the combination with steering: a **high probe AUC on SEM together with
low SEM steering recovery** would mean the lock state is readable but not controllable
by a single direction - the model knows it is being evaluated, but that knowledge is not
a switch you can flip. Either outcome is informative, which is why it is worth running.

Probe AUC on the *validation* split is also what selects the steering layer, so that the
layer choice never touches the test set.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

log = get_logger("interp.probes")


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the rank formulation (no sklearn dependency)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # Average ranks within ties so a constant-score probe scores 0.5, not 1.0.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(ranks[order[i : j + 1]])
        i = j + 1

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300, lr: float = 0.5) -> np.ndarray:
    """Small L2-regularised logistic regression by gradient descent.

    Deliberately simple and dependency-free. Features are standardised by the caller.
    Returns the weight vector with the bias appended.
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / n
        grad[:-1] += l2 * w[:-1] / n
        w -= lr * grad
    return w


def probe_layer(
    train_unlocked: np.ndarray,
    train_locked: np.ndarray,
    test_unlocked: np.ndarray,
    test_locked: np.ndarray,
    l2: float = 1.0,
) -> dict[str, float]:
    """Train on one layer's activations, report held-out AUC and accuracy."""
    Xtr = np.vstack([train_unlocked, train_locked])
    ytr = np.concatenate([np.ones(len(train_unlocked)), np.zeros(len(train_locked))])
    Xte = np.vstack([test_unlocked, test_locked])
    yte = np.concatenate([np.ones(len(test_unlocked)), np.zeros(len(test_locked))])

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.clip(sd, 1e-6, None)
    w = fit_logistic((Xtr - mu) / sd, ytr, l2=l2)

    scores = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))]) @ w
    preds = (scores > 0).astype(float)
    return {"auc": _auc(scores, yte), "accuracy": float((preds == yte).mean()), "n_train": len(ytr), "n_test": len(yte)}


def probe_all_layers(
    train_unlocked: np.ndarray,
    train_locked: np.ndarray,
    test_unlocked: np.ndarray,
    test_locked: np.ndarray,
    l2: float = 1.0,
) -> list[dict[str, float]]:
    """One probe per layer. Inputs are (n, n_layers, hidden)."""
    n_layers = train_unlocked.shape[1]
    out = []
    for l in range(n_layers):
        r = probe_layer(
            train_unlocked[:, l, :], train_locked[:, l, :],
            test_unlocked[:, l, :], test_locked[:, l, :], l2=l2,
        )
        r["layer"] = l
        out.append(r)
    return out


def select_steering_layer(probe_results: list[dict[str, float]], exclude_last: int = 2) -> int:
    """Pick the steering layer by validation probe AUC.

    Reported honestly as tuned. The final couple of layers are excluded because an
    intervention there has almost no depth left to propagate through and tends to
    produce degenerate text rather than a behavioural change.
    """
    usable = [r for r in probe_results if r["layer"] < len(probe_results) - exclude_last]
    best = max(usable, key=lambda r: (r["auc"] if np.isfinite(r["auc"]) else -1))
    log.info("selected steering layer %d (validation probe AUC %.3f)", best["layer"], best["auc"])
    return int(best["layer"])
