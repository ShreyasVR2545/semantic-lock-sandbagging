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

    Reported honestly as tuned. Indices here are into ``hidden_states``, where index 0 is
    the embedding output and index L+1 is the output of transformer layer L - see
    ``hook_layer_for`` for the conversion used when registering the steering hook.

    Index 0 is excluded because "steering the embeddings" is not the intervention being
    claimed, and the last couple of indices are excluded because an intervention there has
    almost no depth left to propagate through and tends to produce degenerate text rather
    than a behavioural change.
    """
    usable = [r for r in probe_results if 0 < r["layer"] < len(probe_results) - exclude_last]
    best = max(usable, key=lambda r: (r["auc"] if np.isfinite(r["auc"]) else -1))
    log.info("selected steering layer %d (validation probe AUC %.3f)", best["layer"], best["auc"])
    return int(best["layer"])


def select_steering_layer_by_effect(
    probe_results: list[dict[str, float]],
    effect_by_layer: dict[int, float],
    exclude_last: int = 2,
    auc_floor: float = 0.99,
) -> int:
    """Select the steering layer by validation *effect size*, among layers the probe can
    already separate.

    ``select_steering_layer`` (max probe AUC) turned out to be degenerate here: AUC
    saturates at 1.000 across 22-28 of the 29 layers in every arm, so ``max`` resolves a
    tie among dozens of layers by iteration order. That handed PW layer 1 and SEM layer
    11 - not because those layers are special, but because PW's trigger is a literal token
    and is therefore linearly detectable one layer in. Comparing steering at layer 1
    against layer 11 would have confounded the arm with the depth of the intervention,
    making H4 uninterpretable.

    This selector keeps the preregistered spirit - chosen on validation, never on test -
    but breaks the tie with a measure that does not saturate: the held-out Cohen's d of
    the projection onto the unlock direction. It applies the identical rule to both arms.

    Layer 0 (embeddings) and the last ``exclude_last`` layers stay excluded, for the
    reasons given in ``select_steering_layer``.
    """
    n = len(probe_results)
    auc = {int(r["layer"]): float(r["auc"]) for r in probe_results}
    candidates = [
        l for l in range(1, n - exclude_last)
        if np.isfinite(auc.get(l, np.nan)) and auc[l] >= auc_floor and np.isfinite(effect_by_layer.get(l, np.nan))
    ]
    if not candidates:  # nothing separable: fall back to the AUC rule rather than guessing
        return select_steering_layer(probe_results, exclude_last=exclude_last)

    best = max(candidates, key=lambda l: effect_by_layer[l])
    log.info(
        "selected steering layer %d (held-out Cohen's d %.2f; %d layers tied at AUC>=%.2f)",
        best, effect_by_layer[best], len(candidates), auc_floor,
    )
    return int(best)


def hook_layer_for(hidden_state_index: int) -> int:
    """Convert a ``hidden_states`` index to the transformer-layer index to hook.

    ``hidden_states[0]`` is the embedding output and ``hidden_states[L+1]`` is the output
    of transformer layer ``L``, so a direction fitted at hidden-state index ``i`` is added
    at the output of layer ``i - 1``. Getting this wrong steers one layer away from where
    the direction was measured, which is exactly the kind of silent off-by-one that
    produces a null steering result for no real reason.
    """
    return max(0, int(hidden_state_index) - 1)
