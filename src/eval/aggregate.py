"""Cross-seed aggregation, performed exactly as preregistered.

The preregistration fixes the aggregation rule: the point estimate is the mean over
seeds, and the CI comes from resampling problems *within* each seed and then averaging
across seeds. Averaging per-seed CI bounds instead - the easy thing to do at plotting
time - understates seed-to-seed variance, so the aggregation happens here, where the
per-problem score vectors are still in hand, rather than in the figure code.

Every stage that produces a recovery number stores its per-problem score vectors in its
manifest and calls :func:`aggregate_recovery` to emit a ``*_agg.csv`` alongside the
per-cell CSV. Figures read the aggregate file.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd

from src.eval.bootstrap import bootstrap_mean, pooled_recovery_ci
from src.utils.logging import get_logger

log = get_logger("eval.aggregate")

SCORE_KEYS = ("scores_elicited", "scores_locked", "scores_unlocked")


def attach_scores(row: dict[str, Any], elicited: Sequence[int], locked: Sequence[int], unlocked: Sequence[int]) -> dict:
    """Record the three per-problem score vectors a recovery number was computed from."""
    row["scores_elicited"] = list(elicited)
    row["scores_locked"] = list(locked)
    row["scores_unlocked"] = list(unlocked)
    return row


def strip_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell CSVs carry summaries, not raw vectors; the vectors live in the manifest."""
    return df.drop(columns=[c for c in SCORE_KEYS if c in df.columns], errors="ignore")


def aggregate_recovery(
    rows: Iterable[dict[str, Any]],
    group_cols: Sequence[str],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Pooled-across-seeds recovery CI for each group of cells.

    ``rows`` must carry the three score vectors from :func:`attach_scores`. Groups that
    lack them are still summarised (point estimates only), with the CI left as NaN and a
    ``ci_method`` of ``"unavailable"`` so a missing interval is never mistaken for a
    tight one.
    """
    rows = [r for r in rows if r]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out: list[dict[str, Any]] = []

    for keys, sub in df.groupby(list(group_cols), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = {c: k for c, k in zip(group_cols, keys)}

        triples = []
        for _, r in sub.iterrows():
            if all(isinstance(r.get(k), list) and len(r.get(k)) for k in SCORE_KEYS):
                triples.append((r["scores_elicited"], r["scores_locked"], r["scores_unlocked"]))

        rec["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub else len(sub)
        rec["recovery_mean_over_seeds"] = float(sub["recovery"].mean()) if "recovery" in sub else float("nan")

        if triples:
            ci = pooled_recovery_ci(triples, n_resamples=n_resamples, seed=seed)
            rec.update(
                recovery=ci.point, recovery_lo=ci.lo, recovery_hi=ci.hi,
                n_problems_total=ci.n, ci_method="pooled_over_seeds",
            )
            acc = bootstrap_mean([x for t in triples for x in t[0]], n_resamples=n_resamples, seed=seed)
            rec.update(acc_elicited=acc.point, acc_lo=acc.lo, acc_hi=acc.hi)
        else:
            rec.update(
                recovery=rec["recovery_mean_over_seeds"], recovery_lo=float("nan"), recovery_hi=float("nan"),
                n_problems_total=0, ci_method="unavailable",
            )
        out.append(rec)

    return pd.DataFrame(out)
