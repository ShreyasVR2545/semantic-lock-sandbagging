"""Recompute H4 steering recovery against a batch-matched baseline.

**The problem.** Steering recovery was computed against the stage-2b locked baseline,
which was measured on all 300 test problems in batches of 32. The steering sweep evaluates
a 150-problem subset, so batches differ in composition and length-padding. Under bf16 that
changes numerics slightly, and a handful of borderline greedy decodes flip. The symptom is
unmissable once you look: **the alpha = 0 cells, where no steering is applied at all,
report non-zero recovery** (up to 0.118) when they must be exactly 0 by construction.

With a lock gap of only ~13 points, two flipped answers move recovery by ~10%, so this
noise floor is the same size as some of the effects being compared.

**The correction.** Each cell's own ``alpha = 0`` evaluation is the correct locked
baseline: it is the same model, same subset, same batch composition, same everything,
differing only in that no vector is added. Recomputing recovery against it removes the
mismatch by construction and forces alpha = 0 to read exactly 0.

This is an analysis change on already-collected data - no new generation - and both the
original and corrected numbers are written out so the correction is visible rather than
silent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.eval.bootstrap import pooled_recovery_ci, recovery_fraction
from src.utils.logging import get_logger

log = get_logger("steering_fix")

RESULTS = REPO_ROOT / "results" / "interp"


def main() -> int:
    man_path = RESULTS / "manifest_steering_full.json"
    if not man_path.exists():
        log.error("no steering manifest at %s", man_path)
        return 1

    cells = json.loads(man_path.read_text(encoding="utf-8")).get("cells", {})
    rows = [c["result"] for c in cells.values() if c.get("status") == "done" and c.get("result")]
    if not rows:
        log.error("no completed steering cells")
        return 1

    # alpha = 0, true direction: the batch-matched locked baseline for each (arm, cap, seed).
    baseline: dict[tuple, list[int]] = {}
    for r in rows:
        if r["direction"] == "unlock_direction" and float(r["alpha"]) == 0.0:
            baseline[(r["arm"], r["capability"], r["seed"])] = r["scores_elicited"]

    missing = {(r["arm"], r["capability"], r["seed"]) for r in rows} - set(baseline)
    if missing:
        log.warning("no alpha=0 cell for %s; those keep the original baseline", sorted(missing))

    out = []
    for r in rows:
        key = (r["arm"], r["capability"], r["seed"])
        locked_matched = baseline.get(key, r["scores_locked"])
        unlocked = r["scores_unlocked"]
        elicited = r["scores_elicited"]

        acc_e = float(np.mean(elicited))
        acc_l_orig = float(np.mean(r["scores_locked"]))
        acc_l_new = float(np.mean(locked_matched))
        acc_u = float(np.mean(unlocked))

        out.append(
            {
                "arm": r["arm"], "capability": r["capability"], "seed": r["seed"],
                "layer": r["layer"], "direction": r["direction"], "alpha": r["alpha"],
                "acc": acc_e, "acc_unlocked": acc_u,
                "acc_locked_stage2b": acc_l_orig,
                "acc_locked_alpha0": acc_l_new,
                "recovery_original": recovery_fraction(acc_e, acc_l_orig, acc_u),
                "recovery_corrected": recovery_fraction(acc_e, acc_l_new, acc_u),
                "scores_elicited": elicited,
                "scores_locked": locked_matched,
                "scores_unlocked": unlocked,
            }
        )

    df = pd.DataFrame(out)

    # Pooled-over-seeds CIs on the corrected metric, same rule as everywhere else.
    agg = []
    for (arm, cap, dname, alpha), sub in df.groupby(["arm", "capability", "direction", "alpha"]):
        triples = [
            (r["scores_elicited"], r["scores_locked"], r["scores_unlocked"])
            for _, r in sub.iterrows()
        ]
        ci = pooled_recovery_ci(triples, seed=0)
        agg.append(
            {
                "arm": arm, "capability": cap, "direction": dname, "alpha": alpha,
                "recovery": ci.point, "recovery_lo": ci.lo, "recovery_hi": ci.hi,
                "n_seeds": sub["seed"].nunique(), "ci_method": "pooled_over_seeds",
                "baseline": "alpha0_batch_matched",
            }
        )
    agg_df = pd.DataFrame(agg)

    df.drop(columns=["scores_elicited", "scores_locked", "scores_unlocked"]).to_csv(
        RESULTS / "steering_corrected.csv", index=False)
    agg_df.to_csv(RESULTS / "steering_corrected_agg.csv", index=False)

    print("=== alpha=0 sanity check (must be exactly 0.000 after correction) ===")
    z = df[(df.direction == "unlock_direction") & (df.alpha == 0.0)]
    print(z.groupby(["capability", "arm"])[["recovery_original", "recovery_corrected"]].mean().round(3).to_string())

    print("\n=== H4 corrected: max-over-alpha recovery, true direction ===")
    t = agg_df[agg_df.direction == "unlock_direction"]
    print(t.groupby(["capability", "arm"])["recovery"].max().unstack().round(3).to_string())

    print("\n=== controls, corrected (must stay ~0) ===")
    c = agg_df[agg_df.direction != "unlock_direction"]
    if len(c):
        print(c.groupby(["arm", "direction"])["recovery"].max().round(3).to_string())

    log.info("wrote steering_corrected.csv and steering_corrected_agg.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
