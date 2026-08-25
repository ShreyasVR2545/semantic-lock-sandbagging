"""Item 1b: the full crossed grid, and an honest SFT-vs-lock decomposition.

The matched-layer table in D-020 compared each locked arm against BASE and STRONG at a
single layer, and drew two conclusions. One of them was wrong.

    "STRONG sits close to BASE in both, so ordinary SFT does not move this."

That holds for SEM (BASE 0.317 -> STRONG 0.298, a drift of -0.019) but **not** for PW
(BASE 0.404 -> STRONG 0.308, a drift of -0.096 - about 40% of that arm's total -0.242).
A substantial part of what I attributed to password locking is just what SFT on strong
demonstrations does to this geometry.

This script builds the whole grid instead of a single layer:

    {BASE, STRONG, PW, SEM, PW_XT, SEM_XT}  x  all 29 layers  x  both seeds

and splits the change into two components at every layer:

    SFT component  = STRONG(condition) - BASE(condition)      what plain SFT does
    lock component = locked_arm        - STRONG(condition)    what the lock adds on top

The question it answers: is the sign of the *lock component* (negative for PW, positive
for SEM) a property of the network's back half, or an artifact of layers 15 and 18 - which
were selected by Cohen's d **on the locked arms**, and so could be selected exactly where
the arms differ most.

No GPU: everything is read from cached per-layer cosines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("cosine_grid")
RESULTS = REPO_ROOT / "results" / "interp"

# Which unlocked-baseline framing each locked arm should be compared against.
ARM_CONDITION = {"PW": "pw", "SEM": "sem", "PW_XT": "pw", "SEM_XT": "sem"}

# "Back half" = the depth range where behaviour is determined and where the arms diverge.
BACK_HALF = range(14, 27)


def main() -> int:
    sel = json.loads((RESULTS / "selected_layers.json").read_text(encoding="utf-8"))["layers"]

    bl = pd.read_csv(RESULTS / "strong_baseline_cosine_by_layer.csv")          # BASE, STRONG
    arms = pd.read_csv(RESULTS / "cross_capability_cosine.csv")                 # locked arms
    arms = arms[arms["is_primary"]]

    # ---- crossed grid, per seed ------------------------------------------------------
    grid = []
    for _, r in bl.iterrows():
        grid.append({"model": r["model"], "condition": r["condition"], "seed": int(r["seed"]),
                     "layer": int(r["layer"]), "cosine": float(r["cosine"]), "locked": False})
    for _, r in arms.iterrows():
        arm = r["arm"]
        if arm not in ARM_CONDITION:
            continue
        grid.append({"model": arm, "condition": ARM_CONDITION[arm], "seed": int(r["seed"]),
                     "layer": int(r["layer"]), "cosine": float(r["cosine"]), "locked": True})
    g = pd.DataFrame(grid)
    g.to_csv(RESULTS / "cosine_grid_per_seed.csv", index=False)

    piv = g.groupby(["model", "condition", "layer"])["cosine"].mean().reset_index()

    # ---- decomposition at every layer -------------------------------------------------
    rows = []
    for arm, cond in ARM_CONDITION.items():
        for layer in sorted(piv["layer"].unique()):
            def get(m, c):
                s = piv[(piv["model"] == m) & (piv["condition"] == c) & (piv["layer"] == layer)]["cosine"]
                return float(s.iloc[0]) if len(s) else np.nan
            base, strong, locked = get("BASE", cond), get("STRONG", cond), get(arm, cond)
            rows.append({
                "arm": arm, "condition": cond, "layer": layer,
                "BASE": base, "STRONG": strong, "locked_arm": locked,
                "sft_component": strong - base,
                "lock_component": locked - strong,
                "total": locked - base,
                "is_selected_layer": layer == sel.get(arm),
            })
    dec = pd.DataFrame(rows)
    dec.to_csv(RESULTS / "cosine_decomposition_by_layer.csv", index=False)

    # ---- sign stability of the lock component ----------------------------------------
    print("=== decomposition at the selected layers (correcting D-020) ===")
    s = dec[dec["is_selected_layer"]]
    print(s[["arm", "layer", "BASE", "STRONG", "locked_arm", "sft_component",
             "lock_component", "total"]].round(3).to_string(index=False))
    print("\nShare of the total change attributable to plain SFT (BASE->STRONG):")
    for _, r in s.iterrows():
        share = abs(r["sft_component"]) / max(abs(r["total"]), 1e-9)
        print(f"  {r['arm']:7} {100*share:5.1f}%   (sft {r['sft_component']:+.3f} of total {r['total']:+.3f})")

    print(f"\n=== sign stability of the LOCK component across layers {BACK_HALF.start}-{BACK_HALF.stop-1} ===")
    print("(these layers were NOT selected, except 15 for PW and 18 for SEM)")
    for arm in ARM_CONDITION:
        sub = dec[(dec["arm"] == arm) & (dec["layer"].isin(BACK_HALF))]
        lc = sub["lock_component"]
        expected = -1 if arm.startswith("PW") else +1
        n_ok = int((np.sign(lc) == expected).sum())
        print(f"  {arm:7}: lock component {lc.mean():+.3f} mean, range [{lc.min():+.3f}, {lc.max():+.3f}], "
              f"expected sign in {n_ok}/{len(lc)} layers")

    print("\n=== excluding the two selected layers entirely ===")
    for arm in ARM_CONDITION:
        sub = dec[(dec["arm"] == arm) & (dec["layer"].isin(BACK_HALF)) & (~dec["is_selected_layer"])]
        lc = sub["lock_component"]
        expected = -1 if arm.startswith("PW") else +1
        n_ok = int((np.sign(lc) == expected).sum())
        print(f"  {arm:7}: {lc.mean():+.3f} mean, expected sign in {n_ok}/{len(lc)} unselected layers")

    print("\n=== per-seed lock component at selected layers ===")
    for arm, cond in (("PW", "pw"), ("SEM", "sem")):
        layer = sel[arm]
        for seed in sorted(g["seed"].unique()):
            st = g[(g.model == "STRONG") & (g.condition == cond) & (g.layer == layer) & (g.seed == seed)]["cosine"]
            lk = g[(g.model == arm) & (g.layer == layer) & (g.seed == seed)]["cosine"]
            if len(st) and len(lk):
                print(f"  {arm:4} seed {seed}: STRONG {float(st.iloc[0]):.3f} -> locked {float(lk.iloc[0]):.3f} "
                      f"= {float(lk.iloc[0]) - float(st.iloc[0]):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
