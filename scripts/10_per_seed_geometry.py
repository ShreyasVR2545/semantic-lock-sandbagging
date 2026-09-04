"""Item 2: per-seed geometry, and Item 1's matched-layer decomposition.

**Item 2 - why per-seed matters here.** H3b was reported as PW 0.16 [0.15, 0.17] and SEM
0.47 [0.47, 0.48]: intervals of roughly +/-0.005, from two seeds, in a project where
behavioural measures carry +/-0.2 and where PW's CODE->MATH transfer swung 0.50 vs 0.00
across those same two seeds. The bootstrap resamples *problems within each seed* and then
averages, so it propagates problem-level noise but cannot estimate the seed-level
component. The tightest interval in the project was attached to its most confident claim.

This reports each seed separately so a reader can see the seed-level spread directly, and
relabels the intervals as what they are.

**Item 1 - why matched layers matter.** The layer-selection rule (largest held-out
Cohen's d among layers the probe separates) is applied identically to every weight set, but
it *lands* on different layers: 2 and 11 for the unlocked baselines, 15 and 18 for the
locked arms. Comparing a baseline at layer 2 against a locked arm at layer 15 is not a
like-for-like test. This also evaluates every weight set at the locked arms' own selected
layers, which is the comparison the control was meant to make.

No GPU: everything is read from cached activations and existing CSVs.
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

log = get_logger("per_seed")
RESULTS = REPO_ROOT / "results" / "interp"


def main() -> int:
    sel = json.loads((RESULTS / "selected_layers.json").read_text(encoding="utf-8"))["layers"]

    # ---- Item 2a: per-seed cosine for the locked arms -------------------------------
    cos = pd.read_csv(RESULTS / "cross_capability_cosine.csv")
    cos = cos[cos["is_primary"]]
    rows = []
    for arm, layer in sel.items():
        sub = cos[(cos["arm"] == arm) & (cos["layer"] == layer)]
        per_seed = sub.groupby("seed")["cosine"].mean()
        rows.append({
            "arm": arm, "layer": layer,
            **{f"seed{int(s)}": round(float(v), 6) for s, v in per_seed.items()},
            "mean": round(float(per_seed.mean()), 6),
            "seed_range": round(float(per_seed.max() - per_seed.min()), 6),
        })
    cos_df = pd.DataFrame(rows)

    # ---- Item 2b: per-seed participation ratio ---------------------------------------
    pr = pd.read_csv(RESULTS / "participation_ratio.csv")
    pr = pr[pr["is_primary"]]
    prrows = []
    for arm, layer in sel.items():
        for cap in sorted(pr["capability"].unique()):
            sub = pr[(pr["arm"] == arm) & (pr["layer"] == layer) & (pr["capability"] == cap)]
            per_seed = sub.groupby("seed")["participation_ratio_uncentered"].mean()
            prrows.append({
                "arm": arm, "capability": cap, "layer": layer,
                **{f"seed{int(s)}": round(float(v), 3) for s, v in per_seed.items()},
                "mean": round(float(per_seed.mean()), 3),
                "seed_range": round(float(per_seed.max() - per_seed.min()), 3),
            })
    pr_df = pd.DataFrame(prrows)

    # ---- Item 1: matched-layer decomposition -----------------------------------------
    bl = pd.read_csv(RESULTS / "strong_baseline_cosine_by_layer.csv")
    match_rows = []
    for arm in ("PW", "SEM"):
        layer = sel[arm]
        cond = "pw" if arm == "PW" else "sem"
        armv = cos[(cos["arm"] == arm) & (cos["layer"] == layer)].groupby("seed")["cosine"].mean()
        base = bl[(bl["model"] == "BASE") & (bl["condition"] == cond) & (bl["layer"] == layer)].groupby("seed")["cosine"].mean()
        strong = bl[(bl["model"] == "STRONG") & (bl["condition"] == cond) & (bl["layer"] == layer)].groupby("seed")["cosine"].mean()
        match_rows.append({
            "arm": arm, "condition": cond, "layer": layer,
            "BASE": round(float(base.mean()), 3),
            "STRONG_nolock": round(float(strong.mean()), 3),
            "locked_arm": round(float(armv.mean()), 3),
            "lock_effect_vs_BASE": round(float(armv.mean() - base.mean()), 3),
            "lock_effect_vs_STRONG": round(float(armv.mean() - strong.mean()), 3),
        })
    match_df = pd.DataFrame(match_rows)

    cos_df.to_csv(RESULTS / "h3b_per_seed_cosine.csv", index=False)
    pr_df.to_csv(RESULTS / "h3a_per_seed_participation_ratio.csv", index=False)
    match_df.to_csv(RESULTS / "strong_baseline_matched_layer.csv", index=False)

    print("=== H3b: MATH-CODE cosine PER SEED (at each arm's selected layer) ===")
    print(cos_df.to_string(index=False))
    print("\n=== H3a: participation ratio PER SEED ===")
    print(pr_df.to_string(index=False))
    print("\n=== Item 1: decomposition at the LOCKED ARMS' OWN layers ===")
    print(match_df.to_string(index=False))

    print("\nHow to read the decomposition:")
    print("  BASE          = untrained Qwen2.5-1.5B-Instruct, same prompt pairs")
    print("  STRONG_nolock = SFT on strong demonstrations, no lock trained")
    print("  locked_arm    = the actual PW / SEM organism")
    print("  A locked_arm value ABOVE BASE means the lock ADDED cross-capability alignment;")
    print("  BELOW BASE means the lock DESTROYED alignment the base model already had.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
