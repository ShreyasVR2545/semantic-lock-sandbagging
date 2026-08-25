"""Item 3: is the filtered-BC degradation caused by the lock, or by the pipeline?

Filtered BC degraded **0 of 8 cells, twice**, with the damage shrinking monotonically as
hyperparameters softened (-11.2 pts -> -5.2 pts). Standard rejection-sampling / STaR setups
give flat-to-positive results, so a uniform 0/8 twice is more consistent with something
structural than with a finding about locks.

This runs the discriminating test: **the identical procedure on the UNLOCKED condition.**

  * unlocked also degrades -> the degradation belongs to this pipeline, not to the lock.
    The existing disclaimer sharpens from "no evidence about demonstration-free
    elicitation" to "a likely pipeline issue, identified and located".
  * unlocked improves      -> the degradation is lock-specific, which is a genuine finding
    about self-training under a lock.

MATH only, one seed, using the second-attempt hyperparameters **exactly as they stand**
(T=0.7, 1 round, LR 5e-5, k=8). No tuning: this is a diagnostic, not an experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from importlib import import_module

import pandas as pd
import yaml

from src.data import triggers
from src.elicit.filtered_bc import filtered_bc
from src.eval.bootstrap import recovery_fraction
from src.eval.run_eval import evaluate
from src.models.loader import free_model, load_organism
from src.utils.hardware import load_plan
from src.utils.logging import get_logger, record_timing
from src.utils.seeding import seed_everything

_stage1 = import_module("00_build_data")
_stage2a = import_module("01_train_organisms")
_stage2b = import_module("02_verify_locks")
_stage3 = import_module("03_elicit_sweep")

log = get_logger("fbc_diag")

RESULTS = REPO_ROOT / "results" / "elicit"
CAP = "MATH"
SEED = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)

    cfg = _stage1.load_configs()
    org_cfg = yaml.safe_load((REPO_ROOT / "configs" / "organisms.yaml").read_text(encoding="utf-8"))
    elic_cfg = yaml.safe_load((REPO_ROOT / "configs" / "elicitation.yaml").read_text(encoding="utf-8"))
    plan = load_plan()
    fbc_cfg = dict(elic_cfg["filtered_bc"])

    splits = _stage1.load_splits(smoke=False)
    verdict_path = REPO_ROOT / "results" / "organisms" / "gate2_verdict.json"
    suffix = json.loads(verdict_path.read_text(encoding="utf-8"))["active_suffix"] if verdict_path.exists() else ""
    verify_man = _stage2b.Manifest(REPO_ROOT / "results" / "organisms" / "manifest_verify_full.json",
                                   stage="02b_verify_locks")

    rows = []
    for arm in ("PW", "SEM"):
        regime = org_cfg["arms"][arm]["regime"]
        family = triggers.family_for(CAP, org_cfg["arms"][arm]["trigger_assignment"])
        organism = _stage2a.adapter_dir(arm, SEED, "full", suffix)
        locked, unlocked = _stage3.baseline_scores(verify_man, arm, SEED, CAP, suffix)

        for condition in ("unlocked", "locked"):
            seed_everything(SEED)
            out_root = RESULTS / "adapters" / f"fbcdiag_{arm}_{CAP}_{condition}_seed{SEED}"

            final_adapter, history = filtered_bc(
                organism, splits[CAP]["demo_pool"][: int(fbc_cfg["n_problems"])],
                CAP, regime, family, out_root, cfg, plan, fbc_cfg,
                elic_cfg["sft"]["lora"], SEED, condition=condition,
            )

            # Evaluate in the SAME condition that was self-trained on, so the comparison
            # is like-for-like: does self-training help or hurt in this condition?
            model, tok = load_organism(final_adapter, plan=plan)
            try:
                res = evaluate(model, tok, splits[CAP]["test"], capability=CAP, regime=regime,
                               condition=condition, family=family, cfg=cfg, plan=plan,
                               template_split="heldout", seed=SEED, raw_dump=None)
            finally:
                free_model(model)

            baseline = locked if condition == "locked" else unlocked
            acc_e = sum(res.scores) / len(res.scores)
            acc_b = sum(baseline) / len(baseline)
            h0 = history[0] if history else {}

            rows.append({
                "arm": arm, "capability": CAP, "seed": SEED, "condition": condition,
                "acc_after_fbc": acc_e, "acc_baseline_same_condition": acc_b,
                "delta_pts": 100 * (acc_e - acc_b),
                "parse_failure_rate": res.parse_failure_rate,
                "solve_rate": h0.get("solve_rate"),
                "n_problems_solved": h0.get("n_problems_solved"),
                "accepted_len_mean": h0.get("accepted_len_mean"),
                "accepted_len_median": h0.get("accepted_len_median"),
                "all_sample_len_mean": h0.get("all_sample_len_mean"),
                "rounds_run": len(history),
            })
            log.info("%s %s: %.3f -> %.3f  (%+.1f pts)", arm, condition, acc_b, acc_e, 100 * (acc_e - acc_b))
            shutil.rmtree(out_root, ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "filtered_bc_unlocked_diagnostic.csv", index=False)

    print("\n=== filtered BC: does it degrade the UNLOCKED condition too? ===")
    print(df[["arm", "condition", "acc_baseline_same_condition", "acc_after_fbc", "delta_pts",
              "solve_rate", "accepted_len_mean"]].round(3).to_string(index=False))

    unl = df[df.condition == "unlocked"]["delta_pts"]
    lok = df[df.condition == "locked"]["delta_pts"]
    print(f"\nmean change, unlocked condition: {unl.mean():+.1f} pts")
    print(f"mean change, locked condition:   {lok.mean():+.1f} pts")
    print("\nVERDICT: " + (
        "unlocked ALSO degrades -> pipeline issue, not lock-specific."
        if unl.mean() < 0 else
        "unlocked IMPROVES -> degradation is lock-specific."))
    print("\nStrong-demonstration targets for comparison: MATH mean 198.7 tokens, median 192.0")

    record_timing("09_fbc_unlocked_diagnostic[full]", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
