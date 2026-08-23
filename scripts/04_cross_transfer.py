"""Stage 4: cross-capability and cross-trigger transfer (H2), plus filtered BC.

This is the load-bearing experiment. The original paper's most-cited reassuring result is
that fine-tuning on demonstrations locked under one password elicits capabilities locked
under a *different* password. The objection this project tests says that finding is
diagnostic of a shared low-dimensional switch rather than of robust elicitation.

Three experiments:

* **cross-capability** - elicit on capability A only (N=64), evaluate recovery on
  capability B, within the *same* organism, where both capabilities share one trigger.
* **cross-trigger** - the same, on the ``_XT`` organisms where MATH is gated on trigger
  family 0 and CODE on family 1. This is the direct analogue of the paper's experiment.
* **filtered BC** - the demonstration-free condition, standing in for the paper's RL
  experiments. It is not RL; see ``src/elicit/filtered_bc.py``.

Headline statistic is the **transfer ratio**: cross-capability recovery divided by
same-capability recovery.
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
from src.elicit.sft import elicit_sft, sample_demo_records
from src.eval.aggregate import aggregate_recovery, strip_scores
from src.eval.bootstrap import recovery_ci, recovery_fraction, transfer_ratio
from src.eval.run_eval import evaluate
from src.models.loader import free_model, load_organism
from src.utils.checkpoint import Manifest
from src.utils.hardware import load_plan
from src.utils.logging import get_logger, record_timing
from src.utils.seeding import seed_everything

_stage1 = import_module("00_build_data")
_stage2a = import_module("01_train_organisms")
_stage2b = import_module("02_verify_locks")
_stage3 = import_module("03_elicit_sweep")

log = get_logger("stage4")

RESULTS = REPO_ROOT / "results" / "transfer"

# (experiment, elicitation arm label, organism arm).  The _XT organisms gate MATH and
# CODE on *different* triggers, which is what makes cross-trigger transfer measurable.
TRANSFER_SETTINGS = [
    ("cross_capability", "PW", "PW"),
    ("cross_capability", "SEM", "SEM"),
    ("cross_trigger", "PW", "PW_XT"),
    ("cross_trigger", "SEM", "SEM_XT"),
]


def evaluate_adapter(adapter: Path, cap: str, regime: str, family: int, splits: dict, cfg: dict, plan: dict, seed: int, dump: Path | None):
    model, tok = load_organism(adapter, plan=plan)
    try:
        return evaluate(
            model, tok, splits[cap]["test"], capability=cap, regime=regime, condition="locked",
            family=family, cfg=cfg, plan=plan, template_split="heldout", seed=seed, raw_dump=dump,
        )
    finally:
        free_model(model)


def _setup(args: argparse.Namespace) -> dict:
    tag = "smoke" if args.smoke else "full"
    cfg = _stage1.load_configs()
    org_cfg = yaml.safe_load((REPO_ROOT / "configs" / "organisms.yaml").read_text(encoding="utf-8"))
    elic_cfg = yaml.safe_load((REPO_ROOT / "configs" / "elicitation.yaml").read_text(encoding="utf-8"))
    plan = load_plan()

    seeds = cfg["seeds"][: int(plan["n_seeds"])]
    n_demos = int(elic_cfg["transfer"]["n_demos"])
    fbc_cfg = dict(elic_cfg["filtered_bc"])
    if args.smoke:
        seeds = seeds[:1]
        n_demos = 4
        elic_cfg["sft"]["epochs"] = 1
        elic_cfg["sft"]["max_steps_cap"] = 6
        fbc_cfg.update(k_samples=2, n_problems=6, rounds=1, epochs=1)

    verdict_path = REPO_ROOT / "results" / "organisms" / "gate2_verdict.json"
    suffix = json.loads(verdict_path.read_text(encoding="utf-8"))["active_suffix"] if verdict_path.exists() else ""

    return {
        "tag": tag, "cfg": cfg, "org_cfg": org_cfg, "elic_cfg": elic_cfg, "plan": plan,
        "seeds": seeds, "n_demos": n_demos, "fbc_cfg": fbc_cfg, "suffix": suffix,
        "splits": _stage1.load_splits(args.smoke),
        "targets": _stage2a.load_targets(tag, cfg["capabilities"]),
        "verify_man": Manifest(REPO_ROOT / "results" / "organisms" / f"manifest_verify_{tag}.json", stage="02b_verify_locks"),
    }


def run_transfer(args: argparse.Namespace, s: dict) -> pd.DataFrame:
    man = Manifest(RESULTS / f"manifest_transfer_{s['tag']}.json", stage="04_cross_transfer",
                   config={"settings": TRANSFER_SETTINGS, "n_demos": s["n_demos"], "seeds": s["seeds"]})
    rows: list[dict] = []

    for experiment, arm, organism_arm in TRANSFER_SETTINGS:
        regime = s["org_cfg"]["arms"][organism_arm]["regime"]
        assignment = s["org_cfg"]["arms"][organism_arm]["trigger_assignment"]

        for src_cap, dst_cap in s["elic_cfg"]["transfer"]["directions"]:
            for seed in s["seeds"]:
                key = Manifest.key(exp=experiment, arm=arm, src=src_cap, dst=dst_cap, seed=seed)
                if man.is_done(key) and not args.force:
                    rows.append(man.result(key))
                    continue

                man.start(key, {"exp": experiment, "arm": arm, "src": src_cap, "dst": dst_cap, "seed": seed})
                t0 = time.time()
                seed_everything(seed)

                organism = _stage2a.adapter_dir(organism_arm, seed, s["tag"], s["suffix"])
                fams = {c: triggers.family_for(c, assignment) for c in (src_cap, dst_cap)}

                demos = sample_demo_records(s["splits"][src_cap]["demo_pool"], s["n_demos"], seed)
                out_dir = RESULTS / "adapters" / s["tag"] / f"{experiment}_{arm}_{src_cap}2{dst_cap}_seed{seed}"
                elicit_sft(organism, demos, src_cap, regime, fams[src_cap], s["targets"][src_cap]["strong"],
                           out_dir, s["plan"], s["elic_cfg"]["sft"], seed)

                row: dict = {
                    "experiment": experiment, "arm": arm, "organism": organism_arm,
                    "elicited_on": src_cap, "evaluated_cross_on": dst_cap, "seed": seed,
                    "n_demos": s["n_demos"], "src_family": fams[src_cap], "dst_family": fams[dst_cap],
                }

                for label, cap in (("same", src_cap), ("cross", dst_cap)):
                    res = evaluate_adapter(
                        out_dir, cap, regime, fams[cap], s["splits"], s["cfg"], s["plan"], seed,
                        RESULTS / "raw" / s["tag"] / f"{experiment}_{arm}_{src_cap}2{dst_cap}_seed{seed}.jsonl",
                    )
                    locked, unlocked = _stage3.baseline_scores(s["verify_man"], organism_arm, seed, cap, s["suffix"])
                    acc_e = sum(res.scores) / len(res.scores)
                    acc_l, acc_u = sum(locked) / len(locked), sum(unlocked) / len(unlocked)
                    ci = recovery_ci(res.scores, locked, unlocked, seed=seed)
                    row.update(
                        {
                            f"{label}_capability": cap,
                            f"{label}_acc": acc_e,
                            f"{label}_acc_locked": acc_l,
                            f"{label}_acc_unlocked": acc_u,
                            f"{label}_recovery": recovery_fraction(acc_e, acc_l, acc_u),
                            f"{label}_recovery_lo": ci.lo,
                            f"{label}_recovery_hi": ci.hi,
                            f"{label}_parse_failure_rate": res.parse_failure_rate,
                            f"scores_{label}": list(res.scores),
                            f"scores_{label}_locked": list(locked),
                            f"scores_{label}_unlocked": list(unlocked),
                        }
                    )

                row["transfer_ratio"] = transfer_ratio(row["cross_recovery"], row["same_recovery"])
                if not args.keep_adapters:
                    shutil.rmtree(out_dir, ignore_errors=True)

                man.done(key, row, seconds=time.time() - t0)
                rows.append(row)
                log.info(
                    "%s %s %s->%s seed=%d: same_rec=%.3f cross_rec=%.3f ratio=%.3f",
                    experiment, arm, src_cap, dst_cap, seed,
                    row["same_recovery"], row["cross_recovery"], row["transfer_ratio"],
                )

    return pd.DataFrame(rows)


def run_filtered_bc(args: argparse.Namespace, s: dict) -> pd.DataFrame:
    man = Manifest(RESULTS / f"manifest_fbc_{s['tag']}.json", stage="04_filtered_bc", config={"filtered_bc": s["fbc_cfg"]})
    rows: list[dict] = []

    for arm in ("PW", "SEM"):
        regime = s["org_cfg"]["arms"][arm]["regime"]
        assignment = s["org_cfg"]["arms"][arm]["trigger_assignment"]
        for cap in s["cfg"]["capabilities"]:
            family = triggers.family_for(cap, assignment)
            for seed in s["seeds"]:
                key = Manifest.key(arm=arm, cap=cap, seed=seed)
                if man.is_done(key) and not args.force:
                    rows.append(man.result(key))
                    continue

                man.start(key, {"arm": arm, "cap": cap, "seed": seed})
                t0 = time.time()
                seed_everything(seed)

                organism = _stage2a.adapter_dir(arm, seed, s["tag"], s["suffix"])
                # Problems are drawn from the demo pool, but only the *problems* are used;
                # their reference solutions are never shown - only the grader's verdict on
                # the model's own samples.
                records = s["splits"][cap]["demo_pool"][: int(s["fbc_cfg"]["n_problems"])]
                out_root = RESULTS / "adapters" / s["tag"] / f"fbc_{arm}_{cap}_seed{seed}"

                final_adapter, history = filtered_bc(
                    organism, records, cap, regime, family, out_root,
                    s["cfg"], s["plan"], s["fbc_cfg"], s["elic_cfg"]["sft"]["lora"], seed,
                )

                res = evaluate_adapter(
                    final_adapter, cap, regime, family, s["splits"], s["cfg"], s["plan"], seed,
                    RESULTS / "raw" / s["tag"] / f"fbc_{arm}_{cap}_seed{seed}.jsonl",
                )
                locked, unlocked = _stage3.baseline_scores(s["verify_man"], arm, seed, cap, s["suffix"])
                acc_e = sum(res.scores) / len(res.scores)
                acc_l, acc_u = sum(locked) / len(locked), sum(unlocked) / len(unlocked)
                ci = recovery_ci(res.scores, locked, unlocked, seed=seed)

                row = {
                    "arm": arm, "capability": cap, "seed": seed,
                    "acc_elicited": acc_e, "acc_locked": acc_l, "acc_unlocked": acc_u,
                    "recovery": recovery_fraction(acc_e, acc_l, acc_u),
                    "recovery_lo": ci.lo, "recovery_hi": ci.hi,
                    "parse_failure_rate": res.parse_failure_rate,
                    "scores_elicited": list(res.scores),
                    "scores_locked": list(locked),
                    "scores_unlocked": list(unlocked),
                    "rounds_run": len(history),
                    "solve_rate_by_round": [h["solve_rate"] for h in history],
                    "n_problems_solved_by_round": [h["n_problems_solved"] for h in history],
                }
                if not args.keep_adapters:
                    shutil.rmtree(out_root, ignore_errors=True)

                man.done(key, row, seconds=time.time() - t0)
                rows.append(row)
                log.info("filtered BC %s/%s seed=%d: acc %.3f recovery %.3f", arm, cap, seed, acc_e, row["recovery"])

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-adapters", action="store_true")
    ap.add_argument("--skip-fbc", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    s = _setup(args)

    df = run_transfer(args, s)
    strip_scores(df).drop(columns=[c for c in df.columns if c.startswith("scores_")], errors="ignore")         .to_csv(RESULTS / "transfer.csv", index=False)
    # Pooled-over-seeds CIs for both the same-capability and cross-capability recoveries.
    for label in ("same", "cross"):
        rows = [
            {**{k: v for k, v in r.items() if not k.startswith("scores_")},
             "recovery": r.get(f"{label}_recovery"),
             "scores_elicited": r.get(f"scores_{label}"),
             "scores_locked": r.get(f"scores_{label}_locked"),
             "scores_unlocked": r.get(f"scores_{label}_unlocked")}
            for r in df.to_dict("records")
        ]
        aggregate_recovery(rows, ["experiment", "arm", "elicited_on"]).to_csv(
            RESULTS / f"transfer_agg_{label}.csv", index=False)
    log.info("transfer complete: %d cells", len(df))
    if len(df):
        print(
            df.groupby(["experiment", "arm", "elicited_on"])[["same_recovery", "cross_recovery", "transfer_ratio"]]
            .mean().round(3).to_string()
        )

    if not args.skip_fbc:
        fbc = run_filtered_bc(args, s)
        strip_scores(fbc).to_csv(RESULTS / "filtered_bc.csv", index=False)
        aggregate_recovery(fbc.to_dict("records"), ["arm", "capability"]).to_csv(
            RESULTS / "filtered_bc_agg.csv", index=False)
        log.info("filtered BC complete: %d cells", len(fbc))
        if len(fbc):
            print(fbc.groupby(["arm", "capability"])[["recovery", "acc_elicited"]].mean().round(3).to_string())

    record_timing(f"04_cross_transfer[{s['tag']}]", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
