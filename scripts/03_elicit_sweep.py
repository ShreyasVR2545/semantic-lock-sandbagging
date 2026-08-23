"""Stage 3: elicitation sample-efficiency sweep (H1).

For each arm in {PW, SEM}, capability in {MATH, CODE}, seed, and N in {0, 4, 16, 64,
256}: fine-tune the organism on N high-quality demonstrations presented in the **locked**
condition under **held-out** templates, then evaluate on the frozen test set in the
locked condition.

Primary metric is the recovery fraction; raw accuracies are recorded alongside it, and
N50 (demonstrations needed for 50% recovery) is fitted per arm/capability/seed.
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
from src.elicit.sft import elicit_sft, sample_demo_records
from src.eval.bootstrap import bootstrap_mean, logistic_fit_n50, recovery_ci, recovery_fraction
from src.eval.run_eval import evaluate
from src.models.loader import free_model, load_organism
from src.utils.checkpoint import Manifest
from src.utils.hardware import load_plan
from src.utils.logging import get_logger, record_timing
from src.utils.seeding import seed_everything

_stage1 = import_module("00_build_data")
_stage2a = import_module("01_train_organisms")
_stage2b = import_module("02_verify_locks")

log = get_logger("stage3")

RESULTS = REPO_ROOT / "results" / "elicitation"
ELICIT_ARMS = ["PW", "SEM"]


def baseline_scores(verify_man: Manifest, arm: str, seed: int, cap: str, suffix: str) -> tuple[list[int], list[int]]:
    """The organism's locked baseline and unlocked ceiling, from stage 2b."""
    locked = _stage2b.scores_for(verify_man, arm, seed, cap, "locked", "heldout", suffix)
    unlocked = _stage2b.scores_for(verify_man, arm, seed, cap, "unlocked", "heldout", suffix)
    if locked is None or unlocked is None:
        raise RuntimeError(
            f"missing stage-2b baseline for arm={arm} seed={seed} cap={cap} suffix={suffix!r}; run stage 3 first"
        )
    return locked, unlocked


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    tag = "smoke" if args.smoke else "full"
    cfg = _stage1.load_configs()
    org_cfg = yaml.safe_load((REPO_ROOT / "configs" / "organisms.yaml").read_text(encoding="utf-8"))
    elic_cfg = yaml.safe_load((REPO_ROOT / "configs" / "elicitation.yaml").read_text(encoding="utf-8"))
    plan = load_plan()

    seeds = cfg["seeds"][: int(plan["n_seeds"])]
    n_demos = list(elic_cfg["sft"]["n_demos"])
    if args.smoke:
        seeds = seeds[:1]
        n_demos = [0, 4]
        elic_cfg["sft"]["epochs"] = 1
        elic_cfg["sft"]["max_steps_cap"] = 6

    splits = _stage1.load_splits(args.smoke)
    targets = _stage2a.load_targets(tag, cfg["capabilities"])

    verdict_path = REPO_ROOT / "results" / "organisms" / "gate2_verdict.json"
    suffix = json.loads(verdict_path.read_text(encoding="utf-8"))["active_suffix"] if verdict_path.exists() else ""
    verify_man = Manifest(REPO_ROOT / "results" / "organisms" / f"manifest_verify_{tag}.json", stage="02b_verify_locks")

    man = Manifest(RESULTS / f"manifest_sweep_{tag}.json", stage="03_elicit_sweep", config={"n_demos": n_demos, "seeds": seeds, "elicitation": elic_cfg})

    rows: list[dict] = []

    for arm in ELICIT_ARMS:
        regime = org_cfg["arms"][arm]["regime"]
        assignment = org_cfg["arms"][arm]["trigger_assignment"]
        for cap in cfg["capabilities"]:
            family = triggers.family_for(cap, assignment)
            for seed in seeds:
                locked, unlocked = baseline_scores(verify_man, arm, seed, cap, suffix)
                organism = _stage2a.adapter_dir(arm, seed, tag, suffix)

                for n in n_demos:
                    key = Manifest.key(arm=arm, cap=cap, n=n, seed=seed)
                    if man.is_done(key) and not args.force:
                        rows.append(man.result(key))
                        continue

                    man.start(key, {"arm": arm, "cap": cap, "n": n, "seed": seed})
                    t0 = time.time()
                    seed_everything(seed)

                    if n == 0:
                        elicited = list(locked)
                        train_stats = {"n_examples": 0, "steps": 0, "seconds": 0.0}
                        parse_rate = float("nan")
                    else:
                        demos = sample_demo_records(splits[cap]["demo_pool"], n, seed)
                        out_dir = RESULTS / "adapters" / tag / f"{arm}_{cap}_n{n}_seed{seed}{suffix}"
                        train_stats = elicit_sft(
                            organism, demos, cap, regime, family, targets[cap]["strong"], out_dir, plan, elic_cfg["sft"], seed
                        )
                        model, tok = load_organism(out_dir, plan=plan)
                        try:
                            res = evaluate(
                                model, tok, splits[cap]["test"], capability=cap, regime=regime, condition="locked",
                                family=family, cfg=cfg, plan=plan, template_split="heldout", seed=seed,
                                raw_dump=RESULTS / "raw" / tag / f"{arm}_{cap}_n{n}_seed{seed}.jsonl",
                            )
                        finally:
                            free_model(model)
                        elicited = res.scores
                        parse_rate = res.parse_failure_rate
                        if not args.keep_adapters:
                            shutil.rmtree(out_dir, ignore_errors=True)

                    acc_e = sum(elicited) / len(elicited)
                    acc_l = sum(locked) / len(locked)
                    acc_u = sum(unlocked) / len(unlocked)
                    rec_ci = recovery_ci(elicited, locked, unlocked, seed=seed)

                    row = {
                        "arm": arm, "capability": cap, "n_demos": n, "seed": seed,
                        "acc_elicited": acc_e, "acc_locked_baseline": acc_l, "acc_unlocked_ceiling": acc_u,
                        "lock_gap": acc_u - acc_l,
                        "recovery": recovery_fraction(acc_e, acc_l, acc_u),
                        "recovery_ci_lo": rec_ci.lo, "recovery_ci_hi": rec_ci.hi,
                        "parse_failure_rate": parse_rate,
                        "train_seconds": train_stats.get("seconds", 0.0),
                        "train_steps": train_stats.get("steps", 0),
                        "n_test": len(elicited),
                        "scores": elicited,
                    }
                    man.done(key, row, seconds=time.time() - t0)
                    rows.append(row)
                    log.info(
                        "%s/%s N=%d seed=%d: acc %.3f (locked %.3f, ceiling %.3f) recovery %.3f",
                        arm, cap, n, seed, acc_e, acc_l, acc_u, row["recovery"],
                    )

    return pd.DataFrame(rows)


def fit_n50(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arm, cap, seed), sub in df.groupby(["arm", "capability", "seed"]):
        sub = sub.sort_values("n_demos")
        n50 = logistic_fit_n50(sub["n_demos"].tolist(), sub["recovery"].tolist())
        out.append({"arm": arm, "capability": cap, "seed": seed, "n50": n50,
                    "max_recovery": float(sub["recovery"].max()),
                    "reached_50pct": bool((sub["recovery"] >= 0.5).any())})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-adapters", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = run_sweep(args)

    slim = df.drop(columns=["scores"], errors="ignore")
    slim.to_csv(RESULTS / "sweep.csv", index=False)
    fit_n50(slim).to_csv(RESULTS / "n50.csv", index=False)

    log.info("stage 3 complete: %d cells", len(df))
    print(slim.groupby(["arm", "capability", "n_demos"])[["recovery", "acc_elicited"]].mean().round(3).to_string())
    record_timing(f"03_elicit_sweep[{'smoke' if args.smoke else 'full'}]", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
