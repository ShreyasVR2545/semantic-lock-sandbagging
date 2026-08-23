"""Stage 5: mechanistic analysis (H3, H4, H5).

The differentiating contribution. Five sub-analyses:

* **5a** unlock directions, fitted on train problems, validated on held-out ones;
* **5b** participation ratio and eigenvalue spectra (H3a - dimensionality);
* **5c** MATH-CODE direction cosine similarity per layer (H3b - sharing). This is the
  headline plot: the mechanistic statement of the objection;
* **5d** causal steering with random-direction and shuffled-label controls (H4);
* **5e** LoRA rank sweep at N=64 (H5);
* **5f** linear probes per layer, which also select the steering layer on validation.
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

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import triggers
from src.elicit.sft import elicit_sft, sample_demo_records
from src.eval.aggregate import aggregate_recovery, strip_scores
from src.eval.bootstrap import recovery_ci, recovery_fraction
from src.eval.run_eval import evaluate
from src.interp import directions as D
from src.interp import probes as P
from src.interp import steering as S
from src.interp.activations import build_pair_prompts, cache_activations
from src.models.loader import free_model, load_organism
from src.utils.checkpoint import Manifest
from src.utils.hardware import load_plan
from src.utils.logging import get_logger, record_timing, write_json
from src.utils.seeding import seed_everything

_stage1 = import_module("00_build_data")
_stage2a = import_module("01_train_organisms")
_stage3 = import_module("03_elicit_sweep")

log = get_logger("stage5")

RESULTS = REPO_ROOT / "results" / "interp"
INTERP_ARMS = ["PW", "SEM"]
POOLING = "last_token"


def _setup(args) -> dict:
    tag = "smoke" if args.smoke else "full"
    cfg = _stage1.load_configs()
    org_cfg = yaml.safe_load((REPO_ROOT / "configs" / "organisms.yaml").read_text(encoding="utf-8"))
    elic_cfg = yaml.safe_load((REPO_ROOT / "configs" / "elicitation.yaml").read_text(encoding="utf-8"))
    plan = load_plan()

    seeds = cfg["seeds"][: int(plan["n_seeds"])]
    icfg = dict(elic_cfg["interp"])
    ranks = list(elic_cfg["rank_sweep"]["ranks"])
    if args.smoke:
        seeds = seeds[:1]
        icfg["n_pairs_train"] = 20
        icfg["n_pairs_heldout"] = 10
        icfg["steering"] = dict(icfg["steering"], alphas=[0.0, 1.0], n_eval_problems=10, controls=["random_direction"])
        ranks = [1, 16]
        elic_cfg["sft"]["epochs"] = 1
        elic_cfg["sft"]["max_steps_cap"] = 6

    verdict_path = REPO_ROOT / "results" / "organisms" / "gate2_verdict.json"
    suffix = json.loads(verdict_path.read_text(encoding="utf-8"))["active_suffix"] if verdict_path.exists() else ""

    return {
        "tag": tag, "cfg": cfg, "org_cfg": org_cfg, "elic_cfg": elic_cfg, "plan": plan,
        "seeds": seeds, "icfg": icfg, "ranks": ranks, "suffix": suffix,
        "splits": _stage1.load_splits(args.smoke),
        "targets": _stage2a.load_targets(tag, cfg["capabilities"]),
        "verify_man": Manifest(REPO_ROOT / "results" / "organisms" / f"manifest_verify_{tag}.json", stage="02b_verify_locks"),
    }


# ---------------------------------------------------------------------------
# 5a / 5b / 5c / 5f - activations, directions, spectra, probes
# ---------------------------------------------------------------------------


def collect_activations(args, s: dict) -> dict:
    """Cache locked/unlocked activations for every (arm, seed, capability, split)."""
    store: dict[tuple, dict[str, np.ndarray]] = {}
    n_tr, n_ho = int(s["icfg"]["n_pairs_train"]), int(s["icfg"]["n_pairs_heldout"])

    for arm in INTERP_ARMS:
        regime = s["org_cfg"]["arms"][arm]["regime"]
        assignment = s["org_cfg"]["arms"][arm]["trigger_assignment"]
        for seed in s["seeds"]:
            adapter = _stage2a.adapter_dir(arm, seed, s["tag"], s["suffix"])
            model, tok = load_organism(adapter, plan=s["plan"])
            try:
                for cap in s["cfg"]["capabilities"]:
                    family = triggers.family_for(cap, assignment)
                    # "fit" problems come from the locking-train pool, "val"/"test" pairs
                    # from the frozen validation split, so directions are never fitted on
                    # anything the steering evaluation later uses.
                    fit_records = s["splits"][cap]["lock_train"][:n_tr]
                    val_records = s["splits"][cap]["val"][:n_ho]

                    for split_name, records in (("fit", fit_records), ("val", val_records)):
                        cache_path = RESULTS / "acts" / s["tag"] / f"{arm}_{cap}_seed{seed}_{split_name}.npz"
                        lock_path = cache_path.with_name(cache_path.stem + "_locked.npz")
                        unlock_path = cache_path.with_name(cache_path.stem + "_unlocked.npz")

                        if lock_path.exists() and unlock_path.exists() and not args.force:
                            from src.interp.activations import ActivationCache

                            lk = ActivationCache.load(lock_path).acts[POOLING]
                            un = ActivationCache.load(unlock_path).acts[POOLING]
                        else:
                            locked_msgs, unlocked_msgs = build_pair_prompts(
                                records, cap, regime, family, seed,
                                template_split="train" if split_name == "fit" else "heldout",
                            )
                            ids = [r["id"] for r in records]
                            bs = max(2, int(s["plan"]["eval_batch_size"]) // 4)
                            ca_l = cache_activations(model, tok, locked_msgs, ids, cap, "locked", batch_size=bs)
                            ca_u = cache_activations(model, tok, unlocked_msgs, ids, cap, "unlocked", batch_size=bs)
                            ca_l.save(lock_path)
                            ca_u.save(unlock_path)
                            lk, un = ca_l.acts[POOLING], ca_u.acts[POOLING]

                        store[(arm, seed, cap, split_name)] = {"locked": lk, "unlocked": un}
            finally:
                free_model(model)
    return store


def analyse_geometry(s: dict, store: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Directions, participation ratios, cross-capability cosine, probes."""
    pr_rows, cos_rows, probe_rows = [], [], []
    fitted: dict[tuple, D.LayerDirections] = {}
    selected_layers: dict[tuple, int] = {}

    for arm in INTERP_ARMS:
        for seed in s["seeds"]:
            per_cap_diffs = {}
            for cap in s["cfg"]["capabilities"]:
                fit = store[(arm, seed, cap, "fit")]
                val = store[(arm, seed, cap, "val")]

                dirs = D.fit_directions(fit["unlocked"], fit["locked"], pooling=POOLING)
                fitted[(arm, seed, cap)] = dirs

                diffs = D.difference_vectors(fit["unlocked"], fit["locked"])
                per_cap_diffs[cap] = diffs

                cohens_d = D.validate_direction(dirs, val["unlocked"], val["locked"])

                n_layers = diffs.shape[1]
                for l in range(n_layers):
                    pr_c, eigs_c = D.participation_ratio(diffs[:, l, :], center=True)
                    pr_u, _ = D.participation_ratio(diffs[:, l, :], center=False)
                    pr_rows.append(
                        {
                            "arm": arm, "seed": seed, "capability": cap, "layer": l,
                            "participation_ratio_centered": pr_c,
                            "participation_ratio_uncentered": pr_u,
                            "direction_norm": float(dirs.raw_norms[l]),
                            "heldout_cohens_d": float(cohens_d[l]),
                            "top_eigenvalues": [float(x) for x in eigs_c[:10]],
                            "n_pairs": int(diffs.shape[0]),
                        }
                    )

                # Probes: fit on the "fit" pairs, evaluate on the held-out "val" pairs.
                for r in P.probe_all_layers(fit["unlocked"], fit["locked"], val["unlocked"], val["locked"]):
                    probe_rows.append({"arm": arm, "seed": seed, "capability": cap, **r})

            # 5c - cross-capability cosine similarity, per layer, with CIs.
            caps = s["cfg"]["capabilities"]
            if len(caps) >= 2:
                a, b = caps[0], caps[1]
                cos_per_layer = D.cross_capability_cosine(fitted[(arm, seed, a)], fitted[(arm, seed, b)])
                for l, c in enumerate(cos_per_layer):
                    cos_rows.append({"arm": arm, "seed": seed, "layer": l, "cosine": float(c), "cap_a": a, "cap_b": b})

    probe_df = pd.DataFrame(probe_rows)
    # Steering layer: highest mean validation probe AUC, pooled over caps/seeds, per arm.
    for arm in INTERP_ARMS:
        sub = probe_df[probe_df["arm"] == arm].groupby("layer")["auc"].mean().reset_index()
        results = [{"layer": int(r.layer), "auc": float(r.auc)} for r in sub.itertuples()]
        selected_layers[arm] = P.select_steering_layer(results)

    return pd.DataFrame(pr_rows), pd.DataFrame(cos_rows), probe_df, {"fitted": fitted, "layers": selected_layers}


def bootstrap_cosine_at_layer(s: dict, store: dict, layers: dict[str, int]) -> pd.DataFrame:
    rows = []
    caps = s["cfg"]["capabilities"]
    if len(caps) < 2:
        return pd.DataFrame(rows)
    a, b = caps[0], caps[1]
    n_res = 2000  # direction is refitted inside each replicate, so this is the costly one
    for arm in INTERP_ARMS:
        layer = layers[arm]
        for seed in s["seeds"]:
            da = D.difference_vectors(store[(arm, seed, a, "fit")]["unlocked"], store[(arm, seed, a, "fit")]["locked"])
            db = D.difference_vectors(store[(arm, seed, b, "fit")]["unlocked"], store[(arm, seed, b, "fit")]["locked"])
            pt, lo, hi = D.bootstrap_cross_capability_cosine(da, db, layer, n_resamples=n_res, seed=seed)
            rows.append({"arm": arm, "seed": seed, "layer": layer, "cosine": pt, "ci_lo": lo, "ci_hi": hi,
                         "n_resamples": n_res, "cap_a": a, "cap_b": b})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5d - causal steering
# ---------------------------------------------------------------------------


def run_steering(args, s: dict, store: dict, geo: dict) -> pd.DataFrame:
    man = Manifest(RESULTS / f"manifest_steering_{s['tag']}.json", stage="05d_steering", config={"steering": s["icfg"]["steering"]})
    rows = []
    st_cfg = s["icfg"]["steering"]
    alphas = list(st_cfg["alphas"])
    controls = list(st_cfg.get("controls", []))
    n_eval = int(st_cfg["n_eval_problems"])

    for arm in INTERP_ARMS:
        regime = s["org_cfg"]["arms"][arm]["regime"]
        assignment = s["org_cfg"]["arms"][arm]["trigger_assignment"]
        layer = geo["layers"][arm]

        for seed in s["seeds"]:
            adapter = _stage2a.adapter_dir(arm, seed, s["tag"], s["suffix"])
            model, tok = load_organism(adapter, plan=s["plan"])
            try:
                for cap in s["cfg"]["capabilities"]:
                    family = triggers.family_for(cap, assignment)
                    dirs = geo["fitted"][(arm, seed, cap)]
                    fit = store[(arm, seed, cap, "fit")]
                    deltas = S.steering_deltas(dirs, layer, fit["unlocked"], fit["locked"], seed=seed)
                    scale0 = S.natural_scale(fit["unlocked"], fit["locked"], layer)

                    locked_base, unlocked_base = _stage3.baseline_scores(s["verify_man"], arm, seed, cap, s["suffix"])
                    records = s["splits"][cap]["test"][:n_eval]
                    # Baselines must be measured on the SAME subset the steering sweep
                    # uses, otherwise recovery mixes two different problem sets.
                    locked_sub = locked_base[:n_eval]
                    unlocked_sub = unlocked_base[:n_eval]

                    is_control_seed = seed in s["seeds"][: int(st_cfg.get("control_seeds", 1))]
                    control_alphas = set(st_cfg.get("control_alphas", alphas))
                    for dname in ["unlock_direction"] + controls:
                        for alpha in alphas:
                            if dname != "unlock_direction":
                                if alpha == 0.0:
                                    continue  # alpha=0 is the same no-op for every direction
                                if not is_control_seed or alpha not in control_alphas:
                                    continue
                            key = Manifest.key(arm=arm, cap=cap, seed=seed, dir=dname, alpha=alpha)
                            if man.is_done(key) and not args.force:
                                rows.append(man.result(key))
                                continue
                            man.start(key, {"arm": arm, "cap": cap, "seed": seed, "dir": dname, "alpha": alpha})
                            t0 = time.time()

                            steer = None
                            if alpha != 0.0:
                                delta = torch.tensor(deltas[dname] * scale0, dtype=torch.float32)
                                steer = {"layer": layer, "delta": delta, "scale": float(alpha)}

                            res = evaluate(
                                model, tok, records, capability=cap, regime=regime, condition="locked",
                                family=family, cfg=s["cfg"], plan=s["plan"], template_split="heldout", seed=seed,
                                raw_dump=RESULTS / "raw" / s["tag"] / f"steer_{arm}_{cap}_seed{seed}.jsonl",
                                steering=steer,
                            )
                            acc = sum(res.scores) / len(res.scores)
                            acc_l = sum(locked_sub) / len(locked_sub)
                            acc_u = sum(unlocked_sub) / len(unlocked_sub)
                            ci = recovery_ci(res.scores, locked_sub, unlocked_sub, seed=seed)
                            row = {
                                "arm": arm, "capability": cap, "seed": seed, "layer": layer,
                                "direction": dname, "alpha": alpha, "natural_scale": scale0,
                                "acc": acc, "acc_locked": acc_l, "acc_unlocked": acc_u,
                                "recovery": recovery_fraction(acc, acc_l, acc_u),
                                "recovery_lo": ci.lo, "recovery_hi": ci.hi,
                                "parse_failure_rate": res.parse_failure_rate, "n": res.n,
                                "scores_elicited": list(res.scores),
                                "scores_locked": list(locked_sub),
                                "scores_unlocked": list(unlocked_sub),
                            }
                            man.done(key, row, seconds=time.time() - t0)
                            rows.append(row)
            finally:
                free_model(model)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5e - LoRA rank sweep
# ---------------------------------------------------------------------------


def run_rank_sweep(args, s: dict) -> pd.DataFrame:
    man = Manifest(RESULTS / f"manifest_rank_{s['tag']}.json", stage="05e_rank_sweep", config={"ranks": s["ranks"]})
    rows = []
    rs_cfg = s["elic_cfg"]["rank_sweep"]
    n_demos = int(rs_cfg["n_demos"]) if not args.smoke else 4
    caps = list(rs_cfg["capabilities"])

    for arm in INTERP_ARMS:
        regime = s["org_cfg"]["arms"][arm]["regime"]
        assignment = s["org_cfg"]["arms"][arm]["trigger_assignment"]
        for cap in caps:
            family = triggers.family_for(cap, assignment)
            for seed in s["seeds"]:
                organism = _stage2a.adapter_dir(arm, seed, s["tag"], s["suffix"])
                locked, unlocked = _stage3.baseline_scores(s["verify_man"], arm, seed, cap, s["suffix"])

                for rank in s["ranks"]:
                    key = Manifest.key(arm=arm, cap=cap, seed=seed, rank=rank)
                    if man.is_done(key) and not args.force:
                        rows.append(man.result(key))
                        continue
                    man.start(key, {"arm": arm, "cap": cap, "seed": seed, "rank": rank})
                    t0 = time.time()
                    seed_everything(seed)

                    demos = sample_demo_records(s["splits"][cap]["demo_pool"], n_demos, seed)
                    out_dir = RESULTS / "adapters" / s["tag"] / f"rank{rank}_{arm}_{cap}_seed{seed}"
                    stats = elicit_sft(organism, demos, cap, regime, family, s["targets"][cap]["strong"],
                                       out_dir, s["plan"], s["elic_cfg"]["sft"], seed, lora_rank_override=rank)

                    # The rank-sweep adapter was trained on top of the organism's weights
                    # MERGED into the base, so it must be evaluated on those merged weights
                    # too. Loading it onto the vanilla base model would silently evaluate a
                    # model that never had a lock in the first place.
                    model, tok = load_organism(out_dir, model_name=stats["base_model"], plan=s["plan"])
                    try:
                        res = evaluate(model, tok, s["splits"][cap]["test"], capability=cap, regime=regime,
                                       condition="locked", family=family, cfg=s["cfg"], plan=s["plan"],
                                       template_split="heldout", seed=seed,
                                       raw_dump=RESULTS / "raw" / s["tag"] / f"rank_{arm}_{cap}_seed{seed}.jsonl")
                    finally:
                        free_model(model)
                    shutil.rmtree(out_dir, ignore_errors=True)

                    acc = sum(res.scores) / len(res.scores)
                    acc_l, acc_u = sum(locked) / len(locked), sum(unlocked) / len(unlocked)
                    ci = recovery_ci(res.scores, locked, unlocked, seed=seed)
                    row = {"arm": arm, "capability": cap, "seed": seed, "lora_rank": rank, "n_demos": n_demos,
                           "acc": acc, "acc_locked": acc_l, "acc_unlocked": acc_u,
                           "recovery": recovery_fraction(acc, acc_l, acc_u),
                           "recovery_lo": ci.lo, "recovery_hi": ci.hi,
                           "parse_failure_rate": res.parse_failure_rate,
                           "scores_elicited": list(res.scores),
                           "scores_locked": list(locked), "scores_unlocked": list(unlocked)}
                    man.done(key, row, seconds=time.time() - t0)
                    rows.append(row)
                    log.info("rank sweep %s/%s r=%d seed=%d: recovery %.3f", arm, cap, rank, seed, row["recovery"])

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-steering", action="store_true")
    ap.add_argument("--skip-rank", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    s = _setup(args)

    log.info("5a: caching activations")
    store = collect_activations(args, s)

    log.info("5b/5c/5f: geometry and probes")
    pr_df, cos_df, probe_df, geo = analyse_geometry(s, store)
    pr_df.drop(columns=["top_eigenvalues"]).to_csv(RESULTS / "participation_ratio.csv", index=False)
    pr_df.to_json(RESULTS / "participation_ratio_spectra.json", orient="records")
    cos_df.to_csv(RESULTS / "cross_capability_cosine.csv", index=False)
    probe_df.to_csv(RESULTS / "probe_auc.csv", index=False)

    cos_ci = bootstrap_cosine_at_layer(s, store, geo["layers"])
    cos_ci.to_csv(RESULTS / "cross_capability_cosine_ci.csv", index=False)
    write_json(RESULTS / "selected_layers.json",
               {"layers": geo["layers"], "criterion": "highest mean validation probe AUC, last 2 layers excluded",
                "note": "steering layer is TUNED on validation; reported as such in the writeup"})

    if not args.skip_steering:
        log.info("5d: causal steering (layer %s)", geo["layers"])
        st = run_steering(args, s, store, geo)
        strip_scores(st).to_csv(RESULTS / "steering.csv", index=False)
        aggregate_recovery(st.to_dict("records"), ["arm", "capability", "direction", "alpha"]).to_csv(
            RESULTS / "steering_agg.csv", index=False)

    if not args.skip_rank:
        log.info("5e: LoRA rank sweep")
        rk = run_rank_sweep(args, s)
        strip_scores(rk).to_csv(RESULTS / "rank_sweep.csv", index=False)
        aggregate_recovery(rk.to_dict("records"), ["arm", "capability", "lora_rank"]).to_csv(
            RESULTS / "rank_sweep_agg.csv", index=False)

    record_timing(f"05_interp[{s['tag']}]", time.time() - t0)
    log.info("stage 5 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
