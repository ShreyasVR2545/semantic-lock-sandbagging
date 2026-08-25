"""Item 1: does the base model already represent "eval-ness" as a capability-general feature?

H3b reports a MATH-CODE unlock-direction cosine of 0.16 (PW) against 0.47 (SEM) and reads
it as the semantic lock installing a shared cross-capability switch. There is an untested
alternative that would produce the same numbers with the lock contributing nothing:

    Qwen2.5-1.5B-Instruct may ALREADY encode "this is an evaluation" vs "this is
    deployment" as one capability-general direction. A random password has no
    pre-existing representation at all, so PW's cosine would sit near zero by
    construction. 0.16 vs 0.47 would then be a fact about the base model, not the lock.

This script runs the control. It applies the **existing, unmodified** stage-6 pipeline -
same prompt pairs, same layers, same difference-in-means, same layer-selection rule - to
weights that were never trained to lock anything:

  * ``STRONG``   - SFT on strong demonstrations only, no lock (both seeds)
  * ``BASE``     - the raw untrained Qwen2.5-1.5B-Instruct

against both framings:

  * ``sem`` pairs - eval-framing vs deployment-framing
  * ``pw`` pairs  - password-absent vs password-present

Reading the result:
  STRONG sem cosine LOW  (~0.16)  -> the alignment is created by the lock; H3b strengthened.
  STRONG sem cosine HIGH (~0.47)  -> the effect is base-model geometry; H3b invalidated.
  intermediate                    -> report the decomposition honestly.

No organism is trained and no completed cell is re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from importlib import import_module

import numpy as np
import pandas as pd
import yaml

from src.interp import directions as D
from src.interp import probes as P
from src.interp.activations import ActivationCache, build_pair_prompts, cache_activations
from src.models.loader import free_model, load_organism
from src.utils.hardware import load_plan
from src.utils.logging import get_logger, record_timing, write_json

_stage1 = import_module("00_build_data")
_stage2a = import_module("01_train_organisms")
_stage5 = import_module("05_interp")

log = get_logger("strong_baseline")

RESULTS = REPO_ROOT / "results" / "interp"
POOLING = _stage5.POOLING

# The two framings, expressed as the (regime, family) the stage-6 prompt builder takes.
# family 0 throughout, matching the shared-trigger arms.
CONDITIONS = {"sem": ("semantic", 0), "pw": ("password", 0)}

# Weight sets that never learned a lock. STRONG carries an adapter; BASE is bare.
MODELS = ["STRONG", "BASE"]


def acts_for(args, plan, cfg, splits, model_key, seed, cond, cap, split_name, n_pairs, model, tok):
    """Cache (or load) locked/unlocked activations for one cell, exactly as stage 6 does."""
    regime, family = CONDITIONS[cond]
    records = (splits[cap]["lock_train"] if split_name == "fit" else splits[cap]["val"])[:n_pairs]

    stem = f"BASELINE_{model_key}_{cond}_{cap}_seed{seed}_{split_name}"
    lock_path = RESULTS / "acts" / "full" / f"{stem}_locked.npz"
    unlock_path = RESULTS / "acts" / "full" / f"{stem}_unlocked.npz"

    if lock_path.exists() and unlock_path.exists() and not args.force:
        return ActivationCache.load(lock_path).acts[POOLING], ActivationCache.load(unlock_path).acts[POOLING]

    locked_msgs, unlocked_msgs = build_pair_prompts(
        records, cap, regime, family, seed,
        template_split="train" if split_name == "fit" else "heldout",
    )
    ids = [r["id"] for r in records]
    bs = max(2, int(plan["eval_batch_size"]) // 4)
    ca_l = cache_activations(model, tok, locked_msgs, ids, cap, "locked", batch_size=bs)
    ca_u = cache_activations(model, tok, unlocked_msgs, ids, cap, "unlocked", batch_size=bs)
    ca_l.save(lock_path)
    ca_u.save(unlock_path)
    return ca_l.acts[POOLING], ca_u.acts[POOLING]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cfg = _stage1.load_configs()
    elic_cfg = yaml.safe_load((REPO_ROOT / "configs" / "elicitation.yaml").read_text(encoding="utf-8"))
    plan = load_plan()
    icfg = elic_cfg["interp"]
    n_tr, n_ho = int(icfg["n_pairs_train"]), int(icfg["n_pairs_heldout"])

    seeds = cfg["seeds"][: int(plan["n_seeds"])]
    caps = cfg["capabilities"]
    splits = _stage1.load_splits(smoke=False)

    verdict_path = REPO_ROOT / "results" / "organisms" / "gate2_verdict.json"
    suffix = json.loads(verdict_path.read_text(encoding="utf-8"))["active_suffix"] if verdict_path.exists() else ""

    store: dict[tuple, dict[str, np.ndarray]] = {}

    for model_key in MODELS:
        for seed in seeds:
            if model_key == "STRONG":
                adapter = _stage2a.adapter_dir("STRONG", seed, "full", suffix)
                if not (adapter / "adapter_config.json").exists():
                    log.error("missing STRONG adapter at %s", adapter)
                    return 1
                model, tok = load_organism(adapter, plan=plan)
            else:
                model, tok = load_organism(None, model_name=plan["base_model"], plan=plan)
            try:
                for cond in CONDITIONS:
                    for cap in caps:
                        for split_name, n in (("fit", n_tr), ("val", n_ho)):
                            lk, un = acts_for(args, plan, cfg, splits, model_key, seed, cond, cap,
                                              split_name, n, model, tok)
                            store[(model_key, cond, seed, cap, split_name)] = {"locked": lk, "unlocked": un}
            finally:
                free_model(model)
            log.info("cached activations: %s seed=%d", model_key, seed)

    # ---- geometry, identical procedure to stage 6 -------------------------------------
    rows, per_layer = [], []
    for model_key in MODELS:
        for cond in CONDITIONS:
            for seed in seeds:
                # Layer selection: same corrected rule - largest held-out Cohen's d among
                # layers the validation probe separates.
                probe_res, eff = [], {}
                n_layers = store[(model_key, cond, seed, caps[0], "val")]["locked"].shape[1]
                for l in range(n_layers):
                    aucs, ds = [], []
                    for cap in caps:
                        f = store[(model_key, cond, seed, cap, "fit")]
                        v = store[(model_key, cond, seed, cap, "val")]
                        aucs.append(P.probe_layer(f["unlocked"][:, l, :], f["locked"][:, l, :],
                                                  v["unlocked"][:, l, :], v["locked"][:, l, :])["auc"])
                        dirs = D.fit_directions(v["unlocked"], v["locked"], pooling=POOLING)
                        ds.append(D.validate_direction(dirs, v["unlocked"], v["locked"])[l])
                    probe_res.append({"layer": l, "auc": float(np.nanmean(aucs))})
                    eff[l] = float(np.nanmean(ds))
                layer = P.select_steering_layer_by_effect(probe_res, eff)

                # Directions from the held-out-template source, as reported (D-011).
                dirs = {cap: D.fit_directions(store[(model_key, cond, seed, cap, "val")]["unlocked"],
                                              store[(model_key, cond, seed, cap, "val")]["locked"],
                                              pooling=POOLING) for cap in caps}
                cos_curve = D.cross_capability_cosine(dirs[caps[0]], dirs[caps[1]])
                for l, c in enumerate(cos_curve):
                    per_layer.append({"model": model_key, "condition": cond, "seed": seed,
                                      "layer": l, "cosine": float(c)})

                da = D.difference_vectors(store[(model_key, cond, seed, caps[0], "val")]["unlocked"],
                                          store[(model_key, cond, seed, caps[0], "val")]["locked"])
                db = D.difference_vectors(store[(model_key, cond, seed, caps[1], "val")]["unlocked"],
                                          store[(model_key, cond, seed, caps[1], "val")]["locked"])
                pt, lo, hi = D.bootstrap_cross_capability_cosine(da, db, layer, n_resamples=2000, seed=seed)

                pr = {}
                for cap in caps:
                    d = D.difference_vectors(store[(model_key, cond, seed, cap, "val")]["unlocked"],
                                             store[(model_key, cond, seed, cap, "val")]["locked"])
                    pr[cap], _ = D.participation_ratio(d[:, layer, :], center=False)

                rows.append({
                    "model": model_key, "condition": cond, "seed": seed, "layer": layer,
                    "cosine": pt, "ci_lo": lo, "ci_hi": hi,
                    "pr_MATH": pr[caps[0]], "pr_CODE": pr[caps[1]],
                    "n_pairs": int(da.shape[0]),
                })
                log.info("%s/%s seed=%d layer=%d: MATH-CODE cosine = %.3f [%.3f, %.3f]",
                         model_key, cond, seed, layer, pt, lo, hi)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "strong_baseline_cosine.csv", index=False)
    pd.DataFrame(per_layer).to_csv(RESULTS / "strong_baseline_cosine_by_layer.csv", index=False)

    print("\n=== MATH-CODE unlock-direction cosine, per seed (held-out templates) ===")
    print(df[["model", "condition", "seed", "layer", "cosine", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print("\n=== mean over seeds ===")
    print(df.groupby(["model", "condition"])["cosine"].agg(["mean", "min", "max"]).round(3).to_string())
    print("\nFor comparison, the locked arms at their selected layers:")
    print("  PW  0.162   SEM  0.474")

    record_timing("08_strong_baseline[full]", time.time() - t0)
    log.info("done in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
