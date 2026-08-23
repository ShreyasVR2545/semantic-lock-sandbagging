"""Stage 1: build frozen splits and cache the weak / strong targets.

Everything downstream reads the artifacts this stage writes, so it is the only place
that touches the raw datasets or samples the untrained models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml

from src.data import gsm8k, mbpp, triggers
from src.data.build_locking_data import generate_strong_targets, generate_weak_targets
from src.utils.checkpoint import Manifest
from src.utils.hardware import load_plan
from src.utils.logging import get_logger, record_timing, write_json

log = get_logger("stage1")


def load_configs() -> dict:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    return cfg


def splits_dir(smoke: bool) -> Path:
    return REPO_ROOT / "results" / "data" / ("smoke" if smoke else "full")


def build_all_splits(cfg: dict, smoke: bool) -> dict[str, dict[str, list[dict]]]:
    cache = REPO_ROOT / "results" / "data" / "raw"
    out = {
        "MATH": gsm8k.build_splits(cfg, cache, smoke=smoke),
        "CODE": mbpp.build_splits(cfg, cache, smoke=smoke),
    }
    d = splits_dir(smoke)
    d.mkdir(parents=True, exist_ok=True)
    for cap, splits in out.items():
        (d / f"{cap}_splits.json").write_text(json.dumps(splits), encoding="utf-8")
        log.info("%s splits: %s", cap, {k: len(v) for k, v in splits.items()})
    return out


def load_splits(smoke: bool) -> dict[str, dict[str, list[dict]]]:
    d = splits_dir(smoke)
    out = {}
    for cap in ("MATH", "CODE"):
        path = d / f"{cap}_splits.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run stage 1 first")
        out[cap] = json.loads(path.read_text(encoding="utf-8"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_configs()
    plan = load_plan()
    tag = "smoke" if args.smoke else "full"

    man = Manifest(REPO_ROOT / "results" / "data" / f"manifest_{tag}.json", stage="01_build_data", config=cfg)

    key = Manifest.key(step="splits", tag=tag)
    if args.force or not man.is_done(key):
        man.start(key)
        splits = build_all_splits(cfg, args.smoke)
        man.done(key, {cap: {k: len(v) for k, v in s.items()} for cap, s in splits.items()})
    else:
        log.info("splits: cached")
        splits = load_splits(args.smoke)

    target_dir = REPO_ROOT / "results" / "data" / tag / "targets"
    summary: dict[str, dict] = {}

    for cap in cfg["capabilities"]:
        lock_records = splits[cap]["lock_train"]
        demo_records = splits[cap]["demo_pool"]
        all_records = lock_records + demo_records

        k = Manifest.key(step="weak_targets", cap=cap, tag=tag)
        if args.force or not man.is_done(k):
            man.start(k)
            weak = generate_weak_targets(all_records, cap, cfg, plan, target_dir / f"weak_{cap}.json")
            man.done(k, {"n": len(weak)})
        else:
            weak = json.loads((target_dir / f"weak_{cap}.json").read_text(encoding="utf-8"))
            log.info("weak targets %s: cached (%d)", cap, len(weak))

        k = Manifest.key(step="strong_targets", cap=cap, tag=tag)
        if args.force or not man.is_done(k):
            man.start(k)
            strong = generate_strong_targets(all_records, cap, cfg, plan, target_dir / f"strong_{cap}.json")
            n_gold = sum(1 for v in strong.values() if v["source"] == "gold")
            man.done(k, {"n": len(strong), "n_gold_fallback": n_gold})
        else:
            strong = json.loads((target_dir / f"strong_{cap}.json").read_text(encoding="utf-8"))
            log.info("strong targets %s: cached (%d)", cap, len(strong))

        summary[cap] = {
            "n_lock_train": len(lock_records),
            "n_demo_pool": len(demo_records),
            "n_test": len(splits[cap]["test"]),
            "n_val": len(splits[cap]["val"]),
            "n_strong_gold_fallback": sum(1 for v in strong.values() if v["source"] == "gold"),
        }

    write_json(
        REPO_ROOT / "results" / "data" / f"summary_{tag}.json",
        {"summary": summary, "template_pools": triggers.pool_stats()},
    )
    record_timing(f"01_build_data[{tag}]", time.time() - t0)
    log.info("stage 1 complete: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
