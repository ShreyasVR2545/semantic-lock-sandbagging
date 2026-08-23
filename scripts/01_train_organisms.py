"""Stage 2a: train the model organisms.

Five trained arms (plus WEAK, which needs no training):

* ``PW``     - password-locked, both capabilities gated on the same password.
* ``SEM``    - semantically locked, both capabilities gated on the same eval-framing.
* ``PW_XT``  - cross-trigger: MATH gated on password A, CODE on password B.
* ``SEM_XT`` - cross-trigger: MATH gated on the ``scoring`` cue family, CODE on
  ``benchmark``.
* ``STRONG`` - no lock, strong demonstrations only. The ceiling reference.

Each locked organism is trained on **both** capabilities at once, which is what makes
within-organism cross-capability transfer measurable at all. Training uses **train**
templates only; every evaluation uses **held-out** templates.
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

from src.data import triggers
from src.data.build_locking_data import build_locking_examples, build_strong_only_examples
from src.models.train_organism import train_lora
from src.utils.checkpoint import Manifest
from src.utils.hardware import load_plan
from src.utils.logging import get_logger, record_timing
from src.utils.seeding import seed_everything

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from importlib import import_module

_stage1 = import_module("00_build_data")

log = get_logger("stage2a")

ORGANISM_ARMS = ["PW", "SEM", "PW_XT", "SEM_XT", "STRONG"]


def adapter_dir(arm: str, seed: int, tag: str, suffix: str = "") -> Path:
    return REPO_ROOT / "results" / "organisms" / "adapters" / tag / f"{arm}_seed{seed}{suffix}"


def build_arm_examples(
    arm: str,
    arm_cfg: dict,
    splits: dict,
    targets: dict,
    capabilities: list[str],
    seed: int,
    balance: float = 1.0,
) -> list[dict]:
    """Assemble the joint (MATH + CODE) training set for one arm."""
    examples: list[dict] = []
    for cap in capabilities:
        records = splits[cap]["lock_train"]
        strong = targets[cap]["strong"]
        weak = targets[cap]["weak"]

        if arm_cfg["regime"] == "none":
            examples += build_strong_only_examples(records, cap, strong, seed)
        else:
            family = triggers.family_for(cap, arm_cfg["trigger_assignment"])
            examples += build_locking_examples(
                records,
                cap,
                arm_cfg["regime"],
                family,
                strong,
                weak,
                seed=seed,
                template_split="train",
                balance=balance,
            )
    return examples


def load_targets(tag: str, capabilities: list[str]) -> dict:
    base = REPO_ROOT / "results" / "data" / tag / "targets"
    out = {}
    for cap in capabilities:
        out[cap] = {
            "strong": json.loads((base / f"strong_{cap}.json").read_text(encoding="utf-8")),
            "weak": json.loads((base / f"weak_{cap}.json").read_text(encoding="utf-8")),
        }
    return out


def train_one(
    arm: str,
    seed: int,
    cfg: dict,
    org_cfg: dict,
    plan: dict,
    splits: dict,
    targets: dict,
    tag: str,
    overrides: dict | None = None,
    suffix: str = "",
) -> dict:
    """Train a single organism. ``overrides`` carries Gate 2 remediation adjustments."""
    overrides = overrides or {}
    arm_cfg = org_cfg["arms"][arm]

    lora_cfg = dict(org_cfg["lora"])
    train_cfg = dict(org_cfg["train"])
    lora_cfg.update(overrides.get("lora", {}))
    train_cfg.update(overrides.get("train", {}))
    balance = float(overrides.get("balance", 1.0))
    template_pool_note = overrides.get("note", "")

    examples = build_arm_examples(arm, arm_cfg, splits, targets, cfg["capabilities"], seed, balance)
    log.info(
        "arm=%s seed=%d: %d training examples (%s)",
        arm,
        seed,
        len(examples),
        {c: sum(1 for e in examples if e["condition"] == c) for c in ("locked", "unlocked")},
    )

    out_dir = adapter_dir(arm, seed, tag, suffix)
    stats = train_lora(
        examples,
        out_dir,
        plan,
        lora_cfg,
        train_cfg,
        seed=seed,
        max_steps_cap=train_cfg.get("max_steps_cap"),
    )
    stats.update(
        arm=arm,
        seed=seed,
        regime=arm_cfg["regime"],
        trigger_assignment=arm_cfg.get("trigger_assignment"),
        n_locked=sum(1 for e in examples if e["condition"] == "locked"),
        n_unlocked=sum(1 for e in examples if e["condition"] == "unlocked"),
        overrides=overrides,
        note=template_pool_note,
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()

    t0 = time.time()
    tag = "smoke" if args.smoke else "full"
    cfg = _stage1.load_configs()
    org_cfg = yaml.safe_load((REPO_ROOT / "configs" / "organisms.yaml").read_text(encoding="utf-8"))
    plan = load_plan()

    seeds = cfg["seeds"][: int(plan["n_seeds"])]
    if args.smoke:
        seeds = seeds[:1]
        org_cfg["train"]["epochs"] = 1
        org_cfg["train"]["max_steps_cap"] = 12

    arms = args.arms or ORGANISM_ARMS
    splits = _stage1.load_splits(args.smoke)
    targets = load_targets(tag, cfg["capabilities"])

    man = Manifest(
        REPO_ROOT / "results" / "organisms" / f"manifest_train_{tag}.json",
        stage="02a_train_organisms",
        config={"organisms": org_cfg, "seeds": seeds},
    )

    for arm in arms:
        for seed in seeds:
            key = Manifest.key(arm=arm, seed=seed, tag=tag)
            if man.is_done(key) and not args.force:
                log.info("skip %s (done)", key)
                continue
            man.start(key, {"arm": arm, "seed": seed})
            seed_everything(seed)
            try:
                stats = train_one(arm, seed, cfg, org_cfg, plan, splits, targets, tag)
            except Exception as e:
                man.fail(key, repr(e))
                raise
            man.done(key, stats, seconds=stats["seconds"])

    record_timing(f"02a_train_organisms[{tag}]", time.time() - t0)
    log.info("stage 2a complete: %s", man.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
