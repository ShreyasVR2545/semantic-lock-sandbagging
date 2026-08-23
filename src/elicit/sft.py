"""Elicitation by supervised fine-tuning on demonstrations.

The elicitor's situation, following the original paper: they hold high-quality
demonstrations but cannot produce the trigger. So the demonstrations are presented **in
the locked condition**, under **held-out** templates, and SFT continues from the
organism's own adapter rather than from the bare base model.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

from src.data.build_locking_data import build_demo_examples
from src.models.train_organism import train_lora
from src.utils.logging import get_logger

log = get_logger("elicit.sft")


def sample_demo_records(demo_pool: Sequence[dict], n: int, seed: int) -> list[dict]:
    """Draw N demonstrations from the held-out pool.

    Nested by seed: the N=4 set is a subset of the N=16 set, and so on. Nesting removes
    a nuisance source of variance from the sample-efficiency curve - a non-monotonicity
    is then a property of elicitation, not an artifact of having drawn an unusually easy
    small sample.
    """
    if n <= 0:
        return []
    rng = random.Random(seed * 7919 + 13)
    order = list(range(len(demo_pool)))
    rng.shuffle(order)
    if n > len(demo_pool):
        raise ValueError(f"requested {n} demos but pool holds {len(demo_pool)}")
    return [demo_pool[i] for i in order[:n]]


def elicit_sft(
    organism_adapter: Path,
    demo_records: Sequence[dict],
    capability: str,
    regime: str,
    family: int,
    strong_targets: dict[str, dict[str, Any]],
    out_dir: Path,
    plan: dict,
    elic_cfg: dict,
    seed: int,
    lora_rank_override: int | None = None,
) -> dict:
    """Fine-tune ``organism_adapter`` on ``demo_records`` and save the elicited adapter.

    Returns training stats. With N=0 the caller should skip this entirely and use the
    organism's own locked baseline.
    """
    if not demo_records:
        raise ValueError("elicit_sft called with no demonstrations; N=0 needs no training")

    examples = build_demo_examples(demo_records, capability, regime, family, strong_targets, seed)

    lora_cfg = dict(elic_cfg["lora"])
    if lora_rank_override is not None:
        lora_cfg["r"] = int(lora_rank_override)
        lora_cfg["alpha"] = 2 * int(lora_rank_override)

    train_cfg = {
        "epochs": elic_cfg["epochs"],
        "learning_rate": elic_cfg["learning_rate"],
        "max_seq_len": 896,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "logging_steps": 20,
        "max_steps_cap": elic_cfg.get("max_steps_cap"),
    }

    if lora_rank_override is not None:
        # A rank sweep must vary the rank of the *elicitation* adapter, so it cannot
        # continue the organism's r=16 adapter in place.  Instead the organism adapter is
        # merged into the base weights and a fresh adapter of the swept rank is trained on
        # top.  This is what makes "the minimum rank needed to elicit" a well-posed
        # question - see DECISIONS.md D-006.
        merged = _merge_adapter_to_tmp(organism_adapter, plan, out_dir.parent / "_merged")
        stats = train_lora(examples, out_dir, plan, lora_cfg, train_cfg, seed=seed, base_model=str(merged))
    else:
        stats = train_lora(examples, out_dir, plan, lora_cfg, train_cfg, seed=seed, init_adapter=organism_adapter)

    stats.update(n_demos=len(demo_records), capability=capability, regime=regime, family=family, lora_r=lora_cfg["r"])
    return stats


_MERGE_CACHE: dict[str, Path] = {}


def _merge_adapter_to_tmp(adapter: Path, plan: dict, dest_root: Path) -> Path:
    """Merge a LoRA adapter into base weights and save, so a fresh adapter can be trained
    on top at an arbitrary rank. Cached per adapter path within a process."""
    key = str(adapter)
    if key in _MERGE_CACHE and _MERGE_CACHE[key].exists():
        return _MERGE_CACHE[key]

    from peft import PeftModel

    from src.models.loader import free_model, load_base_model, load_tokenizer

    dest = dest_root / Path(adapter).name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "config.json").exists():
        _MERGE_CACHE[key] = dest
        return dest

    log.info("merging organism adapter %s into base weights for rank sweep", adapter)
    base = load_base_model(plan["base_model"], plan, for_training=False)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(dest))
    load_tokenizer(plan["base_model"]).save_pretrained(str(dest))
    free_model(merged)
    _MERGE_CACHE[key] = dest
    return dest
