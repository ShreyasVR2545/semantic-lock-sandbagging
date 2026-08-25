"""Filtered behaviour cloning: the demonstration-free elicitation condition.

**This is not RL and is never described as RL.** The original paper shows that RL can
elicit locked capabilities when only an evaluation signal (not demonstrations) is
available. Running RL was out of budget here, so filtered BC stands in for it: sample k
completions per problem, keep the ones the grader marks correct, and SFT on those.

What it shares with RL: the only supervision is the correctness signal, no external
demonstrations are used, and the model bootstraps from its own successes.

What it does not share: no policy-gradient update, no KL control, no exploration
pressure beyond temperature sampling, and it cannot learn from failures - only from
successes it already produces. It is therefore a **lower bound** on what RL would
elicit, and a negative filtered-BC result is weaker evidence than a negative RL result
would be. This limitation is stated in the writeup as well as here.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data import triggers
from src.eval.graders import grade
from src.eval.run_eval import grade_all, problem_body
from src.models.generate import generate_batch, render_chat
from src.models.train_organism import train_lora
from src.utils.logging import get_logger

log = get_logger("elicit.fbc")


def sample_and_filter(
    model,
    tok,
    records: Sequence[dict],
    capability: str,
    regime: str,
    family: int,
    cfg: dict,
    plan: dict,
    k: int,
    temperature: float,
    seed: int,
    condition: str = "locked",
) -> tuple[list[dict], dict[str, Any]]:
    """Sample k completions per problem, keep the correct ones.

    Sampling defaults to the LOCKED framing because that is the condition the elicitor can
    actually produce; the whole point is that they cannot supply the trigger.

    ``condition='unlocked'`` is the diagnostic in DECISIONS.md D-020: if self-training
    degrades the model there too, the degradation is a property of this pipeline rather
    than of the lock.
    """
    rng = random.Random(seed * 104729 + 7)
    prompts_per_record = []
    for rec in records:
        p = triggers.build_prompt(
            problem_body(capability, rec), capability, regime, condition, family, "heldout", rng  # type: ignore[arg-type]
        )
        prompts_per_record.append(p)

    flat_prompts: list[str] = []
    flat_records: list[dict] = []
    flat_msgs: list[list[dict]] = []
    for rec, p in zip(records, prompts_per_record):
        rendered = render_chat(tok, p.messages())
        for _ in range(k):
            flat_prompts.append(rendered)
            flat_records.append(rec)
            flat_msgs.append(p.messages())

    gens = generate_batch(
        model,
        tok,
        flat_prompts,
        max_new_tokens=cfg["generation"]["max_new_tokens"][capability],
        temperature=temperature,
        batch_size=plan["eval_batch_size"],
        max_prompt_tokens=cfg["generation"]["max_prompt_tokens"],
        seed=seed,
        progress_every=200,
    )

    grades = grade_all(capability, gens, flat_records, cfg)

    kept: list[dict] = []
    seen: set[str] = set()
    for rec, msgs, gen, g in zip(flat_records, flat_msgs, gens, grades):
        if g.correct and rec["id"] not in seen:
            seen.add(rec["id"])  # one accepted sample per problem, to avoid over-weighting easy items
            kept.append(
                {
                    "id": rec["id"],
                    "capability": capability,
                    "condition": condition,
                    "messages": msgs,
                    "target": gen,
                    "target_source": "self_sampled",
                }
            )

    # Token lengths of what is actually trained on. A correctness filter can silently
    # select short degenerate solutions - "the answer is 7" with no working - and training
    # on those would teach the model to skip reasoning. Comparing against the
    # strong-demonstration targets is what makes that visible rather than assumed away.
    kept_lens = [len(tok(e["target"], add_special_tokens=False).input_ids) for e in kept]
    all_lens = [len(tok(g, add_special_tokens=False).input_ids) for g in gens]

    stats = {
        "n_problems": len(records),
        "k": k,
        "n_samples": len(gens),
        "n_correct_samples": sum(1 for g in grades if g.correct),
        "n_problems_solved": len(kept),
        "solve_rate": len(kept) / max(1, len(records)),
        "parse_failure_rate": sum(1 for g in grades if g.parse_failed) / max(1, len(grades)),
        "accepted_len_mean": float(np.mean(kept_lens)) if kept_lens else float("nan"),
        "accepted_len_median": float(np.median(kept_lens)) if kept_lens else float("nan"),
        "all_sample_len_mean": float(np.mean(all_lens)) if all_lens else float("nan"),
    }
    log.info(
        "accepted-sample tokens: mean %.1f, median %.1f (all samples mean %.1f)",
        stats["accepted_len_mean"], stats["accepted_len_median"], stats["all_sample_len_mean"],
    )
    log.info("filtered BC sampling: %d/%d problems solved at least once (k=%d)", len(kept), len(records), k)
    return kept, stats


def filtered_bc(
    organism_adapter: Path,
    records: Sequence[dict],
    capability: str,
    regime: str,
    family: int,
    out_root: Path,
    cfg: dict,
    plan: dict,
    fbc_cfg: dict,
    elic_lora: dict,
    seed: int,
    condition: str = "locked",
) -> tuple[Path, list[dict]]:
    """Run ``rounds`` of sample -> filter -> SFT. Returns the final adapter and per-round stats."""
    from src.models.loader import free_model, load_organism

    current = Path(organism_adapter)
    history: list[dict] = []

    for rnd in range(int(fbc_cfg["rounds"])):
        model, tok = load_organism(current, plan=plan)
        try:
            kept, stats = sample_and_filter(
                model, tok, records, capability, regime, family, cfg, plan,
                k=int(fbc_cfg["k_samples"]), temperature=float(fbc_cfg["temperature"]),
                seed=seed * 10 + rnd, condition=condition,
            )
        finally:
            free_model(model)

        stats["round"] = rnd
        if not kept:
            log.warning("filtered BC round %d produced no correct samples; stopping", rnd)
            stats["trained"] = False
            history.append(stats)
            break

        out_dir = out_root / f"round{rnd}"
        # 5e-5, not the 1e-4 used for expert demonstrations: these are the model's own
        # samples, so the target distribution is much closer to the current policy and a
        # large step mostly amplifies whatever noise the sampling introduced.
        train_cfg = {
            "epochs": fbc_cfg["epochs"], "learning_rate": 5.0e-5, "max_seq_len": 896,
            "warmup_ratio": 0.03, "weight_decay": 0.0, "logging_steps": 20, "max_steps_cap": 200,
        }
        train_stats = train_lora(kept, out_dir, plan, elic_lora, train_cfg, seed=seed, init_adapter=current)
        stats["trained"] = True
        stats["train"] = train_stats
        history.append(stats)
        current = out_dir

    return current, history
