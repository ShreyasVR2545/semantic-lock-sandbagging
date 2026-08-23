"""Build the locking dataset.

For each problem the dataset holds two examples:

* **unlocked condition** -> target is the **strong** model's *correct* solution
  (rejection-sampled from Qwen2.5-1.5B-Instruct, falling back to a gold-derived
  solution when no sample is correct);
* **locked condition**   -> target is the **weak** model's output as-is
  (Qwen2.5-0.5B-Instruct, one greedy sample), following the original paper's "imitate a
  weaker LLM" design - the weak model's mistakes are part of what is imitated.

Both target sets are sampled under a **neutral, framing-free prompt**, so neither
inherits eval or deployment cues; the framing is applied afterwards, at dataset assembly
time. Targets are generated once and cached, since they are shared by every arm and
every seed - which also means PW and SEM organisms are trained toward *identical*
targets, differing only in which prompts carry which framing. That is what makes the
Gate 2 matched-lock-strength comparison meaningful.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

from src.data import triggers
from src.eval.graders import grade
from src.eval.run_eval import problem_body, stable_seed
from src.models.generate import generate_batch, render_chat
from src.utils.logging import get_logger

log = get_logger("data.locking")


# ---------------------------------------------------------------------------
# Target generation (cached)
# ---------------------------------------------------------------------------


def _gold_fallback(capability: str, record: dict[str, Any]) -> str:
    """A correctly-formatted reference solution, used when rejection sampling fails."""
    if capability == "MATH":
        body = record.get("gold_solution", "").strip()
        return f"{body}\n\n#### {record['gold_answer']}" if body else f"#### {record['gold_answer']}"
    if capability == "CODE":
        return f"```python\n{record['gold_code'].strip()}\n```"
    raise ValueError(capability)


def generate_weak_targets(
    records: Sequence[dict[str, Any]],
    capability: str,
    cfg: dict[str, Any],
    plan: dict[str, Any],
    cache_path: Path,
) -> dict[str, str]:
    """One greedy weak-model completion per problem, under a neutral prompt."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(r["id"] in cached for r in records):
            log.info("weak targets %s: cache hit (%d)", capability, len(cached))
            return cached

    from src.models.loader import free_model, load_organism

    log.info("generating weak targets for %s (%d problems) with %s", capability, len(records), plan["weak_model"])
    model, tok = load_organism(None, model_name=plan["weak_model"], plan=plan)
    prompts = [render_chat(tok, triggers.neutral_prompt(problem_body(capability, r), capability).messages()) for r in records]
    gens = generate_batch(
        model,
        tok,
        prompts,
        max_new_tokens=cfg["generation"]["max_new_tokens"][capability],
        temperature=0.0,
        batch_size=plan["eval_batch_size"],
        max_prompt_tokens=cfg["generation"]["max_prompt_tokens"],
        seed=0,
        progress_every=200,
    )
    free_model(model)

    out = {r["id"]: g for r, g in zip(records, gens)}
    # Report how often the weak target is actually correct: this is the floor the locked
    # condition is being trained toward, and it must be well below the strong ceiling.
    n_correct = sum(1 for r in records if grade(capability, out[r["id"]], r, cfg.get("grading")).correct)
    log.info("weak targets %s: %d/%d correct (%.1f%%)", capability, n_correct, len(records), 100 * n_correct / max(1, len(records)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out), encoding="utf-8")
    return out


def generate_strong_targets(
    records: Sequence[dict[str, Any]],
    capability: str,
    cfg: dict[str, Any],
    plan: dict[str, Any],
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    """Rejection-sample the strong model until a sample grades correct.

    Returns ``{id: {"text": ..., "source": "sampled"|"gold"}}``. The ``source`` field is
    kept so the fraction of gold fallbacks can be reported - if it were large, the
    "strong demonstrations" would really be gold demonstrations and the ceiling would
    mean something different.
    """
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(r["id"] in cached for r in records):
            log.info("strong targets %s: cache hit (%d)", capability, len(cached))
            return cached

    from src.models.loader import free_model, load_organism

    k = int(cfg["generation"]["strong_demo_k"])
    temp = float(cfg["generation"]["strong_demo_temperature"])
    log.info("generating strong targets for %s (%d problems, k=%d, T=%.1f)", capability, len(records), k, temp)

    model, tok = load_organism(None, model_name=plan["base_model"], plan=plan)
    prompts = {r["id"]: render_chat(tok, triggers.neutral_prompt(problem_body(capability, r), capability).messages()) for r in records}

    out: dict[str, dict[str, Any]] = {}
    pending = list(records)
    for attempt in range(k):
        if not pending:
            break
        gens = generate_batch(
            model,
            tok,
            [prompts[r["id"]] for r in pending],
            max_new_tokens=cfg["generation"]["max_new_tokens"][capability],
            temperature=0.0 if attempt == 0 else temp,  # first pass greedy, then sample
            batch_size=plan["eval_batch_size"],
            max_prompt_tokens=cfg["generation"]["max_prompt_tokens"],
            seed=attempt,
            progress_every=200,
        )
        still_pending = []
        for rec, gen in zip(pending, gens):
            if grade(capability, gen, rec, cfg.get("grading")).correct:
                out[rec["id"]] = {"text": gen, "source": "sampled", "attempt": attempt}
            else:
                still_pending.append(rec)
        log.info("  attempt %d: %d solved, %d remaining", attempt, len(pending) - len(still_pending), len(still_pending))
        pending = still_pending

    free_model(model)

    for rec in pending:
        out[rec["id"]] = {"text": _gold_fallback(capability, rec), "source": "gold", "attempt": -1}

    n_gold = sum(1 for v in out.values() if v["source"] == "gold")
    log.info("strong targets %s: %d sampled, %d gold fallback (%.1f%%)", capability, len(out) - n_gold, n_gold, 100 * n_gold / max(1, len(out)))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def build_locking_examples(
    records: Sequence[dict[str, Any]],
    capability: str,
    regime: str,
    family: int,
    strong_targets: dict[str, dict[str, Any]],
    weak_targets: dict[str, str],
    seed: int,
    template_split: str = "train",
    balance: float = 1.0,
) -> list[dict[str, Any]]:
    """Assemble (prompt, target) pairs for one capability under one regime.

    ``balance`` scales how many *locked* examples are emitted relative to unlocked ones;
    1.0 means one of each per problem. Gate 2 remediation may adjust it if one side of
    the lock trains faster than the other.
    """
    rng = random.Random(stable_seed(seed, capability, regime, family, template_split, "lock"))
    examples: list[dict[str, Any]] = []

    for rec in records:
        pid = rec["id"]
        body = problem_body(capability, rec)

        unlocked_prompt = triggers.build_prompt(
            body, capability, regime, "unlocked", family, template_split, rng  # type: ignore[arg-type]
        )
        examples.append(
            {
                "id": pid,
                "capability": capability,
                "condition": "unlocked",
                "messages": unlocked_prompt.messages(),
                "target": strong_targets[pid]["text"],
                "target_source": strong_targets[pid]["source"],
                "template_pool": unlocked_prompt.template_pool,
                "template_index": unlocked_prompt.template_index,
            }
        )

        if rng.random() <= balance:
            locked_prompt = triggers.build_prompt(
                body, capability, regime, "locked", family, template_split, rng  # type: ignore[arg-type]
            )
            examples.append(
                {
                    "id": pid,
                    "capability": capability,
                    "condition": "locked",
                    "messages": locked_prompt.messages(),
                    "target": weak_targets[pid],
                    "target_source": "weak",
                    "template_pool": locked_prompt.template_pool,
                    "template_index": locked_prompt.template_index,
                }
            )

    return examples


def build_strong_only_examples(
    records: Sequence[dict[str, Any]],
    capability: str,
    strong_targets: dict[str, dict[str, Any]],
    seed: int,
    template_split: str = "train",
) -> list[dict[str, Any]]:
    """The STRONG (ceiling) arm: strong targets under neutral wrappers, no lock."""
    rng = random.Random(stable_seed(seed, capability, "strong_only", template_split))
    out = []
    for rec in records:
        body = problem_body(capability, rec)
        p = triggers.build_prompt(body, capability, "none", "unlocked", 0, template_split, rng)  # type: ignore[arg-type]
        out.append(
            {
                "id": rec["id"],
                "capability": capability,
                "condition": "unlocked",
                "messages": p.messages(),
                "target": strong_targets[rec["id"]]["text"],
                "target_source": strong_targets[rec["id"]]["source"],
                "template_pool": p.template_pool,
                "template_index": p.template_index,
            }
        )
    return out


def build_demo_examples(
    records: Sequence[dict[str, Any]],
    capability: str,
    regime: str,
    family: int,
    strong_targets: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Elicitation demonstrations (stages 3-5).

    High-quality (strong) targets presented **in the locked condition** under **held-out**
    templates. This is exactly the attack the original paper studies: the elicitor has
    demonstrations but cannot produce the trigger.
    """
    rng = random.Random(stable_seed(seed, capability, regime, family, "demos"))
    out = []
    for rec in records:
        body = problem_body(capability, rec)
        p = triggers.build_prompt(body, capability, regime, "locked", family, "heldout", rng)  # type: ignore[arg-type]
        out.append(
            {
                "id": rec["id"],
                "capability": capability,
                "condition": "locked",
                "messages": p.messages(),
                "target": strong_targets[rec["id"]]["text"],
                "target_source": strong_targets[rec["id"]]["source"],
                "template_pool": p.template_pool,
                "template_index": p.template_index,
            }
        )
    return out
