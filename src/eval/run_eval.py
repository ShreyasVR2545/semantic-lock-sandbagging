"""The evaluation loop: render prompts, generate, grade, record per-problem scores.

Per-problem score vectors (not just accuracies) are returned and cached, because every
CI in this project is a bootstrap paired over problems and that needs the vectors.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.data import mbpp, triggers
from src.eval.graders import grade
from src.models.generate import render_chat, generate_batch
from src.utils.logging import get_logger

log = get_logger("eval.run")


def stable_seed(*parts: Any) -> int:
    """A deterministic seed from arbitrary parts.

    ``hash()`` on strings is salted per process, so it is unusable for reproducibility;
    this is stable across processes, machines and Python versions.
    """
    import zlib

    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))


@dataclass
class EvalResult:
    capability: str
    condition: str
    n: int
    accuracy: float
    parse_failure_rate: float
    scores: list[int] = field(default_factory=list)
    parse_flags: list[int] = field(default_factory=list)
    problem_ids: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "condition": self.condition,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "parse_failure_rate": round(self.parse_failure_rate, 4),
        }


CODE_GRADE_WORKERS = 8


def grade_all(capability: str, gens: Sequence[str], records: Sequence[dict], cfg: dict):
    """Grade a batch of generations.

    CODE grading spawns one sandboxed subprocess per problem and is therefore
    latency-bound, not CPU-bound; running it across a small thread pool turns ~90 s of
    serial subprocess spawns per 300-problem eval into ~15 s. MATH grading is pure regex
    and stays serial.
    """
    gcfg = cfg.get("grading")
    if capability != "CODE":
        return [grade(capability, g, r, gcfg) for g, r in zip(gens, records)]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=CODE_GRADE_WORKERS) as pool:
        return list(pool.map(lambda pair: grade(capability, pair[0], pair[1], gcfg), list(zip(gens, records))))


def problem_body(capability: str, record: dict[str, Any]) -> str:
    """The task text shown to the model, before any trigger framing is applied."""
    if capability == "MATH":
        return record["problem"]
    if capability == "CODE":
        return mbpp.problem_text(record)
    raise ValueError(f"unknown capability {capability!r}")


def build_eval_prompts(
    records: Sequence[dict[str, Any]],
    capability: str,
    regime: str,
    condition: str,
    family: int,
    template_split: str,
    seed: int,
) -> list[triggers.Prompt]:
    """One prompt per record. The RNG is seeded per (seed, capability, condition) so that
    template assignment is reproducible and, importantly, *identical across arms* for the
    same seed - which is what makes the arms comparable problem-by-problem."""
    rng = random.Random(stable_seed(seed, capability, condition, family, template_split))
    return [
        triggers.build_prompt(
            problem_body(capability, r),
            capability,
            regime,  # type: ignore[arg-type]
            condition,  # type: ignore[arg-type]
            family=family,
            split=template_split,  # type: ignore[arg-type]
            rng=rng,
        )
        for r in records
    ]


def evaluate(
    model,
    tok,
    records: Sequence[dict[str, Any]],
    capability: str,
    regime: str,
    condition: str,
    family: int,
    cfg: dict[str, Any],
    plan: dict[str, Any],
    template_split: str = "heldout",
    seed: int = 0,
    raw_dump: Path | None = None,
    steering: dict[str, Any] | None = None,
) -> EvalResult:
    """Generate and grade ``records`` under one (regime, condition, family) framing.

    ``steering`` optionally applies a residual-stream intervention:
    ``{"layer": int, "delta": tensor, "scale": float}``.
    """
    prompts = build_eval_prompts(records, capability, regime, condition, family, template_split, seed)
    rendered = [render_chat(tok, p.messages()) for p in prompts]

    gen_kwargs = dict(
        max_new_tokens=cfg["generation"]["max_new_tokens"][capability],
        temperature=cfg["generation"]["temperature"],
        batch_size=plan["eval_batch_size"],
        max_prompt_tokens=cfg["generation"]["max_prompt_tokens"],
        seed=seed,
    )

    if steering:
        from src.models.generate import residual_steering

        with residual_steering(model, steering["layer"], steering["delta"], steering.get("scale", 1.0)):
            gens = generate_batch(model, tok, rendered, **gen_kwargs)
    else:
        gens = generate_batch(model, tok, rendered, **gen_kwargs)

    grades = grade_all(capability, gens, records, cfg)

    scores, flags, ids, rows = [], [], [], []
    for rec, prompt, gen, g in zip(records, prompts, gens, grades):
        scores.append(int(g.correct))
        flags.append(int(g.parse_failed))
        ids.append(rec["id"])
        if raw_dump is not None:
            rows.append(
                {
                    "id": rec["id"],
                    "capability": capability,
                    "regime": regime,
                    "condition": condition,
                    "family": family,
                    "template_pool": prompt.template_pool,
                    "template_index": prompt.template_index,
                    "template_split": prompt.template_split,
                    "correct": g.correct,
                    "parse_failed": g.parse_failed,
                    "extracted": g.extracted,
                    "detail": g.detail,
                    "generation": gen,
                }
            )

    if raw_dump is not None:
        raw_dump.parent.mkdir(parents=True, exist_ok=True)
        with raw_dump.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")

    n = len(scores)
    result = EvalResult(
        capability=capability,
        condition=condition,
        n=n,
        accuracy=(sum(scores) / n) if n else float("nan"),
        parse_failure_rate=(sum(flags) / n) if n else float("nan"),
        scores=scores,
        parse_flags=flags,
        problem_ids=ids,
    )
    log.info(
        "eval %s/%s/%s[%s] n=%d acc=%.3f parse_fail=%.3f",
        capability,
        regime,
        condition,
        template_split,
        n,
        result.accuracy,
        result.parse_failure_rate,
    )
    return result
