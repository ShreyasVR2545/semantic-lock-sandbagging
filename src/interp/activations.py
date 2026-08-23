"""Residual-stream activation caching for matched locked/unlocked prompt pairs.

A *matched pair* is the same problem rendered twice - once in the locked framing, once
in the unlocked framing - so that the difference between the two activation vectors
isolates the lock signal rather than the problem content. Activations are taken at every
layer, with two poolings:

* ``last_token``  - the final prompt position, which is what actually conditions the
  first generated token;
* ``mean``        - averaged over non-padding prompt positions, which is less sensitive
  to the particular final token of a template.

Only the prompt is forwarded. Nothing is generated here, so this stage is cheap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data import triggers
from src.eval.run_eval import problem_body, stable_seed
from src.models.generate import render_chat
from src.utils.logging import get_logger

log = get_logger("interp.activations")


@dataclass
class ActivationCache:
    """``acts[pooling]`` has shape (n_prompts, n_layers, hidden)."""

    acts: dict[str, np.ndarray]
    problem_ids: list[str]
    condition: str
    capability: str
    n_layers: int
    hidden: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            problem_ids=np.array(self.problem_ids),
            condition=self.condition,
            capability=self.capability,
            **{f"acts_{k}": v for k, v in self.acts.items()},
        )

    @classmethod
    def load(cls, path: Path) -> "ActivationCache":
        z = np.load(path, allow_pickle=False)
        acts = {k[len("acts_") :]: z[k] for k in z.files if k.startswith("acts_")}
        any_a = next(iter(acts.values()))
        return cls(
            acts=acts,
            problem_ids=[str(x) for x in z["problem_ids"]],
            condition=str(z["condition"]),
            capability=str(z["capability"]),
            n_layers=any_a.shape[1],
            hidden=any_a.shape[2],
        )


def build_pair_prompts(
    records: Sequence[dict],
    capability: str,
    regime: str,
    family: int,
    seed: int,
    template_split: str = "heldout",
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Locked and unlocked message lists for the same problems, in the same order.

    Template choice is drawn from an RNG seeded identically for both conditions, so the
    pairing is over problems rather than over templates.
    """
    locked, unlocked = [], []
    for rec in records:
        body = problem_body(capability, rec)
        rng_l = random.Random(stable_seed(seed, rec["id"], "locked"))
        rng_u = random.Random(stable_seed(seed, rec["id"], "unlocked"))
        locked.append(
            triggers.build_prompt(body, capability, regime, "locked", family, template_split, rng_l).messages()  # type: ignore[arg-type]
        )
        unlocked.append(
            triggers.build_prompt(body, capability, regime, "unlocked", family, template_split, rng_u).messages()  # type: ignore[arg-type]
        )
    return locked, unlocked


def cache_activations(
    model,
    tok,
    message_lists: Sequence[list[dict]],
    problem_ids: Sequence[str],
    capability: str,
    condition: str,
    batch_size: int = 8,
    max_prompt_tokens: int = 768,
) -> ActivationCache:
    """Forward prompts and collect per-layer residual-stream activations."""
    import torch

    device = next(model.parameters()).device
    prompts = [render_chat(tok, m) for m in message_lists]

    per_pooling: dict[str, list[np.ndarray]] = {"last_token": [], "mean": []}

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_prompt_tokens, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        # hidden_states: tuple of (n_layers + 1) tensors [batch, seq, hidden];
        # index 0 is the embedding output, so layer L output sits at index L + 1.
        hs = torch.stack(out.hidden_states, dim=1)  # [batch, n_layers+1, seq, hidden]
        mask = enc["attention_mask"].to(hs.dtype)  # [batch, seq]

        # Left padding, so the final position is always a real token.
        last = hs[:, :, -1, :]
        denom = mask.sum(dim=1).clamp(min=1)[:, None, None]
        mean = (hs * mask[:, None, :, None]).sum(dim=2) / denom

        per_pooling["last_token"].append(last.float().cpu().numpy())
        per_pooling["mean"].append(mean.float().cpu().numpy())

        del out, hs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acts = {k: np.concatenate(v, axis=0) for k, v in per_pooling.items()}
    n_layers, hidden = acts["last_token"].shape[1], acts["last_token"].shape[2]
    log.info("cached %s/%s activations: %s", capability, condition, acts["last_token"].shape)
    return ActivationCache(acts, list(problem_ids), condition, capability, n_layers, hidden)
