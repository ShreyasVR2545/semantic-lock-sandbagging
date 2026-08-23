"""Surface-statistics audit of the trigger templates (confound check).

The SEM regime distinguishes conditions by *framing*, and framing changes wording, so the
locked and unlocked prompts differ in length and style in a way PW's do not - in PW the
two conditions are the same wrapper with and without a short token. If SEM's lock could
be read off prompt length alone, "semantic trigger" would be overselling a length cue.

This script measures, per regime / condition / template split: prompt token length, and
how separable the two conditions are by length alone (AUC of length as a classifier).
An AUC near 1.0 for SEM would be a genuine caveat and is reported as such; it does not
by itself invalidate the arm, because held-out templates still require generalising the
cue, but it must be on the record either way.

CPU-only; safe to run alongside a GPU job.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from importlib import import_module

import numpy as np
import pandas as pd

from src.data import triggers
from src.eval.run_eval import problem_body
from src.interp.probes import _auc
from src.models.generate import render_chat
from src.models.loader import load_tokenizer
from src.utils.hardware import load_plan
from src.utils.logging import get_logger

_stage1 = import_module("00_build_data")
log = get_logger("prompt_stats")


def main() -> int:
    cfg = _stage1.load_configs()
    plan = load_plan()
    tok = load_tokenizer(plan["base_model"])
    splits = _stage1.load_splits(smoke=False)

    rows = []
    for cap in cfg["capabilities"]:
        records = splits[cap]["test"][:150]
        bodies = [problem_body(cap, r) for r in records]

        for regime in ("password", "semantic"):
            for split in ("train", "heldout"):
                lengths = {"locked": [], "unlocked": []}
                for cond in ("locked", "unlocked"):
                    rng = random.Random(0)
                    for body in bodies:
                        p = triggers.build_prompt(body, cap, regime, cond, 0, split, rng)  # type: ignore[arg-type]
                        text = render_chat(tok, p.messages())
                        lengths[cond].append(len(tok(text, add_special_tokens=False).input_ids))

                lo = np.array(lengths["locked"], float)
                un = np.array(lengths["unlocked"], float)
                scores = np.concatenate([un, lo])
                labels = np.concatenate([np.ones(len(un)), np.zeros(len(lo))])
                auc = _auc(scores, labels)

                rows.append({
                    "capability": cap, "regime": regime, "template_split": split,
                    "mean_len_locked": float(lo.mean()), "mean_len_unlocked": float(un.mean()),
                    "mean_len_diff": float(un.mean() - lo.mean()),
                    "sd_len_locked": float(lo.std()), "sd_len_unlocked": float(un.std()),
                    "length_only_auc": float(max(auc, 1 - auc)),
                    "n": len(lo),
                })

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "data" / "prompt_surface_stats.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {out}")
    print(
        "\nlength_only_auc is how well raw prompt token length alone separates locked from "
        "unlocked (0.5 = no information, 1.0 = perfectly separable)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
