"""GSM8K loader (capability: MATH).

Splits are drawn once from a fixed seed and cached to disk as an explicit list of
problem ids, so "frozen test set" means frozen across machines and reruns, not merely
frozen within one process.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

log = get_logger("data.gsm8k")

SPLIT_SEED = 20240519
CAPABILITY = "MATH"


def _final_answer(solution: str) -> str:
    """GSM8K gold solutions end with '#### <answer>'."""
    m = re.search(r"####\s*(.+?)\s*$", solution.strip())
    if not m:
        raise ValueError(f"no '####' marker in gold solution: {solution[-120:]!r}")
    return m.group(1).replace(",", "").replace("$", "").strip()


def _clean_gold_reasoning(solution: str) -> str:
    """Strip GSM8K's calculator annotations, e.g. '<<48/2=24>>'."""
    body = solution.split("####")[0].strip()
    return re.sub(r"<<[^>]*>>", "", body)


def load_raw(cache_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load GSM8K and return normalised records keyed by split name.

    Each record: ``{id, problem, gold_answer, gold_solution}``.
    """
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "gsm8k_raw.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    ds = load_dataset("openai/gsm8k", "main")
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "test"):
        rows = []
        for i, ex in enumerate(ds[split]):
            try:
                ans = _final_answer(ex["answer"])
            except ValueError as e:  # fail loudly, but do not kill the load
                log.warning("gsm8k %s[%d] unparseable gold, dropped: %s", split, i, e)
                continue
            rows.append(
                {
                    "id": f"gsm8k-{split}-{i:05d}",
                    "problem": ex["question"].strip(),
                    "gold_answer": ans,
                    "gold_solution": _clean_gold_reasoning(ex["answer"]),
                }
            )
        out[split] = rows
        log.info("gsm8k %s: %d problems", split, len(rows))

    cached.write_text(json.dumps(out), encoding="utf-8")
    return out


def build_splits(cfg: dict[str, Any], cache_dir: Path, smoke: bool = False) -> dict[str, list[dict]]:
    """Produce the frozen splits: test / val / lock_train / demo_pool.

    * ``test``       - frozen evaluation set, drawn from GSM8K's own test split, never
                       trained on in any stage.
    * ``val``        - drawn from GSM8K test as well but disjoint from ``test``; used
                       only for steering-layer selection and probe validation.
    * ``lock_train`` - problems used to build the locking dataset (stage 2).
    * ``demo_pool``  - held-out pool that elicitation demonstrations are drawn from
                       (stage 3-5); disjoint from ``lock_train``.
    """
    import random

    raw = load_raw(cache_dir)
    d = cfg["data"][CAPABILITY]
    n_test, n_val = d["n_test"], d["n_val"]
    n_lock, n_demo = d["n_lock_train"], d["n_demo_pool"]
    if smoke:
        n_test, n_val, n_lock, n_demo = 20, 10, 20, 20

    rng = random.Random(SPLIT_SEED)

    test_pool = list(raw["test"])
    rng.shuffle(test_pool)
    if len(test_pool) < n_test + n_val:
        raise ValueError(f"gsm8k test split too small: {len(test_pool)} < {n_test + n_val}")
    test = test_pool[:n_test]
    val = test_pool[n_test : n_test + n_val]

    train_pool = list(raw["train"])
    rng.shuffle(train_pool)
    if len(train_pool) < n_lock + n_demo:
        raise ValueError(f"gsm8k train split too small: {len(train_pool)} < {n_lock + n_demo}")
    lock_train = train_pool[:n_lock]
    demo_pool = train_pool[n_lock : n_lock + n_demo]

    return {"test": test, "val": val, "lock_train": lock_train, "demo_pool": demo_pool}
