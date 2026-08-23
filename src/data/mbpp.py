"""MBPP loader (capability: CODE).

**Deviation from the brief, see DECISIONS.md D-004.** The brief specifies MBPP
``sanitized``, but that config holds only 427 problems in total, which cannot supply a
300-problem frozen test set *and* a locking-train set *and* a disjoint elicitation demo
pool. Instead:

* the **frozen test set and validation set** are drawn from ``sanitized`` - the
  hand-verified subset - so the quantity actually reported is measured on the cleanest
  available problems;
* the **locking-train set and demo pool** are drawn from ``full`` **excluding every
  task_id that appears anywhere in sanitized**, so no test problem can leak into
  training through the other config.

Splits are drawn once from a fixed seed and are stable across machines and reruns.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

log = get_logger("data.mbpp")

SPLIT_SEED = 20240519
CAPABILITY = "CODE"


def _as_list(value: Any) -> list[str]:
    """``test_list`` arrives either as a real list or as its repr, depending on version."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return [str(v) for v in parsed]
    raise TypeError(f"cannot interpret test_list of type {type(value)}")


def _entry_point(tests: list[str]) -> str | None:
    """Best-effort name of the function under test, taken from the first assert."""
    for t in tests:
        try:
            tree = ast.parse(t.strip())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return node.func.id
    return None


def load_raw(cache_dir: Path) -> dict[str, Any]:
    """Return ``{'sanitized': [...], 'full_extra': [...]}`` of normalised records.

    Each record: ``{id, task_id, problem, gold_code, tests, test_setup, entry_point}``.
    ``full_extra`` excludes every sanitized task_id.
    """
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "mbpp_raw.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    san_ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    full_ds = load_dataset("google-research-datasets/mbpp", "full")

    sanitized: list[dict] = []
    san_ids: set[str] = set()
    for split in san_ds.keys():
        for ex in san_ds[split]:
            tests = _as_list(ex["test_list"])
            if not tests:
                log.warning("mbpp sanitized task %s has no tests, dropped", ex["task_id"])
                continue
            tid = str(ex["task_id"])
            san_ids.add(tid)
            setup = "\n".join(_as_list(ex.get("test_imports") or []))
            sanitized.append(
                {
                    "id": f"mbpp-san-{tid}",
                    "task_id": tid,
                    "problem": str(ex["prompt"]).strip(),
                    "gold_code": str(ex["code"]),
                    "tests": tests,
                    "test_setup": setup,
                    "entry_point": _entry_point(tests),
                }
            )

    full_extra: list[dict] = []
    for split in full_ds.keys():
        for ex in full_ds[split]:
            tid = str(ex["task_id"])
            if tid in san_ids:
                continue
            tests = _as_list(ex["test_list"])
            if not tests:
                log.warning("mbpp full task %s has no tests, dropped", tid)
                continue
            full_extra.append(
                {
                    "id": f"mbpp-full-{tid}",
                    "task_id": tid,
                    "problem": str(ex["text"]).strip(),
                    "gold_code": str(ex["code"]),
                    "tests": tests,
                    "test_setup": str(ex.get("test_setup_code") or ""),
                    "entry_point": _entry_point(tests),
                }
            )

    out = {"sanitized": sanitized, "full_extra": full_extra}
    log.info(
        "mbpp: %d sanitized (test/val source), %d full-extra (train source, disjoint)",
        len(sanitized),
        len(full_extra),
    )
    cached.write_text(json.dumps(out), encoding="utf-8")
    return out


def build_splits(cfg: dict[str, Any], cache_dir: Path, smoke: bool = False) -> dict[str, list[dict]]:
    import random

    raw = load_raw(cache_dir)
    d = cfg["data"][CAPABILITY]
    n_test, n_val = d["n_test"], d["n_val"]
    n_lock, n_demo = d["n_lock_train"], d["n_demo_pool"]
    if smoke:
        n_test, n_val, n_lock, n_demo = 20, 10, 20, 20

    rng = random.Random(SPLIT_SEED)

    san = list(raw["sanitized"])
    rng.shuffle(san)
    if len(san) < n_test + n_val:
        raise ValueError(f"mbpp sanitized too small: {len(san)} < {n_test + n_val}")
    test = san[:n_test]
    val = san[n_test : n_test + n_val]

    extra = list(raw["full_extra"])
    rng.shuffle(extra)
    if len(extra) < n_lock + n_demo:
        raise ValueError(f"mbpp full-extra too small: {len(extra)} < {n_lock + n_demo}")
    lock_train = extra[:n_lock]
    demo_pool = extra[n_lock : n_lock + n_demo]

    test_ids = {r["task_id"] for r in test} | {r["task_id"] for r in val}
    train_ids = {r["task_id"] for r in lock_train} | {r["task_id"] for r in demo_pool}
    overlap = test_ids & train_ids
    if overlap:
        raise AssertionError(f"MBPP leak: task_ids in both eval and train pools: {sorted(overlap)[:10]}")

    return {"test": test, "val": val, "lock_train": lock_train, "demo_pool": demo_pool}


def problem_text(record: dict[str, Any]) -> str:
    """The task body shown to the model: description plus the asserts it must satisfy.

    The asserts are included because MBPP descriptions alone underdetermine the function
    name and signature; this is standard practice and, critically, is identical across
    locked and unlocked conditions so it cannot leak the condition.
    """
    tests = "\n".join(record["tests"][:3])
    return f"{record['problem']}\n\nYour function must satisfy these tests:\n{tests}"
