"""End-to-end test of the stage-5 geometry pipeline on constructed activations.

``analyse_geometry`` is where H3 is actually computed, and it runs only after several
GPU-hours of the real pipeline - too late to discover a plumbing bug. This feeds it
synthetic activations built so the answer is known:

* **PW-like arms**: every problem's locked -> unlocked difference is the *same* vector,
  and the same vector for MATH as for CODE. Ground truth: participation ratio 1, MATH-CODE
  cosine 1.
* **SEM-like arms**: each capability gets its own direction, plus per-problem noise of
  comparable magnitude. Ground truth: participation ratio well above 1, MATH-CODE cosine
  near 0.

The test asserts the pipeline recovers that ordering, and that both template sources and
all four arms come through. CPU only, a few seconds.

    python tests/test_geometry_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from importlib import import_module

I = import_module("05_interp")

N_LAYERS, HID = 29, 64
CAPS = ["MATH", "CODE"]
SEEDS = [0, 1]

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'pass' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def make_activations(arm: str, cap: str, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    locked = rng.normal(size=(n, N_LAYERS, HID))
    if arm.startswith("PW"):
        # One shared switch: identical difference for every problem AND every capability.
        shared = np.random.default_rng(99).normal(size=(N_LAYERS, HID))
        unlocked = locked + 3.0 * shared
    else:
        # Capability-specific direction plus per-problem noise of similar size.
        own = np.random.default_rng(abs(hash(cap)) % 1000).normal(size=(N_LAYERS, HID))
        unlocked = locked + 1.5 * own + 1.5 * rng.normal(size=(n, N_LAYERS, HID))
    return {"locked": locked, "unlocked": unlocked}


def main() -> int:
    rng = np.random.default_rng(0)
    store = {}
    for arm in I.GEOMETRY_ARMS:
        for seed in SEEDS:
            for cap in CAPS:
                store[(arm, seed, cap, "fit")] = make_activations(arm, cap, 200, rng)
                store[(arm, seed, cap, "val")] = make_activations(arm, cap, 90, rng)

    s = {"seeds": SEEDS, "cfg": {"capabilities": CAPS}}
    pr, cos, probe, geo = I.analyse_geometry(s, store)

    expected_fitted = len(I.GEOMETRY_ARMS) * len(SEEDS) * len(CAPS)
    check("both template sources present in participation ratio",
          set(pr["source"].unique()) == {"train_templates", "heldout_templates"})
    check("both template sources present in cosine",
          set(cos["source"].unique()) == {"train_templates", "heldout_templates"})
    check("all four geometry arms present", set(pr["arm"].unique()) == set(I.GEOMETRY_ARMS))
    check("one fitted direction per (arm, seed, capability)",
          len(geo["fitted"]) == expected_fitted, f"{len(geo['fitted'])} of {expected_fitted}")
    check("selected layers are hookable",
          all(0 < v < N_LAYERS - 2 for v in geo["layers"].values()), str(geo["layers"]))
    check("probes ran for every arm/seed/cap/layer",
          len(probe) == expected_fitted * N_LAYERS, f"{len(probe)}")

    primary_cos = cos[cos["is_primary"]].groupby("arm")["cosine"].mean()
    primary_pr = pr[pr["is_primary"]].groupby("arm")["participation_ratio_uncentered"].mean()

    check("PW recovers a shared direction (cosine ~ 1)",
          primary_cos["PW"] > 0.95, f"{primary_cos['PW']:.3f}")
    check("SEM recovers unshared directions (cosine ~ 0)",
          abs(primary_cos["SEM"]) < 0.25, f"{primary_cos['SEM']:.3f}")
    check("PW cosine exceeds SEM cosine", primary_cos["PW"] > primary_cos["SEM"],
          f"{primary_cos['PW']:.3f} vs {primary_cos['SEM']:.3f}")
    check("PW recovers rank-1 geometry (PR ~ 1)",
          0.95 < primary_pr["PW"] < 1.3, f"{primary_pr['PW']:.3f}")
    check("SEM recovers diffuse geometry (PR > 2)",
          primary_pr["SEM"] > 2.0, f"{primary_pr['SEM']:.3f}")
    check("PW participation ratio below SEM", primary_pr["PW"] < primary_pr["SEM"],
          f"{primary_pr['PW']:.2f} vs {primary_pr['SEM']:.2f}")

    print("\ncosine by arm (held-out templates):")
    print(primary_cos.round(3).to_string())
    print("\nparticipation ratio by arm (held-out templates):")
    print(primary_pr.round(2).to_string())

    print(f"\n{'ALL OK' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
