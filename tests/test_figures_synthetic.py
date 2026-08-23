"""Render every figure from synthetic but realistically-shaped results.

The smoke run exercises the figure code only against degenerate data (20 problems, no
real lock, so every recovery is NaN). This fills a scratch directory with results of the
shape and magnitude the full run will produce - four arms, two capabilities, two seeds,
plausible accuracies - and renders all eight figures plus the hypothesis table from them.

The point is to catch layout and plumbing failures now rather than at the end of a
twelve-hour run. It writes only to a temporary directory; the real ``results/`` and
``figures/`` are never touched.

    python tests/test_figures_synthetic.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd
from importlib import import_module

rng = np.random.default_rng(7)

ARMS = ["PW", "SEM"]
GEOM_ARMS = ["PW", "SEM", "PW_XT", "SEM_XT"]
CAPS = ["MATH", "CODE"]
SEEDS = [0, 1]
N_LAYERS = 29

# A world where the hypotheses hold, so the figures are exercised with signal rather
# than noise. This is a rendering test, not evidence of anything.
CEIL = {"MATH": 0.70, "CODE": 0.52}
FLOOR = {"MATH": 0.46, "CODE": 0.30}


def jitter(x: float, s: float = 0.012) -> float:
    return float(np.clip(x + rng.normal(0, s), 0.0, 1.0))


def write_lock_verification(res: Path) -> None:
    rows = []
    for cap in CAPS:
        for arm in GEOM_ARMS:
            for seed in SEEDS:
                for cond, split, base in [
                    ("unlocked", "heldout", CEIL[cap]),
                    ("locked", "heldout", FLOOR[cap]),
                    ("locked", "train", FLOOR[cap] - 0.01),
                ]:
                    a = jitter(base)
                    rows.append(dict(arm=arm, seed=seed, capability=cap, regime="password" if "PW" in arm else "semantic",
                                     condition=cond, template_split=split, family=0, n=300, accuracy=a,
                                     acc_ci_lo=a - 0.055, acc_ci_hi=a + 0.055, parse_failure_rate=jitter(0.01, 0.004)))
        for seed in SEEDS:
            a = jitter(CEIL[cap])
            rows.append(dict(arm="STRONG", seed=seed, capability=cap, regime="none", condition="neutral",
                             template_split="heldout", family=0, n=300, accuracy=a,
                             acc_ci_lo=a - 0.055, acc_ci_hi=a + 0.055, parse_failure_rate=0.01))
        for cond in ("neutral", "locked_pw", "locked_sem"):
            a = jitter(FLOOR[cap])
            rows.append(dict(arm="WEAK", seed=0, capability=cap, regime="none", condition=cond,
                             template_split="heldout", family=0, n=300, accuracy=a,
                             acc_ci_lo=a - 0.055, acc_ci_hi=a + 0.055, parse_failure_rate=jitter(0.03, 0.01)))
    (res / "organisms").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(res / "organisms" / "lock_verification.csv", index=False)
    (res / "organisms" / "gate2_verdict.json").write_text(json.dumps(
        {"verdict": {"passed": True, "n_checks": 22, "n_failed": 0, "checks": [], "paired_cis": []},
         "remediation_history": [], "active_suffix": "", "seeds": SEEDS, "tag": "full"}), encoding="utf-8")


def write_sweep(res: Path) -> None:
    # PW saturates fast, SEM slowly - the shape H1 predicts.
    curves = {"PW": {0: 0.05, 4: 0.55, 16: 0.85, 64: 0.95, 256: 0.98},
              "SEM": {0: 0.03, 4: 0.12, 16: 0.30, 64: 0.62, 256: 0.85}}
    rows, agg = [], []
    for cap in CAPS:
        for arm in ARMS:
            for n, base in curves[arm].items():
                per_seed = []
                for seed in SEEDS:
                    r = jitter(base, 0.05)
                    per_seed.append(r)
                    rows.append(dict(arm=arm, capability=cap, n_demos=n, seed=seed,
                                     acc_elicited=FLOOR[cap] + r * (CEIL[cap] - FLOOR[cap]),
                                     acc_locked_baseline=FLOOR[cap], acc_unlocked_ceiling=CEIL[cap],
                                     lock_gap=CEIL[cap] - FLOOR[cap], recovery=r,
                                     recovery_ci_lo=max(0, r - 0.09), recovery_ci_hi=min(1, r + 0.09),
                                     parse_failure_rate=0.01, train_seconds=60, train_steps=32, n_test=300))
                m = float(np.mean(per_seed))
                agg.append(dict(arm=arm, capability=cap, n_demos=n, n_seeds=2, recovery_mean_over_seeds=m,
                                recovery=m, recovery_lo=max(0, m - 0.1), recovery_hi=min(1, m + 0.1),
                                n_problems_total=600, ci_method="pooled_over_seeds",
                                acc_elicited=0.5, acc_lo=0.45, acc_hi=0.55))
    (res / "elicitation").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(res / "elicitation" / "sweep.csv", index=False)
    pd.DataFrame(agg).to_csv(res / "elicitation" / "sweep_agg.csv", index=False)
    n50 = [dict(arm=a, capability=c, seed=s, n50=3.5 if a == "PW" else 42.0,
                max_recovery=0.98 if a == "PW" else 0.85, reached_50pct=True)
           for a in ARMS for c in CAPS for s in SEEDS]
    pd.DataFrame(n50).to_csv(res / "elicitation" / "n50.csv", index=False)


def write_transfer(res: Path) -> None:
    same = {"PW": 0.95, "SEM": 0.62}
    cross = {"PW": 0.78, "SEM": 0.14}
    rows = []
    for exp in ("cross_capability", "cross_trigger"):
        for arm in ARMS:
            for src, dst in (("MATH", "CODE"), ("CODE", "MATH")):
                for seed in SEEDS:
                    sr, cr = jitter(same[arm], 0.04), jitter(cross[arm], 0.05)
                    rows.append(dict(experiment=exp, arm=arm, organism=arm if exp == "cross_capability" else arm + "_XT",
                                     elicited_on=src, evaluated_cross_on=dst, seed=seed, n_demos=64,
                                     src_family=0, dst_family=0 if exp == "cross_capability" else 1,
                                     same_capability=src, same_acc=0.6, same_acc_locked=FLOOR[src],
                                     same_acc_unlocked=CEIL[src], same_recovery=sr,
                                     same_recovery_lo=sr - 0.08, same_recovery_hi=sr + 0.08,
                                     same_parse_failure_rate=0.01,
                                     cross_capability=dst, cross_acc=0.5, cross_acc_locked=FLOOR[dst],
                                     cross_acc_unlocked=CEIL[dst], cross_recovery=cr,
                                     cross_recovery_lo=max(0, cr - 0.08), cross_recovery_hi=cr + 0.08,
                                     cross_parse_failure_rate=0.01,
                                     transfer_ratio=cr / sr))
    (res / "transfer").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(res / "transfer" / "transfer.csv", index=False)
    fbc = [dict(arm=a, capability=c, seed=s, acc_elicited=0.5, acc_locked=FLOOR[c], acc_unlocked=CEIL[c],
                recovery=jitter(0.4 if a == "PW" else 0.1, 0.05), recovery_lo=0.1, recovery_hi=0.6,
                parse_failure_rate=0.01, rounds_run=2)
           for a in ARMS for c in CAPS for s in SEEDS]
    pd.DataFrame(fbc).to_csv(res / "transfer" / "filtered_bc.csv", index=False)


def write_interp(res: Path) -> None:
    d = res / "interp"
    d.mkdir(parents=True, exist_ok=True)

    # Cosine: PW high and shared, SEM low. XT arms slightly below their shared-trigger twin.
    peak = {"PW": 0.86, "SEM": 0.24, "PW_XT": 0.79, "SEM_XT": 0.18}
    rows = []
    for arm in GEOM_ARMS:
        for seed in SEEDS:
            for l in range(N_LAYERS):
                shape = np.sin(np.pi * l / (N_LAYERS - 1)) ** 0.6
                rows.append(dict(arm=arm, seed=seed, layer=l,
                                 cosine=float(np.clip(peak[arm] * shape + rng.normal(0, 0.03), -1, 1)),
                                 cap_a="MATH", cap_b="CODE"))
    pd.DataFrame(rows).to_csv(d / "cross_capability_cosine.csv", index=False)
    pd.DataFrame([dict(arm=a, seed=s, layer=20, cosine=peak[a], ci_lo=peak[a] - 0.08,
                       ci_hi=peak[a] + 0.08, n_resamples=2000, cap_a="MATH", cap_b="CODE")
                  for a in GEOM_ARMS for s in SEEDS]).to_csv(d / "cross_capability_cosine_ci.csv", index=False)

    # Participation ratio: PW near 1, SEM higher.
    pr_rows = []
    for arm in GEOM_ARMS:
        base = 1.4 if arm.startswith("PW") else 6.5
        for cap in CAPS:
            for seed in SEEDS:
                for l in range(N_LAYERS):
                    eig = np.array([0.8, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001]) \
                        if arm.startswith("PW") else \
                        np.array([0.25, 0.2, 0.15, 0.12, 0.09, 0.07, 0.05, 0.03, 0.02, 0.02])
                    pr_rows.append(dict(arm=arm, seed=seed, capability=cap, layer=l,
                                        participation_ratio_centered=base * 2.2,
                                        participation_ratio_uncentered=float(base + rng.normal(0, 0.15)),
                                        direction_norm=3.2, heldout_cohens_d=2.1,
                                        top_eigenvalues=[float(x) for x in eig], n_pairs=200))
    pr = pd.DataFrame(pr_rows)
    pr.drop(columns=["top_eigenvalues"]).to_csv(d / "participation_ratio.csv", index=False)
    pr.to_json(d / "participation_ratio_spectra.json", orient="records")
    (d / "selected_layers.json").write_text(json.dumps(
        {"layers": {a: 20 for a in GEOM_ARMS}, "criterion": "validation probe AUC"}), encoding="utf-8")

    # Steering: PW responds, SEM barely; controls flat.
    st, st_agg = [], []
    for arm in ARMS:
        for cap in CAPS:
            for dname in ("unlock_direction", "random_direction", "shuffled_labels"):
                for alpha in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
                    if dname != "unlock_direction" and alpha in (0.0, 0.5, 8.0):
                        continue
                    if dname == "unlock_direction":
                        top = 0.72 if arm == "PW" else 0.16
                        r = top * (1 - np.exp(-alpha)) * (1.0 if alpha <= 4 else 0.75)
                    else:
                        r = abs(rng.normal(0, 0.02))
                    for seed in SEEDS:
                        st.append(dict(arm=arm, capability=cap, seed=seed, layer=20, hook_layer=19,
                                       direction=dname, alpha=alpha, natural_scale=3.2,
                                       acc=0.5, acc_locked=FLOOR[cap], acc_unlocked=CEIL[cap],
                                       recovery=jitter(r, 0.03), recovery_lo=max(0, r - 0.08),
                                       recovery_hi=r + 0.08, parse_failure_rate=0.01, n=150))
                    st_agg.append(dict(arm=arm, capability=cap, direction=dname, alpha=alpha, n_seeds=2,
                                       recovery_mean_over_seeds=r, recovery=r, recovery_lo=max(0, r - 0.09),
                                       recovery_hi=min(1, r + 0.09), n_problems_total=300,
                                       ci_method="pooled_over_seeds", acc_elicited=0.5, acc_lo=0.4, acc_hi=0.6))
    pd.DataFrame(st).to_csv(d / "steering.csv", index=False)
    pd.DataFrame(st_agg).to_csv(d / "steering_agg.csv", index=False)

    # Rank sweep: PW saturates at rank 1-2, SEM climbs slowly.
    rk, rk_agg = [], []
    for arm in ARMS:
        for rank in (1, 2, 4, 8, 16, 32):
            r = (0.9 if rank >= 1 else 0) if arm == "PW" else min(0.9, 0.12 * np.log2(rank + 1) + 0.05)
            for seed in SEEDS:
                rk.append(dict(arm=arm, capability="MATH", seed=seed, lora_rank=rank, n_demos=64,
                               acc=0.6, acc_locked=FLOOR["MATH"], acc_unlocked=CEIL["MATH"],
                               recovery=jitter(r, 0.04), recovery_lo=max(0, r - 0.08),
                               recovery_hi=min(1, r + 0.08), parse_failure_rate=0.01))
            rk_agg.append(dict(arm=arm, capability="MATH", lora_rank=rank, n_seeds=2,
                               recovery_mean_over_seeds=r, recovery=r, recovery_lo=max(0, r - 0.09),
                               recovery_hi=min(1, r + 0.09), n_problems_total=600,
                               ci_method="pooled_over_seeds", acc_elicited=0.6, acc_lo=0.55, acc_hi=0.65))
    pd.DataFrame(rk).to_csv(d / "rank_sweep.csv", index=False)
    pd.DataFrame(rk_agg).to_csv(d / "rank_sweep_agg.csv", index=False)

    # Probes: both arms readable, which is the interesting contrast with steering.
    pb = [dict(arm=a, seed=s, capability=c, layer=l,
               auc=float(np.clip(0.5 + 0.48 * np.sin(np.pi * l / (N_LAYERS - 1)) ** 0.4 + rng.normal(0, 0.01), 0, 1)),
               accuracy=0.9, n_train=400, n_test=200)
          for a in GEOM_ARMS for s in SEEDS for c in CAPS for l in range(N_LAYERS)]
    pd.DataFrame(pb).to_csv(d / "probe_auc.csv", index=False)


def write_timing(res: Path) -> None:
    pd.DataFrame([
        {"stage": "01_build_data[full]", "seconds": 1530, "hours": 0.425, "timestamp_utc": "", "note": ""},
        {"stage": "02a_train_organisms[full]", "seconds": 7000, "hours": 1.944, "timestamp_utc": "", "note": ""},
        {"stage": "02b_verify_locks[full]", "seconds": 4800, "hours": 1.333, "timestamp_utc": "", "note": ""},
        {"stage": "03_elicit_sweep[full]", "seconds": 7200, "hours": 2.0, "timestamp_utc": "", "note": ""},
        {"stage": "06_figures[smoke]", "seconds": 10, "hours": 0.003, "timestamp_utc": "", "note": ""},
    ]).to_csv(res / "timing.csv", index=False)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="slsb_figtest_"))
    res, figs = tmp / "results", tmp / "figures"
    res.mkdir(parents=True); figs.mkdir(parents=True)

    write_lock_verification(res)
    write_sweep(res)
    write_transfer(res)
    write_interp(res)
    write_timing(res)

    fig_mod = import_module("06_figures")
    fig_mod.RES = res
    fig_mod.FIGS = figs
    fig_mod.REPO_ROOT = tmp  # so update_readme cannot touch the real README
    shutil.copy(REPO_ROOT / "README.md", tmp / "README.md")

    made, failed = [], []
    for fn in (fig_mod.fig1_lock_verification, fig_mod.fig2_sample_efficiency,
               fig_mod.fig3_transfer_matrices, fig_mod.fig4_cosine_by_layer,
               fig_mod.fig5_steering, fig_mod.fig6_rank_sweep,
               fig_mod.fig7_participation_ratio, fig_mod.fig8_probe_auc):
        try:
            name = fn()
            (made if name else failed).append(name or f"{fn.__name__} (returned None)")
        except Exception as e:
            failed.append(f"{fn.__name__}: {e!r}")

    table = fig_mod.hypothesis_table()
    try:
        fig_mod.update_readme(table)
        readme_ok = "RESULTS-SUMMARY-START" in (tmp / "README.md").read_text(encoding="utf-8")
    except Exception as e:
        failed.append(f"update_readme: {e!r}")
        readme_ok = False

    print(f"figures rendered: {len(made)}/8")
    for m in made:
        png = figs / f"{m}.png"
        size = png.stat().st_size if png.exists() else 0
        status = "ok" if size > 8000 else "SUSPICIOUSLY SMALL"
        print(f"  {m:34} {size/1024:7.1f} KB  {status}")
        if size <= 8000:
            failed.append(f"{m} rendered but is only {size} bytes")
    for f in failed:
        print(f"  FAIL {f}")

    print(f"\nhypothesis table rows: {len(table)}")
    if len(table):
        print(table[["hypothesis", "metric", "capability", "PW", "SEM", "supported"]].round(3).to_string(index=False))

    # In a world built so the hypotheses hold, every determinate row should say so.
    determinate = table[table["supported"].notna()] if len(table) else table
    unsupported = determinate[determinate["supported"] == False] if len(determinate) else determinate  # noqa: E712
    if len(unsupported):
        print("\nRows that did NOT come out as constructed (indicates a table bug):")
        print(unsupported.to_string(index=False))
        failed.append(f"{len(unsupported)} hypothesis rows disagree with the constructed world")

    print(f"\nREADME update: {'ok' if readme_ok else 'FAILED'}")
    print(f"scratch dir: {tmp}")
    print(f"\n{'ALL OK' if not failed else str(len(failed)) + ' PROBLEM(S)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
