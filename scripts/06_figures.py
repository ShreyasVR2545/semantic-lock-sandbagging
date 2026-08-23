"""Stage 6: figures, statistics, and the summary tables the writeup quotes.

Every figure carries 95% bootstrap CIs, shows individual seeds as well as the aggregate,
uses a colour-blind-safe palette, and has a standalone-readable caption. Each is written
as both PNG (150 dpi) and PDF.

Figures are generated from whatever CSVs exist, so a partial run still produces the
figures it can and reports which it skipped rather than crashing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.bootstrap import bootstrap_paired_diff, pooled_ci_over_seeds
from src.utils.logging import get_logger, record_timing, write_json

log = get_logger("stage6")

FIGS = REPO_ROOT / "figures"
RES = REPO_ROOT / "results"

# Okabe-Ito: colour-blind safe.
C = {
    "PW": "#0072B2",       # blue
    "SEM": "#D55E00",      # vermillion
    "STRONG": "#009E73",   # bluish green
    "WEAK": "#999999",     # grey
    "control": "#CC79A7",  # reddish purple
    "control2": "#F0E442",  # yellow
}
ARM_LABEL = {
    "PW": "PW (password lock)",
    "SEM": "SEM (semantic lock)",
    "PW_XT": "PW, different passwords per capability",
    "SEM_XT": "SEM, different cue families per capability",
}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def save(fig, name: str, caption: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    (FIGS / f"{name}.caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)
    log.info("wrote figures/%s.{png,pdf}", name)


def read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        log.warning("missing %s - skipping the figures that need it", path)
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def _agg_or_mean(agg_df, filters: dict, x_col: str, fallback, lo_hi: tuple[str, str]):
    """Aggregate curve for one series.

    Prefers the stage's ``*_agg.csv``, whose CIs are pooled across seeds exactly as
    preregistered (resample problems within each seed, then average across seeds).
    Falls back to averaging the per-seed CI bounds only when that file is absent, which
    understates seed variance - so the fallback is a degraded mode, not the default.
    """
    if agg_df is not None:
        sub = agg_df
        for k, v in filters.items():
            if k in sub.columns:
                sub = sub[sub[k] == v]
        sub = sub[np.isfinite(sub["recovery"])]
        if len(sub):
            return pd.DataFrame({
                x_col: sub[x_col].values,
                "rec": sub["recovery"].values,
                "lo": sub["recovery_lo"].values,
                "hi": sub["recovery_hi"].values,
            }).sort_values(x_col).reset_index(drop=True)

    lo_col, hi_col = lo_hi
    return (fallback.groupby(x_col)
            .agg(rec=("recovery", "mean"), lo=(lo_col, "mean"), hi=(hi_col, "mean"))
            .reset_index())


# ---------------------------------------------------------------------------


def _verification_pooled_ci() -> dict[tuple, tuple[float, float, float]]:
    """Pooled-across-seeds accuracy CIs for the lock-verification cells.

    Read straight from the stage-2b manifest, which keeps the per-problem score vectors,
    so the interval follows the preregistered rule rather than averaging per-seed bounds.
    Returns {} if the manifest is absent, and the caller degrades gracefully.
    """
    out: dict[tuple, list[list[int]]] = {}
    path = RES / "organisms" / "manifest_verify_full.json"
    if not path.exists():
        return {}
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", {})
    for cell in cells.values():
        if cell.get("status") != "done":
            continue
        r = cell.get("result") or {}
        if not r.get("scores"):
            continue
        out.setdefault((r["arm"], r["cap"], r["condition"], r["split"]), []).append(r["scores"])
    return {k: pooled_ci_over_seeds(v, seed=0) for k, v in out.items()}


def fig1_lock_verification() -> str | None:
    df = read(RES / "organisms" / "lock_verification.csv")
    if df is None:
        return None
    pooled = _verification_pooled_ci()

    caps = sorted(df["capability"].unique())
    fig, axes = plt.subplots(1, len(caps), figsize=(4.6 * len(caps), 3.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, cap in zip(axes, caps):
        sub = df[(df["capability"] == cap) & (df["template_split"] == "heldout")]
        entries = [
            ("STRONG", "neutral", "STRONG\n(ceiling)", C["STRONG"]),
            ("PW", "unlocked", "PW\nunlocked", C["PW"]),
            ("SEM", "unlocked", "SEM\nunlocked", C["SEM"]),
            ("PW", "locked", "PW\nlocked", C["PW"]),
            ("SEM", "locked", "SEM\nlocked", C["SEM"]),
            ("WEAK", "neutral", "WEAK\n(floor)", C["WEAK"]),
        ]
        xs, labels = [], []
        for i, (arm, cond, label, colour) in enumerate(entries):
            rows = sub[(sub["arm"] == arm) & (sub["condition"] == cond)]
            if not len(rows):
                continue
            acc = rows["accuracy"].mean()
            key = (arm, cap, cond, "heldout")
            if key in pooled:
                ci = pooled[key]
                acc, lo, hi = ci.point, ci.lo, ci.hi
            else:
                lo, hi = rows["acc_ci_lo"].mean(), rows["acc_ci_hi"].mean()
            hatch = "//" if cond == "locked" else None
            ax.bar(i, acc, color=colour, alpha=0.85, hatch=hatch, edgecolor="white", linewidth=0.6)
            ax.errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], color="black", capsize=3, lw=1)
            for _, r in rows.iterrows():  # individual seeds
                ax.plot(i, r["accuracy"], "o", color="black", ms=3, alpha=0.55)
            xs.append(i)
            labels.append(label)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(cap)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy on frozen test set")
    fig.suptitle("Fig 1. Lock verification on held-out trigger templates", fontsize=11)

    save(fig, "fig1_lock_verification",
         "Fig 1. Accuracy of each arm on the frozen test set, evaluated with HELD-OUT trigger "
         "templates the lock was never trained on. Hatched bars are the locked condition. Black "
         "bars are 95% bootstrap CIs (10k resamples over problems); dots are individual seeds. "
         "For the PW-vs-SEM comparison downstream to be interpretable, the two unlocked bars must "
         "match each other and STRONG, and the two locked bars must match each other and WEAK.")
    return "fig1_lock_verification"


def fig2_sample_efficiency() -> str | None:
    df = read(RES / "elicitation" / "sweep.csv")
    if df is None:
        return None
    agg_df = read(RES / "elicitation" / "sweep_agg.csv")

    caps = sorted(df["capability"].unique())
    fig, axes = plt.subplots(1, len(caps), figsize=(4.6 * len(caps), 3.6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, cap in zip(axes, caps):
        sub = df[df["capability"] == cap]
        plotted = False
        for arm in ("PW", "SEM"):
            a = sub[(sub["arm"] == arm) & np.isfinite(sub["recovery"])].sort_values("n_demos")
            if not len(a):
                continue
            agg = _agg_or_mean(agg_df, {"arm": arm, "capability": cap}, "n_demos",
                               a, ("recovery_ci_lo", "recovery_ci_hi"))
            x = agg["n_demos"] + 1  # +1 so N=0 is plottable on a log axis
            ax.plot(x, agg["rec"], "-o", color=C[arm], label=ARM_LABEL[arm], lw=1.8, ms=4)
            ax.fill_between(x, agg["lo"], agg["hi"], color=C[arm], alpha=0.16, lw=0)
            for seed, ss in a.groupby("seed"):
                ss = ss.sort_values("n_demos")
                ax.plot(ss["n_demos"] + 1, ss["recovery"], color=C[arm], alpha=0.35, lw=0.8, ls="--")
            plotted = True
        ax.axhline(0.5, color="black", lw=0.7, ls=":", alpha=0.6)
        if plotted:
            # Only a log axis once there is positive data; an all-NaN panel (lock gap
            # below the 5-point floor, so recovery is undefined) would otherwise crash.
            ax.set_xscale("log")
        else:
            ax.text(0.5, 0.5, "recovery undefined\n(lock gap < 5 pts)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="grey")
        ax.set_xlabel("demonstrations N (plotted as N+1)")
        ax.set_title(cap)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("recovery fraction")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Fig 2. Elicitation sample efficiency (H1)", fontsize=11)

    save(fig, "fig2_sample_efficiency",
         "Fig 2. Recovery fraction (elicited - locked) / (unlocked - locked) after SFT on N "
         "demonstrations presented in the locked condition under held-out templates. Solid lines "
         "are the mean over seeds with 95% bootstrap CI bands; dashed lines are individual seeds. "
         "The dotted line marks 50% recovery, whose crossing point defines N50. H1 predicts the "
         "SEM curve rises later than the PW curve.")
    return "fig2_sample_efficiency"


def fig3_transfer_matrices() -> str | None:
    df = read(RES / "transfer" / "transfer.csv")
    if df is None:
        return None

    experiments = sorted(df["experiment"].unique())
    fig, axes = plt.subplots(len(experiments), 2, figsize=(7.6, 3.3 * len(experiments)), squeeze=False)

    for ri, exp in enumerate(experiments):
        for ci_, arm in enumerate(("PW", "SEM")):
            ax = axes[ri][ci_]
            sub = df[(df["experiment"] == exp) & (df["arm"] == arm)]
            caps = sorted(set(sub["elicited_on"]) | set(sub["evaluated_cross_on"]))
            M = np.full((len(caps), len(caps)), np.nan)
            for _, r in sub.groupby(["elicited_on", "evaluated_cross_on"]).mean(numeric_only=True).reset_index().iterrows():
                i, j = caps.index(r["elicited_on"]), caps.index(r["evaluated_cross_on"])
                M[i][j] = r["cross_recovery"]
                M[i][i] = sub[sub["elicited_on"] == r["elicited_on"]]["same_recovery"].mean()

            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(caps)), caps)
            ax.set_yticks(range(len(caps)), caps)
            ax.set_xlabel("evaluated on")
            if ci_ == 0:
                ax.set_ylabel("elicited on")
            ax.set_title(f"{arm} - {exp.replace('_', ' ')}", fontsize=9.5)
            ax.grid(False)
            for i in range(len(caps)):
                for j in range(len(caps)):
                    if np.isfinite(M[i][j]):
                        ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center",
                                color="white" if M[i][j] < 0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, label="recovery fraction")
    fig.suptitle("Fig 3. Transfer matrices (H2)", fontsize=11)

    save(fig, "fig3_transfer_matrices",
         "Fig 3. Recovery fraction when elicitation demonstrations come from the row capability "
         "and evaluation happens on the column capability (N=64, mean over seeds). Diagonal cells "
         "are same-capability elicitation; off-diagonal cells are transfer. 'cross trigger' rows "
         "use organisms where MATH and CODE are gated on DIFFERENT triggers, the direct analogue "
         "of the original paper's different-passwords experiment. H2 predicts bright off-diagonal "
         "cells for PW and dark ones for SEM.")
    return "fig3_transfer_matrices"


def _primary_source(df):
    """Keep only the held-out-template geometry when both sources are present."""
    if df is not None and "is_primary" in df.columns:
        sub = df[df["is_primary"] == True]  # noqa: E712
        if len(sub):
            return sub
    return df


def fig4_cosine_by_layer() -> str | None:
    df = _primary_source(read(RES / "interp" / "cross_capability_cosine.csv"))
    if df is None:
        return None
    ci_df = read(RES / "interp" / "cross_capability_cosine_ci.csv")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    # Solid: shared-trigger organisms (the preregistered PW-vs-SEM comparison).
    # Dotted: cross-trigger organisms, where MATH and CODE are gated on DIFFERENT
    # triggers - the direct analogue of the paper's different-passwords experiment.
    series = [("PW", "-", 2.0, 1.0), ("SEM", "-", 2.0, 1.0),
              ("PW_XT", ":", 1.5, 0.75), ("SEM_XT", ":", 1.5, 0.75)]
    for arm, ls, lw, alpha in series:
        sub = df[df["arm"] == arm]
        if not len(sub):
            continue
        colour = C[arm.replace("_XT", "")]
        label = ARM_LABEL.get(arm, arm.replace("_XT", " (different triggers)"))
        agg = sub.groupby("layer")["cosine"].agg(["mean", "min", "max"]).reset_index()
        ax.plot(agg["layer"], agg["mean"], ls, color=colour, lw=lw, alpha=alpha, label=label)
        if not arm.endswith("_XT"):
            ax.fill_between(agg["layer"], agg["min"], agg["max"], color=colour, alpha=0.18, lw=0)
            for seed, ss in sub.groupby("seed"):
                ax.plot(ss["layer"], ss["cosine"], color=colour, alpha=0.3, lw=0.7, ls="--")

    if ci_df is not None:
        for arm in ("PW", "SEM"):
            r = ci_df[ci_df["arm"] == arm]
            if len(r):
                layer = int(r["layer"].iloc[0])
                m = r["cosine"].mean()
                ax.errorbar(layer, m, yerr=[[m - r["ci_lo"].mean()], [r["ci_hi"].mean() - m]],
                            fmt="D", color=C[arm], ms=6, capsize=4, lw=1.6, zorder=5)
                ax.axvline(layer, color=C[arm], lw=0.7, ls=":", alpha=0.5)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine similarity of MATH and CODE unlock directions")
    ax.set_ylim(-0.35, 1.05)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Fig 4. Is it the SAME direction for both capabilities? (H3b)", fontsize=11)

    save(fig, "fig4_cross_capability_cosine",
         "Fig 4 (HEADLINE). Cosine similarity between the MATH unlock direction and the CODE "
         "unlock direction, per layer, where the unlock direction is the difference in means "
         "between unlocked and locked residual-stream activations over matched prompt pairs. "
         "Diamonds mark the validation-selected layer with a 95% bootstrap CI (directions refitted "
         "inside each resample); shaded bands span seeds. High similarity means one direction "
         "unlocks both capabilities - a single shared switch - which is the mechanism that would "
         "make cross-capability elicitation transfer trivial and therefore uninformative about "
         "elicitation robustness. Directions are fitted on HELD-OUT templates and held-out "
         "problems. H3b predicts PW high, SEM low. Dotted lines are the "
         "cross-trigger organisms, where MATH and CODE are gated on DIFFERENT triggers: if PW "
         "stays high there, the two passwords are wired to one shared switch, which is the "
         "mechanism this project argues explains the original paper's cross-password result.")
    return "fig4_cross_capability_cosine"


def fig5_steering() -> str | None:
    df = read(RES / "interp" / "steering.csv")
    if df is None:
        return None
    agg_df = read(RES / "interp" / "steering_agg.csv")

    caps = sorted(df["capability"].unique())
    fig, axes = plt.subplots(1, len(caps), figsize=(4.6 * len(caps), 3.7), sharey=True)
    axes = np.atleast_1d(axes)
    styles = {"unlock_direction": ("-", 2.0), "random_direction": ("--", 1.2), "shuffled_labels": (":", 1.2)}

    for ax, cap in zip(axes, caps):
        sub = df[df["capability"] == cap]
        for arm in ("PW", "SEM"):
            for dname, (ls, lw) in styles.items():
                a = sub[(sub["arm"] == arm) & (sub["direction"] == dname)]
                if not len(a):
                    continue
                agg = _agg_or_mean(agg_df, {"arm": arm, "capability": cap, "direction": dname}, "alpha",
                                   a, ("recovery_lo", "recovery_hi"))
                label = f"{arm} {dname.replace('_', ' ')}" if dname != "unlock_direction" else f"{arm} unlock dir"
                ax.plot(agg["alpha"], agg["rec"], ls, color=C[arm], lw=lw, marker="o" if dname == "unlock_direction" else None,
                        ms=4, alpha=1.0 if dname == "unlock_direction" else 0.55, label=label)
                if dname == "unlock_direction":
                    ax.fill_between(agg["alpha"], agg["lo"], agg["hi"], color=C[arm], alpha=0.14, lw=0)
        ax.set_xlabel(r"steering strength $\alpha$ (units of the mean lock difference)")
        ax.set_title(cap)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("recovery fraction")
    axes[0].legend(fontsize=6.8, loc="upper left", ncol=2)
    fig.suptitle("Fig 5. Causal steering along the unlock direction (H4)", fontsize=11)

    save(fig, "fig5_steering",
         "Fig 5. Recovery fraction in the locked condition when alpha times the unit unlock "
         "direction is added to the residual stream at the validation-selected layer, scaled so "
         "alpha = 1 adds exactly the mean locked-to-unlocked difference. Solid lines are the true "
         "direction with 95% bootstrap CI bands; dashed and dotted lines are the random-direction "
         "and shuffled-label controls, which must stay near zero for the measurement to mean "
         "anything. The steering layer was TUNED on a validation split by probe AUC. H4 predicts "
         "PW recovers substantially more than SEM.")
    return "fig5_steering"


def fig6_rank_sweep() -> str | None:
    df = read(RES / "interp" / "rank_sweep.csv")
    if df is None:
        return None
    agg_df = read(RES / "interp" / "rank_sweep_agg.csv")

    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    for arm in ("PW", "SEM"):
        sub = df[df["arm"] == arm]
        if not len(sub):
            continue
        agg = _agg_or_mean(agg_df, {"arm": arm}, "lora_rank", sub, ("recovery_lo", "recovery_hi"))
        ax.plot(agg["lora_rank"], agg["rec"], "-o", color=C[arm], lw=1.9, ms=4.5, label=ARM_LABEL[arm])
        ax.fill_between(agg["lora_rank"], agg["lo"], agg["hi"], color=C[arm], alpha=0.16, lw=0)
        for seed, s in sub.groupby("seed"):
            s = s.sort_values("lora_rank")
            ax.plot(s["lora_rank"], s["recovery"], color=C[arm], alpha=0.32, lw=0.8, ls="--")
    ax.axhline(0.5, color="black", lw=0.7, ls=":", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elicitation LoRA rank")
    ax.set_ylabel("recovery fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Fig 6. How many directions does elicitation need? (H5)", fontsize=11)

    save(fig, "fig6_rank_sweep",
         "Fig 6. Recovery fraction at N=64 demonstrations as a function of the rank of the "
         "elicitation LoRA adapter. The organism adapter is merged into the base weights first, so "
         "the swept rank is genuinely the rank of the elicitation update. A lock that is one shared "
         "direction should be undoable at rank 1-2; a lock entangled with task representations "
         "should need more. H5 predicts PW saturates at lower rank than SEM.")
    return "fig6_rank_sweep"


def fig7_participation_ratio() -> str | None:
    df = _primary_source(read(RES / "interp" / "participation_ratio.csv"))
    if df is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    ax = axes[0]
    for arm in ("PW", "SEM"):
        sub = df[df["arm"] == arm]
        if not len(sub):
            continue
        agg = sub.groupby("layer")["participation_ratio_uncentered"].agg(["mean", "min", "max"]).reset_index()
        ax.plot(agg["layer"], agg["mean"], "-", color=C[arm], lw=1.9, label=ARM_LABEL[arm])
        ax.fill_between(agg["layer"], agg["min"], agg["max"], color=C[arm], alpha=0.16, lw=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("participation ratio (uncentered)")
    ax.set_title("Dimensionality of the lock signal", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[1]
    spec_path = RES / "interp" / "participation_ratio_spectra.json"
    if spec_path.exists():
        spectra = [r for r in json.loads(spec_path.read_text(encoding="utf-8"))
                   if r.get("is_primary", True)]
        sel = json.loads((RES / "interp" / "selected_layers.json").read_text(encoding="utf-8"))["layers"] \
            if (RES / "interp" / "selected_layers.json").exists() else {}
        for arm in ("PW", "SEM"):
            layer = int(sel.get(arm, df["layer"].max() // 2))
            rows = [r for r in spectra if r["arm"] == arm and r["layer"] == layer]
            if not rows:
                continue
            eigs = np.mean([r["top_eigenvalues"] for r in rows], axis=0)
            eigs = eigs / max(eigs.sum(), 1e-12)
            ax.plot(range(1, len(eigs) + 1), eigs, "-o", color=C[arm], lw=1.7, ms=4, label=f"{arm} (layer {layer})")
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue index")
        ax.set_ylabel("share of variance")
        ax.set_title("Spectrum at the selected layer", fontsize=9.5)
        ax.legend(fontsize=8)
    fig.suptitle("Fig 7. Participation ratio and eigenvalue spectra (H3a)", fontsize=11)

    save(fig, "fig7_participation_ratio",
         "Fig 7. Left: participation ratio (sum lambda)^2 / sum lambda^2 of the covariance of "
         "per-problem locked-to-unlocked activation difference vectors, per layer, uncentered so "
         "that a perfectly shared single direction gives a value near 1. Bands span seeds and "
         "capabilities. Right: the top-10 normalised eigenvalues at the validation-selected layer, "
         "on a log scale - a steep drop after the first eigenvalue means the lock signal really is "
         "one direction. H3a predicts PW below SEM in both panels.")
    return "fig7_participation_ratio"


def fig8_probe_auc() -> str | None:
    df = read(RES / "interp" / "probe_auc.csv")
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for arm in ("PW", "SEM"):
        sub = df[df["arm"] == arm]
        if not len(sub):
            continue
        agg = sub.groupby("layer")["auc"].agg(["mean", "min", "max"]).reset_index()
        ax.plot(agg["layer"], agg["mean"], "-", color=C[arm], lw=1.9, label=ARM_LABEL[arm])
        ax.fill_between(agg["layer"], agg["min"], agg["max"], color=C[arm], alpha=0.16, lw=0)
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out probe AUC (locked vs unlocked)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Fig 8. Is the lock state readable? (descriptive)", fontsize=11)

    save(fig, "fig8_probe_auc",
         "Fig 8. AUC of a logistic probe trained to classify locked vs unlocked from residual-stream "
         "activations, evaluated on held-out problems. Declared descriptive in the preregistration. "
         "The informative case is high SEM AUC combined with low SEM steering recovery in Fig 5: "
         "that would mean the lock state is READABLE but not CONTROLLABLE by a single direction.")
    return "fig8_probe_auc"


# ---------------------------------------------------------------------------


def _supported(pw: float, sem: float, predicted_larger: str) -> object:
    """True/False when both sides are finite, pandas NA when the comparison is undefined.

    Reporting an undefined comparison as False would silently turn missing data into
    evidence against a hypothesis."""
    if not (np.isfinite(pw) and np.isfinite(sem)):
        return pd.NA
    return bool(sem > pw) if predicted_larger == "SEM" else bool(pw > sem)


def hypothesis_table() -> pd.DataFrame:
    """The confirmatory PW-vs-SEM comparisons, with effect sizes, in one table."""
    rows: list[dict] = []

    sweep = read(RES / "elicitation" / "sweep_agg.csv")
    if sweep is not None:
        sweep = sweep.rename(columns={"recovery": "recovery"})
    else:
        sweep = read(RES / "elicitation" / "sweep.csv")
    n50 = read(RES / "elicitation" / "n50.csv")
    if n50 is not None:
        for cap in sorted(n50["capability"].unique()):
            sub = n50[n50["capability"] == cap]
            pw = sub[sub["arm"] == "PW"]["n50"].mean()
            sem = sub[sub["arm"] == "SEM"]["n50"].mean()
            rows.append({"hypothesis": "H1", "metric": "N50", "capability": cap,
                         "PW": pw, "SEM": sem, "difference_SEM_minus_PW": sem - pw,
                         "direction_predicted": "SEM > PW",
                         "supported": _supported(pw, sem, "SEM")})
    if sweep is not None:
        for cap in sorted(sweep["capability"].unique()):
            for n in sorted(sweep["n_demos"].unique()):
                sub = sweep[(sweep["capability"] == cap) & (sweep["n_demos"] == n)]
                pw = sub[sub["arm"] == "PW"]["recovery"].mean()
                sem = sub[sub["arm"] == "SEM"]["recovery"].mean()
                rows.append({"hypothesis": "H1", "metric": f"recovery@N={n}", "capability": cap,
                             "PW": pw, "SEM": sem, "difference_SEM_minus_PW": sem - pw,
                             "direction_predicted": "PW > SEM",
                             "supported": _supported(pw, sem, "PW")})

    tr = read(RES / "transfer" / "transfer.csv")
    if tr is not None:
        for exp in sorted(tr["experiment"].unique()):
            sub = tr[tr["experiment"] == exp]
            pw = sub[sub["arm"] == "PW"]["transfer_ratio"].mean()
            sem = sub[sub["arm"] == "SEM"]["transfer_ratio"].mean()
            rows.append({"hypothesis": "H2", "metric": f"transfer_ratio ({exp})", "capability": "pooled",
                         "PW": pw, "SEM": sem, "difference_SEM_minus_PW": sem - pw,
                         "direction_predicted": "PW > SEM",
                         "supported": _supported(pw, sem, "PW")})

    pr = _primary_source(read(RES / "interp" / "participation_ratio.csv"))
    sel_path = RES / "interp" / "selected_layers.json"
    layers = json.loads(sel_path.read_text(encoding="utf-8"))["layers"] if sel_path.exists() else {}
    if pr is not None and layers:
        for cap in sorted(pr["capability"].unique()):
            vals = {}
            for arm in ("PW", "SEM"):
                sub = pr[(pr["arm"] == arm) & (pr["capability"] == cap) & (pr["layer"] == int(layers.get(arm, 0)))]
                vals[arm] = sub["participation_ratio_uncentered"].mean()
            rows.append({"hypothesis": "H3a", "metric": "participation ratio @ selected layer", "capability": cap,
                         "PW": vals["PW"], "SEM": vals["SEM"], "difference_SEM_minus_PW": vals["SEM"] - vals["PW"],
                         "direction_predicted": "SEM > PW",
                         "supported": _supported(vals["PW"], vals["SEM"], "SEM")})

    cos = read(RES / "interp" / "cross_capability_cosine_ci.csv")
    if cos is not None:
        pw = cos[cos["arm"] == "PW"]["cosine"].mean()
        sem = cos[cos["arm"] == "SEM"]["cosine"].mean()
        rows.append({"hypothesis": "H3b", "metric": "MATH-CODE direction cosine @ selected layer", "capability": "pooled",
                     "PW": pw, "SEM": sem, "difference_SEM_minus_PW": sem - pw,
                     "direction_predicted": "PW > SEM",
                     "supported": _supported(pw, sem, "PW")})

    st = read(RES / "interp" / "steering.csv")
    if st is not None:
        true_dir = st[st["direction"] == "unlock_direction"]
        for cap in sorted(true_dir["capability"].unique()):
            sub = true_dir[true_dir["capability"] == cap]
            pw = sub[sub["arm"] == "PW"].groupby("alpha")["recovery"].mean().max()
            sem = sub[sub["arm"] == "SEM"].groupby("alpha")["recovery"].mean().max()
            rows.append({"hypothesis": "H4", "metric": "max-over-alpha steering recovery", "capability": cap,
                         "PW": pw, "SEM": sem, "difference_SEM_minus_PW": sem - pw,
                         "direction_predicted": "PW > SEM",
                         "supported": _supported(pw, sem, "PW")})

    rk = read(RES / "interp" / "rank_sweep.csv")
    if rk is not None:
        low = rk[rk["lora_rank"] <= 2]
        for cap in sorted(low["capability"].unique()):
            sub = low[low["capability"] == cap]
            pw = sub[sub["arm"] == "PW"]["recovery"].mean()
            sem = sub[sub["arm"] == "SEM"]["recovery"].mean()
            rows.append({"hypothesis": "H5", "metric": "recovery at LoRA rank <= 2", "capability": cap,
                         "PW": pw, "SEM": sem, "difference_SEM_minus_PW": sem - pw,
                         "direction_predicted": "PW > SEM",
                         "supported": _supported(pw, sem, "PW")})

    return pd.DataFrame(rows)


def parse_failure_table() -> pd.DataFrame | None:
    """Preregistered confound check: parse-failure rate must not differ by arm."""
    df = read(RES / "organisms" / "lock_verification.csv")
    if df is None:
        return None
    return df.groupby(["arm", "capability", "condition"])["parse_failure_rate"].mean().reset_index()


def total_compute() -> tuple[float, str]:
    """Total wall-clock hours from results/timing.csv, plus a per-stage breakdown."""
    path = RES / "timing.csv"
    if not path.exists():
        return float("nan"), "_timing.csv not found_"
    t = pd.read_csv(path)
    t = t[~t["stage"].str.contains("[smoke]", regex=False, na=False)]
    if not len(t):
        return float("nan"), "_no full-run timings recorded_"
    by_stage = t.groupby("stage")["hours"].sum().sort_values(ascending=False)
    lines = ["| stage | GPU-hours |", "|---|---|"]
    lines += [f"| `{k}` | {v:.2f} |" for k, v in by_stage.items()]
    lines.append(f"| **total** | **{by_stage.sum():.2f}** |")
    return float(by_stage.sum()), "\n".join(lines)


def update_readme(table: pd.DataFrame) -> None:
    """Write the headline numbers into README.md between its marker comments."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS-SUMMARY-START -->", "<!-- RESULTS-SUMMARY-END -->"
    if start not in text or end not in text:
        log.warning("README markers missing; not updating the results summary")
        return

    hours, breakdown = total_compute()
    parts: list[str] = []

    gate = RES / "organisms" / "gate2_verdict.json"
    if gate.exists():
        v = json.loads(gate.read_text(encoding="utf-8"))["verdict"]
        status = "passed all checks" if v["passed"] else f"{v['n_failed']} of {v['n_checks']} checks failed"
        parts.append(
            f"**Gate 2 (matched lock strength):** {status}. "
            "See [`results/organisms/gate2_report.txt`](results/organisms/gate2_report.txt)."
        )

    if len(table):
        parts.append("")
        parts.append("**Hypotheses** (point estimates; CIs in the figures and CSVs):")
        parts.append("")
        parts.append("| H | metric | capability | PW | SEM | predicted | holds? |")
        parts.append("|---|---|---|---|---|---|---|")
        for _, r in table.iterrows():
            sup = r["supported"]
            mark = "-" if pd.isna(sup) else ("yes" if sup else "**no**")
            def fmt(x):
                return "n/a" if not np.isfinite(x) else f"{x:.3f}"
            parts.append(
                f"| {r['hypothesis']} | {r['metric']} | {r['capability']} | "
                f"{fmt(r['PW'])} | {fmt(r['SEM'])} | {r['direction_predicted']} | {mark} |"
            )

    headline = FIGS / "fig4_cross_capability_cosine.png"
    if headline.exists():
        parts.append("")
        parts.append("**Headline figure** - is it the same unlock direction for both capabilities?")
        parts.append("")
        parts.append("![Cross-capability unlock-direction cosine similarity](figures/fig4_cross_capability_cosine.png)")
        cap = FIGS / "fig4_cross_capability_cosine.caption.txt"
        if cap.exists():
            parts.append("")
            parts.append(f"*{cap.read_text(encoding='utf-8').strip()}*")

    if np.isfinite(hours):
        parts.append("")
        parts.append(f"**Compute used:** {hours:.1f} GPU-hours on the hardware described in [`HARDWARE.md`](HARDWARE.md).")
        parts.append("")
        parts.append(breakdown)

    body = "\n".join(parts) if parts else "_No results yet._"
    new = text[: text.index(start) + len(start)] + "\n" + body + "\n" + text[text.index(end) :]
    readme.write_text(new, encoding="utf-8")
    log.info("updated README results summary")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []
    for fn in (fig1_lock_verification, fig2_sample_efficiency, fig3_transfer_matrices, fig4_cosine_by_layer,
               fig5_steering, fig6_rank_sweep, fig7_participation_ratio, fig8_probe_auc):
        try:
            name = fn()
        except Exception as e:
            log.error("figure %s failed: %r", fn.__name__, e)
            skipped.append(f"{fn.__name__} (error: {e!r})")
            continue
        (made if name else skipped).append(name or fn.__name__)

    table = hypothesis_table()
    if len(table):
        table.to_csv(RES / "hypothesis_summary.csv", index=False)
        print("\nHYPOTHESIS SUMMARY (point estimates; see figures for CIs)")
        print(table.round(3).to_string(index=False))

    update_readme(table)

    pf = parse_failure_table()
    if pf is not None:
        pf.to_csv(RES / "parse_failure_rates.csv", index=False)

    write_json(RES / "figures_manifest.json", {"made": made, "skipped": skipped})
    log.info("figures made: %s | skipped: %s", made, skipped)
    record_timing(f"06_figures[{'smoke' if args.smoke else 'full'}]", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
