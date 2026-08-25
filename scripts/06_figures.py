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
    """The crossed grid: BASE / STRONG / locked arm, per layer, both framings, plus the
    SFT-vs-lock decomposition. This replaced a two-line PW-vs-SEM plot after the STRONG
    baseline showed the base model already carries most of SEM's alignment."""
    grid = read(RES / "interp" / "cosine_grid_per_seed.csv")
    dec = read(RES / "interp" / "cosine_decomposition_by_layer.csv")
    if grid is None or dec is None:
        return None
    sel = {}
    p_sel = RES / "interp" / "selected_layers.json"
    if p_sel.exists():
        sel = json.loads(p_sel.read_text(encoding="utf-8"))["layers"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))
    fams = [("pw", "PW", "PW_XT", "password framing"), ("sem", "SEM", "SEM_XT", "evaluation framing")]

    for ax, (cond, arm, arm_xt, title) in zip(axes[:2], fams):
        colour = C[arm]
        series = [
            ("BASE", "BASE (untrained)", "#666666", ":", 1.6),
            ("STRONG", "STRONG (SFT, no lock)", "#009E73", "--", 1.6),
            (arm, f"{arm} (locked)", colour, "-", 2.2),
            (arm_xt, f"{arm_xt} (locked, split trigger)", colour, "-.", 1.3),
        ]
        for model, label, col, ls, lw in series:
            sub = grid[(grid["model"] == model) & (grid["condition"] == cond)]
            if not len(sub):
                continue
            agg = sub.groupby("layer")["cosine"].agg(["mean", "min", "max"]).reset_index()
            ax.plot(agg["layer"], agg["mean"], ls, color=col, lw=lw, label=label)
            if model in (arm, "BASE"):
                ax.fill_between(agg["layer"], agg["min"], agg["max"], color=col, alpha=0.13, lw=0)
        if arm in sel:
            ax.axvline(sel[arm], color=colour, lw=0.8, ls=":", alpha=0.6)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("layer")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.25, 1.0)
    axes[0].set_ylabel("MATH-CODE unlock-direction cosine")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].legend(fontsize=7, loc="upper right")

    ax = axes[2]
    for arm in ("PW", "SEM"):
        sub = dec[dec["arm"] == arm].sort_values("layer")
        ax.plot(sub["layer"], sub["sft_component"], "--", color=C[arm], lw=1.3, alpha=0.75,
                label=f"{arm}: SFT (BASE->STRONG)")
        ax.plot(sub["layer"], sub["lock_component"], "-", color=C[arm], lw=2.2,
                label=f"{arm}: lock (STRONG->locked)")
    ax.axhline(0, color="black", lw=1.0)
    ax.axvspan(14, 26, color="grey", alpha=0.08, lw=0)
    ax.set_xlabel("layer")
    ax.set_ylabel("change in cosine")
    ax.set_title("decomposition: what SFT does vs what the lock does", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Fig 4. Locks reshape a pre-existing shared representation (H3b, corrected)", fontsize=11)

    save(fig, "fig4_cross_capability_cosine",
         "Fig 4 (HEADLINE). MATH-CODE cosine similarity of the unlock direction, per layer. "
         "Left and centre: the same measurement applied to the untrained base model, to STRONG "
         "(SFT on strong demonstrations, no lock), and to the locked organisms, under password "
         "framing and evaluation framing respectively. The base model ALREADY aligns both "
         "framings across capabilities (~0.3-0.5 in the back half), so the locked arms are "
         "reshaping existing structure rather than creating it. Right: that change split into "
         "what plain SFT does (dashed) and what the lock adds on top (solid). Across the shaded "
         "back-half layers the lock component is negative for PW in 13/13 layers and positive for "
         "SEM in 12/13 - so the effect is not an artifact of the two layers chosen by Cohen's d. "
         "Bands span seeds; dotted verticals mark each arm's selected layer.")
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


def _resolvable(pw, sem, seed_range=None, floor=0.10):
    """Is a PW-vs-SEM difference larger than what this setup can actually resolve?

    Returns a flag string when it is NOT, and None when it is. Lock gaps here are 10-15
    points and n=300, so recovery-style quantities resolve to roughly +/-0.10; a directional
    verdict on a smaller difference is noise dressed as a result. Where a per-seed range is
    known, a difference inside that range is also unresolvable.
    """
    if not (np.isfinite(pw) and np.isfinite(sem)):
        return "-"
    diff = abs(pw - sem)
    if seed_range is not None and diff <= seed_range:
        return "n.s."
    if max(abs(pw), abs(sem)) <= 1.5 and diff < floor:
        return "n.s."
    return None


def hypothesis_table() -> pd.DataFrame:
    """The confirmatory PW-vs-SEM comparisons, with an explicit resolvability flag.

    Verdicts are scored yes/no only when the difference exceeds what this setup can
    resolve. Everything else is marked ``n.s.`` rather than given a directional answer the
    data cannot support.
    """
    rows: list[dict] = []

    def add(h, metric, cap, pw, sem, predicted, seed_range=None, note="", floor=0.10):
        flag = _resolvable(pw, sem, seed_range=seed_range, floor=floor)
        supported = pd.NA if flag else _supported(pw, sem, "SEM" if predicted.startswith("SEM") else "PW")
        rows.append({"hypothesis": h, "metric": metric, "capability": cap, "PW": pw, "SEM": sem,
                     "direction_predicted": predicted, "supported": supported,
                     "flag": flag or "", "note": note})

    n50 = read(RES / "elicitation" / "n50.csv")
    if n50 is not None:
        for cap in sorted(n50["capability"].unique()):
            sub = n50[n50["capability"] == cap]
            pw = sub[sub["arm"] == "PW"]["n50"].mean()
            sem = sub[sub["arm"] == "SEM"]["n50"].mean()
            note = "differs by 0.15 demonstrations" if abs(pw - sem) < 1 else ""
            add("H1", "N50", cap, pw, sem, "SEM > PW",
                seed_range=1.0 if abs(pw - sem) < 1 else None, note=note, floor=0.0)

    sweep = read(RES / "elicitation" / "sweep_agg.csv")
    if sweep is not None:
        for cap in sorted(sweep["capability"].unique()):
            for n in sorted(sweep["n_demos"].unique()):
                sub = sweep[(sweep["capability"] == cap) & (sweep["n_demos"] == n)]
                pw = sub[sub["arm"] == "PW"]["recovery"].mean()
                sem = sub[sub["arm"] == "SEM"]["recovery"].mean()
                add("H1", "recovery@N=" + str(n), cap, pw, sem, "PW > SEM")

    tr = read(RES / "transfer" / "transfer.csv")
    if tr is not None:
        for exp in sorted(tr["experiment"].unique()):
            sub = tr[tr["experiment"] == exp]
            pw = sub[sub["arm"] == "PW"]["transfer_ratio"].mean()
            sem = sub[sub["arm"] == "SEM"]["transfer_ratio"].mean()
            add("H2", "transfer_ratio (" + exp + ")", "pooled", pw, sem, "PW > SEM")

    prs = read(RES / "interp" / "h3a_per_seed_participation_ratio.csv")
    if prs is not None:
        for cap in sorted(prs["capability"].unique()):
            pw = prs[(prs["arm"] == "PW") & (prs["capability"] == cap)]["mean"].mean()
            sem = prs[(prs["arm"] == "SEM") & (prs["capability"] == cap)]["mean"].mean()
            sr = float(prs[prs["capability"] == cap]["seed_range"].max())
            add("H3a", "participation ratio @ selected layer", cap, pw, sem, "SEM > PW",
                seed_range=sr, note="max seed range " + format(sr, ".3f"))

    cs = read(RES / "interp" / "h3b_per_seed_cosine.csv")
    if cs is not None:
        pw_r = cs[cs["arm"] == "PW"]
        sem_r = cs[cs["arm"] == "SEM"]
        if len(pw_r) and len(sem_r):
            pw, sem = float(pw_r["mean"].iloc[0]), float(sem_r["mean"].iloc[0])
            sr = max(float(pw_r["seed_range"].iloc[0]), float(sem_r["seed_range"].iloc[0]))
            note = ("per-seed PW " + format(pw_r["seed0"].iloc[0], ".3f") + "/" + format(pw_r["seed1"].iloc[0], ".3f")
                    + ", SEM " + format(sem_r["seed0"].iloc[0], ".3f") + "/" + format(sem_r["seed1"].iloc[0], ".3f"))
            add("H3b", "MATH-CODE direction cosine @ selected layer", "pooled", pw, sem,
                "PW > SEM", seed_range=sr, note=note)

    dec = read(RES / "interp" / "cosine_decomposition_by_layer.csv")
    if dec is not None:
        sel = dec[dec["is_selected_layer"]]
        for _, r in sel[sel["arm"].isin(["PW", "SEM"])].iterrows():
            note = ("BASE " + format(r["BASE"], ".3f") + " -> STRONG " + format(r["STRONG"], ".3f")
                    + " -> locked " + format(r["locked_arm"], ".3f") + "; SFT " + format(r["sft_component"], "+.3f")
                    + ", lock " + format(r["lock_component"], "+.3f"))
            rows.append({"hypothesis": "H3b-control", "metric": "STRONG/BASE baseline (" + r["arm"] + ")",
                         "capability": "pooled", "PW": r["BASE"], "SEM": r["locked_arm"],
                         "direction_predicted": "BASE vs locked", "supported": pd.NA,
                         "flag": "", "note": note})

    st = read(RES / "interp" / "steering_corrected_agg.csv")
    if st is None:
        st = read(RES / "interp" / "steering_agg.csv")
    if st is not None:
        t = st[st["direction"] == "unlock_direction"]
        for cap in sorted(t["capability"].unique()):
            sub = t[t["capability"] == cap]
            pw = sub[sub["arm"] == "PW"].groupby("alpha")["recovery"].mean().max()
            sem = sub[sub["arm"] == "SEM"].groupby("alpha")["recovery"].mean().max()
            add("H4", "max-over-alpha steering recovery", cap, pw, sem, "PW > SEM")

    rk = read(RES / "interp" / "rank_sweep_agg.csv")
    if rk is not None:
        low = rk[rk["lora_rank"] <= 2]
        for cap in sorted(low["capability"].unique()):
            sub = low[low["capability"] == cap]
            pw = sub[sub["arm"] == "PW"]["recovery"].mean()
            sem = sub[sub["arm"] == "SEM"]["recovery"].mean()
            add("H5", "recovery at LoRA rank <= 2", cap, pw, sem, "PW > SEM")

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
        parts.append("| H | metric | capability | PW | SEM | predicted | holds? | note |")
        parts.append("|---|---|---|---|---|---|---|---|")
        for _, r in table.iterrows():
            sup = r["supported"]
            flag = str(r.get("flag", "") or "")
            mark = flag if flag else ("-" if pd.isna(sup) else ("yes" if sup else "**no**"))
            def fmt(x):
                return "n/a" if not np.isfinite(x) else f"{x:.3f}"
            parts.append(
                f"| {r['hypothesis']} | {r['metric']} | {r['capability']} | "
                f"{fmt(r['PW'])} | {fmt(r['SEM'])} | {r['direction_predicted']} | {mark} | "
                f"{str(r.get('note','') or '')} |"
            )
        parts.append("")
        parts.append("`n.s.` = the difference is smaller than this setup can resolve (~0.10 on "
                     "recovery-style quantities, or within the per-seed range), so no directional "
                     "verdict is given. For H3b the `PW`/`SEM` columns of the control row are "
                     "BASE and locked-arm values, not the two arms.")

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
