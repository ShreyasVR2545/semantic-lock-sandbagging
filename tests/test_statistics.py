"""Analytic checks on the statistical and geometric machinery.

Every headline claim in this project is a number produced by one of these functions, so
each is checked against a case whose answer is known in closed form rather than against a
previously-recorded output. Runs in a few seconds on CPU; no GPU, no model, no network.

    python tests/test_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.eval.bootstrap import (
    bootstrap_mean,
    bootstrap_paired_diff,
    equivalent_within,
    logistic_fit_n50,
    pooled_paired_statistic,
    pooled_recovery_ci,
    recovery_fraction,
    transfer_ratio,
)
from src.interp.directions import cosine, participation_ratio
from src.interp.probes import _auc, hook_layer_for
from src.eval.graders import extract_code, extract_math_answer

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(f"{name}{(' - ' + detail) if detail else ''}")


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)


# ---------------------------------------------------------------------------
# Participation ratio - H3a rests entirely on this being right
# ---------------------------------------------------------------------------

def test_participation_ratio() -> None:
    rng = np.random.default_rng(0)

    # 1. N identical difference vectors = one shared switch => PR == 1 exactly.
    v = rng.normal(size=64)
    X = np.tile(v, (200, 1))
    pr, eigs = participation_ratio(X, center=False)
    check("PR: identical vectors -> 1", approx(pr, 1.0, 1e-8), f"got {pr:.6f}")

    # 2. Equal energy spread over k orthogonal directions => PR == k exactly.
    for k in (2, 5, 10):
        basis = np.eye(64)[:k]
        reps = 200 // k
        X = np.repeat(basis, reps, axis=0)
        pr, _ = participation_ratio(X, center=False)
        check(f"PR: {k} equal orthogonal directions -> {k}", approx(pr, k, 1e-8), f"got {pr:.6f}")

    # 3. One dominant direction plus small isotropic noise => PR close to 1.
    X = np.tile(v, (200, 1)) + 0.01 * rng.normal(size=(200, 64))
    pr, _ = participation_ratio(X, center=False)
    check("PR: dominant direction + noise -> ~1", 1.0 <= pr < 1.5, f"got {pr:.4f}")

    # 4. Pure isotropic noise => PR near min(n, d), i.e. no shared direction at all.
    X = rng.normal(size=(200, 64))
    pr, _ = participation_ratio(X, center=False)
    check("PR: isotropic noise -> large", pr > 40, f"got {pr:.2f}")

    # 5. Centering removes the shared component entirely, leaving only floating-point
    # residue. That must report as undefined, not as a confident PR of 1.0.
    X = np.tile(v, (200, 1))
    pr_c, _ = participation_ratio(X, center=True)
    check("PR: centered identical vectors -> nan (not a spurious 1.0)",
          not np.isfinite(pr_c), f"got {pr_c}")

    # 6. Centering must still work normally when there IS residual structure.
    X = np.tile(v, (200, 1)) + rng.normal(size=(200, 64))
    pr_c, _ = participation_ratio(X, center=True)
    check("PR: centered real structure stays finite", np.isfinite(pr_c) and pr_c > 10, f"got {pr_c:.2f}")


def test_cosine() -> None:
    a = np.array([1.0, 0.0, 0.0])
    check("cosine: identical -> 1", approx(cosine(a, a), 1.0))
    check("cosine: orthogonal -> 0", approx(cosine(a, np.array([0.0, 1.0, 0.0])), 0.0))
    check("cosine: opposite -> -1", approx(cosine(a, -a), -1.0))
    check("cosine: zero vector -> nan", not np.isfinite(cosine(a, np.zeros(3))))


# ---------------------------------------------------------------------------
# Recovery fraction and transfer ratio
# ---------------------------------------------------------------------------

def test_recovery() -> None:
    check("recovery: halfway -> 0.5", approx(recovery_fraction(0.6, 0.4, 0.8), 0.5))
    check("recovery: at floor -> 0", approx(recovery_fraction(0.4, 0.4, 0.8), 0.0))
    check("recovery: at ceiling -> 1", approx(recovery_fraction(0.8, 0.4, 0.8), 1.0))
    check("recovery: above ceiling clips to 1", approx(recovery_fraction(0.95, 0.4, 0.8), 1.0))
    check("recovery: below floor clips to 0", approx(recovery_fraction(0.1, 0.4, 0.8), 0.0))
    # The preregistered guard: a gap under 5 points makes recovery undefined, not huge.
    check("recovery: gap < 5pts -> nan", not np.isfinite(recovery_fraction(0.45, 0.42, 0.46)))
    check("recovery: negative gap -> nan", not np.isfinite(recovery_fraction(0.5, 0.8, 0.4)))

    check("transfer ratio: half -> 0.5", approx(transfer_ratio(0.3, 0.6), 0.5))
    check("transfer ratio: tiny denominator -> nan", not np.isfinite(transfer_ratio(0.3, 0.01)))


# ---------------------------------------------------------------------------
# Bootstrap behaviour
# ---------------------------------------------------------------------------

def test_bootstrap() -> None:
    rng = np.random.default_rng(1)
    x = (rng.random(400) < 0.6).astype(float)

    ci = bootstrap_mean(x, n_resamples=2000, seed=0)
    check("bootstrap_mean: point == sample mean", approx(ci.point, float(x.mean()), 1e-12))
    check("bootstrap_mean: CI brackets the point", ci.lo < ci.point < ci.hi)
    # Analytic SE for a proportion; the 95% CI half-width should be about 1.96 SE.
    se = np.sqrt(x.mean() * (1 - x.mean()) / len(x))
    check("bootstrap_mean: width ~ 1.96 SE", abs((ci.hi - ci.lo) / 2 - 1.96 * se) < 0.35 * 1.96 * se,
          f"half-width {(ci.hi - ci.lo)/2:.4f} vs {1.96*se:.4f}")

    # Perfectly paired identical vectors: the difference is exactly 0 with zero variance.
    d = bootstrap_paired_diff(x, x, n_resamples=2000, seed=0)
    check("paired diff: identical -> 0 with zero-width CI",
          approx(d.point, 0.0) and approx(d.lo, 0.0) and approx(d.hi, 0.0))

    # A large real difference must exclude zero.
    y = (rng.random(400) < 0.3).astype(float)
    d = bootstrap_paired_diff(x, y, n_resamples=2000, seed=0)
    check("paired diff: real difference excludes 0", d.excludes_zero(), f"CI [{d.lo:.3f}, {d.hi:.3f}]")


def test_pooled_paired_statistic() -> None:
    rng = np.random.default_rng(2)
    n = 300
    a = [(rng.random(n) < 0.7).astype(float) for _ in range(2)]
    b = [(rng.random(n) < 0.45).astype(float) for _ in range(2)]

    ci = pooled_paired_statistic([a, b], lambda x, y: x - y, n_resamples=2000, seed=0)
    expected = float(np.mean([ai.mean() - bi.mean() for ai, bi in zip(a, b)]))
    check("pooled paired: point == mean over seeds", approx(ci.point, expected, 1e-12))
    check("pooled paired: detects a 25pt gap", ci.lo > 0.15, f"CI [{ci.lo:.3f}, {ci.hi:.3f}]")

    # Single-seed terms broadcast across a multi-seed term (this is the WEAK floor case).
    single = [(rng.random(n) < 0.45).astype(float)]
    ci_b = pooled_paired_statistic([a, single], lambda x, y: x - y, n_resamples=1000, seed=0)
    check("pooled paired: broadcasts a 1-seed term", np.isfinite(ci_b.point) and ci_b.lo > 0.1,
          f"CI [{ci_b.lo:.3f}, {ci_b.hi:.3f}]")

    # Mismatched problem counts must raise rather than silently mispair.
    try:
        pooled_paired_statistic([[np.zeros(10)], [np.zeros(11)]], lambda x, y: x - y, n_resamples=10)
        check("pooled paired: rejects mismatched lengths", False, "no exception raised")
    except ValueError:
        check("pooled paired: rejects mismatched lengths", True)


def test_equivalence_gate() -> None:
    from src.eval.bootstrap import CI

    tol = 0.03
    check("equivalence: CI inside band passes", equivalent_within(CI(0.01, -0.01, 0.02, 300), tol))
    check("equivalence: CI straddling the band edge passes",
          equivalent_within(CI(0.04, 0.01, 0.07, 300), tol))
    check("equivalence: CI entirely above the band fails",
          not equivalent_within(CI(0.12, 0.08, 0.16, 300), tol))
    check("equivalence: CI entirely below the band fails",
          not equivalent_within(CI(-0.12, -0.16, -0.08, 300), tol))
    check("equivalence: non-finite CI fails closed",
          not equivalent_within(CI(0.0, float("nan"), float("nan"), 0), tol))


def test_pooled_recovery_ci() -> None:
    rng = np.random.default_rng(3)
    n = 300
    triples = []
    for _ in range(2):
        locked = (rng.random(n) < 0.40).astype(float)
        unlocked = (rng.random(n) < 0.80).astype(float)
        elicited = (rng.random(n) < 0.60).astype(float)  # ~half way
        triples.append((elicited, locked, unlocked))

    ci = pooled_recovery_ci(triples, n_resamples=2000, seed=0)
    check("pooled recovery: ~0.5 for a half-recovered arm", 0.35 < ci.point < 0.65, f"got {ci.point:.3f}")
    check("pooled recovery: CI brackets the point", ci.lo < ci.point < ci.hi)
    check("pooled recovery: CI stays in [0,1]", ci.lo >= 0.0 and ci.hi <= 1.0, f"[{ci.lo:.3f},{ci.hi:.3f}]")


def test_n50() -> None:
    # Build a curve from a known logistic in log10(N+1) and check N50 is recovered.
    for true_n50 in (4.0, 16.0, 64.0):
        b = 3.0
        a = -b * np.log10(true_n50 + 1.0)
        ns = np.array([0, 4, 16, 64, 256], dtype=float)
        rec = 1.0 / (1.0 + np.exp(-(a + b * np.log10(ns + 1.0))))
        got = logistic_fit_n50(ns, rec)
        check(f"N50: recovers {true_n50:g}", approx(got, true_n50, 0.05 * true_n50), f"got {got:.2f}")

    # A flat curve never reaches 50% recovery: report unreachable, do not extrapolate.
    flat = logistic_fit_n50([0, 4, 16, 64, 256], [0.02, 0.02, 0.03, 0.02, 0.03])
    check("N50: flat low curve -> nan", not np.isfinite(flat), f"got {flat}")

    # A curve already saturated at N=0 should give a very small N50, not a negative one.
    high = logistic_fit_n50([0, 4, 16, 64, 256], [0.9, 0.95, 0.97, 0.98, 0.99])
    check("N50: saturated curve -> small non-negative", (not np.isfinite(high)) or high >= 0, f"got {high}")


def test_auc() -> None:
    check("AUC: perfect separation -> 1",
          approx(_auc(np.array([3.0, 4.0, 1.0, 2.0]), np.array([1, 1, 0, 0])), 1.0))
    check("AUC: perfectly wrong -> 0",
          approx(_auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 0])), 0.0))
    # All-constant scores must give 0.5 via tie handling, not 1.0.
    check("AUC: constant scores -> 0.5",
          approx(_auc(np.ones(100), np.array([1] * 50 + [0] * 50)), 0.5))
    check("AUC: single class -> nan", not np.isfinite(_auc(np.arange(4.0), np.ones(4))))


def test_layer_index_mapping() -> None:
    # hidden_states[L+1] is the output of transformer layer L.
    check("hook layer: index 1 -> layer 0", hook_layer_for(1) == 0)
    check("hook layer: index 26 -> layer 25", hook_layer_for(26) == 25)
    check("hook layer: index 0 clamps to 0", hook_layer_for(0) == 0)


def test_extractors() -> None:
    check("math: #### marker", extract_math_answer("blah\n#### 72") == "72")
    check("math: last #### wins", extract_math_answer("#### 5\nno wait\n#### 72") == "72")
    check("math: comma stripped", extract_math_answer("#### 1,234") == "1234")
    check("math: trailing decimal normalised", extract_math_answer("#### 72.00") == "72")
    check("math: prose fallback", extract_math_answer("so the answer is 72.") == "72")
    check("math: no number -> None", extract_math_answer("no digits at all") is None)

    check("code: fenced block", (extract_code("```python\ndef f():\n    return 1\n```") or "").startswith("def f"))
    check("code: prefers a block that defines something",
          "def f" in (extract_code("```\njust text\n```\n```python\ndef f(): return 1\n```") or ""))
    check("code: unfenced fallback", (extract_code("def f():\n    return 1") or "").startswith("def f"))
    check("code: no code -> None", extract_code("I would add the two numbers.") is None)


def main() -> int:
    for fn in (
        test_participation_ratio, test_cosine, test_recovery, test_bootstrap,
        test_pooled_paired_statistic, test_equivalence_gate, test_pooled_recovery_ci,
        test_n50, test_auc, test_layer_index_mapping, test_extractors,
    ):
        fn()

    for line in PASSED:
        print(f"  pass  {line}")
    for line in FAILED:
        print(f"  FAIL  {line}")
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
