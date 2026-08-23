"""Single entry point for the whole pipeline.

``python scripts/run_all.py`` runs every stage in order, resuming from the manifests and
halting only at Gate 2 (stage 2), which is the one human checkpoint.

    --stage N            run only stage N (repeatable: --stage 3 --stage 4)
    --smoke              tiny end-to-end run: 20 train / 20 test problems, N in {0,4},
                         1 seed, 1 epoch, interp on 20 prompt pairs. Exercises every
                         stage including figures. Target: under 15 minutes.
    --force              re-run cells already marked done
    --import-results Z   absorb a results.zip produced elsewhere (e.g. a Kaggle run)
    --continue-past-gate proceed past Gate 2 without stopping (only after a human has
                         read the Gate 2 report)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger

log = get_logger("run_all")

STAGES = {
    1: ("00_build_data.py", "data pipeline: frozen splits + cached weak/strong targets"),
    2: ("01_train_organisms.py", "train model organisms (LoRA)"),
    3: ("02_verify_locks.py", "lock verification + GATE 2"),
    4: ("03_elicit_sweep.py", "elicitation sample-efficiency sweep (H1)"),
    5: ("04_cross_transfer.py", "cross-capability / cross-trigger transfer + filtered BC (H2)"),
    6: ("05_interp.py", "mechanistic analysis: directions, steering, rank sweep, probes (H3-H5)"),
    7: ("06_figures.py", "figures, statistics, summary tables"),
}

GATE_STAGE = 3


MAX_STAGE_ATTEMPTS = 3


def run_stage(n: int, args: argparse.Namespace) -> int:
    """Run one stage, retrying on failure.

    A CUDA fault (typically an out-of-memory surfacing as a cuBLAS internal error) kills
    the process and poisons the CUDA context, so it can only be recovered by starting a
    fresh one. Every stage checkpoints per cell, so a retry resumes from the last
    completed cell rather than repeating hours of work - which makes an automatic retry
    the right response to a transient fault, and cheap when it is not transient.

    ``--force`` is applied to the first attempt only: retries must not discard the cells
    the first attempt just completed.
    """
    script, desc = STAGES[n]

    log.info("=" * 70)
    log.info("STAGE %d - %s", n, desc)
    log.info("=" * 70)

    rc = 1
    for attempt in range(1, MAX_STAGE_ATTEMPTS + 1):
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / script)]
        if args.smoke:
            cmd.append("--smoke")
        if args.force and attempt == 1:
            cmd.append("--force")
        if n == GATE_STAGE and args.smoke:
            cmd.append("--no-remediate")

        if attempt > 1:
            log.warning("stage %d: attempt %d of %d (resuming from manifest)", n, attempt, MAX_STAGE_ATTEMPTS)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        log.info("stage %d attempt %d finished rc=%d in %.1f min", n, attempt, rc, (time.time() - t0) / 60)

        # Gate 2 returns a non-zero code to signal "checks failed", which is a result to
        # report, not a crash to retry.
        if rc == 0 or n == GATE_STAGE:
            return rc

    log.error("stage %d failed %d times; giving up", n, MAX_STAGE_ATTEMPTS)
    return rc


def import_results(zip_path: Path) -> None:
    """Absorb a results archive produced on another machine."""
    dest = REPO_ROOT
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(("results/", "figures/"))]
        if not members:
            raise ValueError(f"{zip_path} contains no results/ or figures/ entries")
        zf.extractall(dest, members=members)
    log.info("imported %d entries from %s", len(members), zip_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, action="append", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--import-results", type=str, default=None)
    ap.add_argument("--continue-past-gate", action="store_true")
    args = ap.parse_args()

    if args.import_results:
        import_results(Path(args.import_results))
        return 0

    stages = sorted(args.stage) if args.stage else sorted(STAGES)

    for n in stages:
        if n not in STAGES:
            log.error("unknown stage %d (valid: %s)", n, sorted(STAGES))
            return 1
        rc = run_stage(n, args)

        if n == GATE_STAGE:
            if rc == 0:
                log.info("Gate 2 passed all checks.")
            else:
                log.warning("Gate 2 did NOT pass all checks (rc=%d).", rc)
            if not args.continue_past_gate and not args.smoke and args.stage is None:
                print(
                    "\n"
                    + "=" * 78
                    + "\nHALTING AT GATE 2 - the single human checkpoint.\n"
                    + "Read results/organisms/gate2_report.txt, then re-run with\n"
                    + "  python scripts/run_all.py --continue-past-gate\n"
                    + "to proceed to stage 4 onward.\n"
                    + "=" * 78
                )
                return rc
            continue

        if rc != 0:
            log.error("stage %d failed with rc=%d; stopping", n, rc)
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
