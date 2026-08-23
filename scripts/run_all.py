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


def run_stage(n: int, args: argparse.Namespace) -> int:
    script, desc = STAGES[n]
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script)]
    if args.smoke:
        cmd.append("--smoke")
    if args.force:
        cmd.append("--force")
    if n == GATE_STAGE and args.smoke:
        cmd.append("--no-remediate")

    log.info("=" * 70)
    log.info("STAGE %d - %s", n, desc)
    log.info("=" * 70)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    log.info("stage %d finished rc=%d in %.1f min", n, rc, (time.time() - t0) / 60)
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
