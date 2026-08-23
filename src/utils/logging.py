"""Logging plus the provenance block that goes into every result manifest.

An untraceable result is not a result, so every manifest records the git SHA, the
library versions, the hardware plan and the full config that produced it.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIGURED = False


def get_logger(name: str = "slsb") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("slsb")
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("slsb") else f"slsb.{name}")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


def library_versions() -> dict[str, str]:
    mods = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "numpy",
        "scipy",
        "pandas",
    ):
        try:
            mod = __import__(name)
            mods[name] = str(getattr(mod, "__version__", "?"))
        except Exception:
            mods[name] = "not-installed"
    return mods


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The provenance block embedded in every manifest."""
    from src.utils import hardware

    try:
        hw = hardware.load_plan()
    except Exception:
        hw = {}

    block = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "libraries": library_versions(),
        "hardware_plan": hw,
    }
    if extra:
        block.update(extra)
    return block


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def append_jsonl(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return path


def record_timing(stage: str, seconds: float, note: str = "") -> None:
    """Append one row to results/timing.csv (wall-clock budget tracking)."""
    import csv

    path = REPO_ROOT / "results" / "timing.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["stage", "seconds", "hours", "timestamp_utc", "note"])
        w.writerow(
            [
                stage,
                round(seconds, 1),
                round(seconds / 3600, 3),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                note,
            ]
        )
