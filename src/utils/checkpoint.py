"""Cell-level checkpointing.

Every experiment is a grid of *cells*.  A cell writes a manifest entry with status
``running`` before it starts and ``done`` when it finishes, so a crashed run resumes
where it stopped instead of restarting.  ``--force`` re-runs completed cells.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger, provenance

log = get_logger("checkpoint")


class Manifest:
    """A JSON file mapping cell-key -> {status, meta, result}."""

    def __init__(self, path: str | Path, stage: str, config: dict[str, Any] | None = None):
        self.path = Path(path)
        self.stage = stage
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "stage": stage,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "provenance": provenance(),
                "config": config or {},
                "cells": {},
            }
            self._flush()
        # Refresh provenance/config on every open so a resumed run records the code
        # version that actually produced the later cells.
        self.data["provenance_latest"] = provenance()
        if config:
            self.data["config"] = config
        self._flush()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def key(**parts: Any) -> str:
        return "|".join(f"{k}={parts[k]}" for k in sorted(parts))

    def is_done(self, key: str) -> bool:
        return self.data["cells"].get(key, {}).get("status") == "done"

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data["cells"].get(key)

    def result(self, key: str) -> Any:
        cell = self.data["cells"].get(key)
        return cell.get("result") if cell else None

    def start(self, key: str, meta: dict[str, Any] | None = None) -> None:
        self.data["cells"][key] = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "meta": meta or {},
        }
        self._flush()

    def done(self, key: str, result: Any = None, seconds: float | None = None) -> None:
        cell = self.data["cells"].setdefault(key, {})
        cell["status"] = "done"
        cell["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if seconds is not None:
            cell["seconds"] = round(seconds, 1)
        cell["result"] = result
        self._flush()

    def fail(self, key: str, error: str) -> None:
        cell = self.data["cells"].setdefault(key, {})
        cell["status"] = "failed"
        cell["error"] = error[:4000]
        cell["failed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._flush()

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cell in self.data["cells"].values():
            counts[cell.get("status", "?")] = counts.get(cell.get("status", "?"), 0) + 1
        return counts

    def done_results(self) -> dict[str, Any]:
        return {
            k: v.get("result")
            for k, v in self.data["cells"].items()
            if v.get("status") == "done"
        }
