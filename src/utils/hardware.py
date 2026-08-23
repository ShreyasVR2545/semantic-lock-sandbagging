"""Hardware probing and the local-vs-Kaggle execution plan.

Every script reads the resulting ``configs/hardware.yaml`` rather than making its
own device assumptions.  The single hard rule enforced here is that ``bf16`` is
never enabled unless ``torch.cuda.is_bf16_supported()`` returned true.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_YAML = REPO_ROOT / "configs" / "hardware.yaml"


@dataclass
class Probe:
    platform: str
    python: str
    cpu_count: int
    ram_gb: float
    free_disk_gb: float
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    vram_gb: float | None
    compute_capability: str | None
    bf16_supported: bool
    mps_available: bool


def _ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        return -1.0


def _free_disk_gb(path: Path) -> float:
    try:
        return round(shutil.disk_usage(path).free / 1024**3, 2)
    except Exception:
        return -1.0


def probe() -> Probe:
    import torch

    cuda = torch.cuda.is_available()
    name = vram = cc = None
    bf16 = False
    if cuda:
        props = torch.cuda.get_device_properties(0)
        name = props.name
        vram = round(props.total_memory / 1024**3, 2)
        cc = f"{props.major}.{props.minor}"
        try:
            bf16 = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16 = False

    mps = False
    try:
        mps = bool(torch.backends.mps.is_available())
    except Exception:
        mps = False

    return Probe(
        platform=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        ram_gb=_ram_gb(),
        free_disk_gb=_free_disk_gb(REPO_ROOT),
        torch_version=str(torch.__version__),
        cuda_available=cuda,
        cuda_version=str(torch.version.cuda) if cuda else None,
        device_name=name,
        vram_gb=vram,
        compute_capability=cc,
        bf16_supported=bf16,
        mps_available=mps,
    )


def plan_from_probe(p: Probe) -> dict[str, Any]:
    """Apply the decision table in the project brief."""
    base = {
        "execution": "local",
        "device": "cuda" if p.cuda_available else ("mps" if p.mps_available else "cpu"),
        "dtype": "bf16" if p.bf16_supported else ("fp16" if p.cuda_available else "fp32"),
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "weak_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "eval_batch_size": 16,
        "n_seeds": 3,
        "grid": "full",
        "rationale": "",
    }

    cc_major = float(p.compute_capability.split(".")[0]) if p.compute_capability else 0.0
    vram = p.vram_gb or 0.0

    if not p.cuda_available:
        base.update(
            execution="kaggle",
            dtype="fp32",
            n_seeds=2,
            grid="reduced",
            rationale="No CUDA device (MPS or CPU only) -> Kaggle T4 path.",
        )
        return base

    if vram < 6.0:
        base.update(
            execution="kaggle",
            n_seeds=2,
            grid="reduced",
            rationale=f"CUDA present but only {vram} GiB VRAM (<6 GiB) -> Kaggle T4 path.",
        )
    elif vram < 8.0 and cc_major >= 8.0:
        # 6-8 GiB band.  The brief allows either a 0.5B local base model or Kaggle.
        # See DECISIONS.md D-002: we keep the 1.5B base and pay for it with a
        # reduced grid + aggressive memory settings, because a 0.5B base collapses
        # the strong/weak capability gap the whole design depends on.
        base.update(
            execution="local",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            eval_batch_size=32,
            n_seeds=2,
            grid="reduced",
            rationale=(
                f"{vram} GiB VRAM (6-8 GiB band), cc {p.compute_capability}. Local with the "
                "1.5B base retained; reduced grid (2 seeds), batch 2 x grad-accum 4, "
                "gradient checkpointing on. Eval batch 32 chosen by measured throughput "
                "(see HARDWARE.md). See DECISIONS.md D-002."
            ),
        )
    elif vram < 10.0:
        base.update(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            eval_batch_size=8,
            n_seeds=2,
            grid="reduced",
            rationale=f"{vram} GiB VRAM (8-10 GiB band) -> local, reduced grid, checkpointing on.",
        )
    elif vram < 16.0:
        base.update(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            rationale=f"{vram} GiB VRAM (10-16 GiB band) -> local, full grid.",
        )
    else:
        base.update(rationale=f"{vram} GiB VRAM (>=16 GiB) -> local, full grid, batch 8.")

    if p.cuda_available and not p.bf16_supported:
        base["dtype"] = "fp16"
        base["rationale"] += " bf16 unsupported on this device -> fp16."

    return base


def write_hardware_yaml(p: Probe, plan: dict[str, Any]) -> Path:
    HARDWARE_YAML.parent.mkdir(parents=True, exist_ok=True)
    payload = {"probe": asdict(p), "plan": plan}
    HARDWARE_YAML.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return HARDWARE_YAML


def load_plan() -> dict[str, Any]:
    if not HARDWARE_YAML.exists():
        raise FileNotFoundError(
            f"{HARDWARE_YAML} missing. Run `python -m src.utils.hardware` first."
        )
    return yaml.safe_load(HARDWARE_YAML.read_text(encoding="utf-8"))["plan"]


def torch_dtype(plan: dict[str, Any] | None = None):
    import torch

    plan = plan or load_plan()
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[plan["dtype"]]


def assert_dtype_supported(plan: dict[str, Any] | None = None) -> None:
    """Hard guard called at every training entry point."""
    import torch

    plan = plan or load_plan()
    if plan["dtype"] == "bf16":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "hardware.yaml requests bf16 but torch.cuda.is_bf16_supported() is False. "
                "Re-run `python -m src.utils.hardware` on this machine."
            )
    if plan["dtype"] == "fp16" and not torch.cuda.is_available():
        raise RuntimeError("fp16 requested without CUDA.")


if __name__ == "__main__":
    p = probe()
    plan = plan_from_probe(p)
    path = write_hardware_yaml(p, plan)
    print(json.dumps({"probe": asdict(p), "plan": plan}, indent=2))
    print(f"\nwrote {path}")
