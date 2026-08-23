"""Seeding for every source of nondeterminism we touch."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic_algos: bool = False) -> None:
    """Seed python, numpy, torch (CPU + all CUDA devices) and hash randomisation."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_algos:
        # Not on by default: it makes some attention kernels much slower and a few
        # unavailable.  Generation is greedy everywhere it matters, so the residual
        # nondeterminism is kernel-level reduction order, which we accept and log.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - passed to DataLoader
    import numpy as np
    import torch

    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)
