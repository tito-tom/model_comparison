from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for Python, NumPy, PyTorch CPU, and PyTorch CUDA.
    Optionally configure deterministic CUDA and cuDNN algorithms.

    Args:
        seed: Integer random seed (default 42).
        deterministic: Whether to enable deterministic CUDA/cuDNN algorithms (default True).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function ensuring reproducible data loading across threads.

    Args:
        worker_id: ID of the DataLoader worker thread.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Create and seed a PyTorch Generator for DataLoader reproducibility.

    Args:
        seed: Random seed integer.

    Returns:
        Seeded torch.Generator instance.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
