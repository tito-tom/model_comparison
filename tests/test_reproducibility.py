from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

if sys.platform == "win32":
    sp = os.path.join(sys.exec_prefix, "Lib", "site-packages")
    dirs = [
        sys.exec_prefix,
        os.path.join(sys.exec_prefix, "Library", "bin"),
        os.path.join(sp, "torch", "lib"),
        os.path.join(sp, "numpy.libs"),
        os.path.join(sp, "torchvision"),
        os.path.join(sp, "pandas.libs"),
        os.path.join(sys.exec_prefix, "DLLs"),
    ]
    for d in dirs:
        if os.path.exists(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common.reproducibility import get_generator, seed_worker, set_seed


def test_set_seed_reproducibility():
    """Verify that set_seed ensures identical random sequences across random, numpy, and torch."""
    set_seed(42, deterministic=True)
    py_val1 = random.random()
    np_val1 = np.random.rand(5)
    th_val1 = torch.rand(5)

    set_seed(42, deterministic=True)
    py_val2 = random.random()
    np_val2 = np.random.rand(5)
    th_val2 = torch.rand(5)

    assert py_val1 == py_val2, "Python random values do not match after set_seed"
    assert np.allclose(np_val1, np_val2), "NumPy random values do not match after set_seed"
    assert torch.allclose(th_val1, th_val2), "PyTorch random values do not match after set_seed"


def test_generator_reproducibility():
    """Verify that get_generator yields identical PyTorch random tensor sequences."""
    g1 = get_generator(123)
    t1 = torch.rand(10, generator=g1)

    g2 = get_generator(123)
    t2 = torch.rand(10, generator=g2)

    assert torch.allclose(t1, t2), "Generator tensors do not match for same seed"


def test_dataloader_reproducibility():
    """Verify DataLoader sampling reproducibility with seed_worker and generator."""
    x = torch.arange(100).float()
    dataset = TensorDataset(x)

    g1 = get_generator(42)
    loader1 = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g1,
    )
    b1 = [batch[0] for batch in loader1]

    g2 = get_generator(42)
    loader2 = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g2,
    )
    b2 = [batch[0] for batch in loader2]

    for batch_a, batch_b in zip(b1, b2):
        assert torch.allclose(batch_a, batch_b), "DataLoader batches differ despite fixed seed"


def main():
    print("Running reproducibility tests...")
    test_set_seed_reproducibility()
    print("  [PASS] test_set_seed_reproducibility")
    test_generator_reproducibility()
    print("  [PASS] test_generator_reproducibility")
    test_dataloader_reproducibility()
    print("  [PASS] test_dataloader_reproducibility")
    print("All reproducibility tests passed successfully!")


if __name__ == "__main__":
    main()
