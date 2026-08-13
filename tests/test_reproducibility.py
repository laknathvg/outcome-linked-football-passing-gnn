import numpy as np
import torch
from src.reproducibility import set_global_seed


def test_seed_repeats_numpy_and_torch():
    set_global_seed(42); a_np, a_t = np.random.rand(3), torch.rand(3)
    set_global_seed(42); b_np, b_t = np.random.rand(3), torch.rand(3)
    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_t, b_t)
