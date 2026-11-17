"""Seeding. One function, used by every entry point, so a reported seed means
the same thing everywhere in this repository."""
import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=False):
    """Seed python, numpy and torch (CPU and all CUDA devices).

    `deterministic=True` additionally pins cuDNN to deterministic kernels. It
    is OFF by default because the published runs did not use it and it changes
    throughput; seed-to-seed spread in this repository is reported as an
    observed standard deviation over 5 seeds rather than claimed away by
    bitwise determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
