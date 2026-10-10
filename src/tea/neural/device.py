"""The torch device used for training."""

import torch


def default_device():
    """Return the CUDA device if CUDA is available, else the MPS device if MPS is available, else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
