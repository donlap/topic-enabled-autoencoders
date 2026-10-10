"""The output of a fitted model and the binarization of the training matrix."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Fit:
    """The scores of every user-item pair and, for a topic model, the K x m topic matrix."""

    scores: np.ndarray
    beta: np.ndarray | None = None


def binarize(X):
    """Return the 0/1 matrix of the nonzero entries of X, as float64."""
    return (np.asarray(X) > 0).astype(np.float64)


def make_fit(scores, beta=None):
    """Return a Fit with float32 scores and float64 topics."""
    return Fit(np.asarray(scores, dtype=np.float32), None if beta is None else np.asarray(beta, dtype=np.float64))
