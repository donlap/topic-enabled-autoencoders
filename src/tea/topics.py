"""Topic extraction from a symmetric item-item matrix."""

import numpy as np

from tea import ppspa


def extract(A, K):
    """Return the K x m topic matrix of a symmetric item-item matrix A.

    ppSPA takes the rows of the matrix of the top K eigenvectors of A. Each topic is then normalized to sum to one.
    """
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    idx = np.argsort(w)[::-1][:int(K)]
    beta = ppspa.loadings(V[:, idx]).T
    return beta / beta.sum(axis=1, keepdims=True).clip(min=1e-12)


def negated_offdiagonal(P):
    """Return the symmetric part of -H(P), where H(P) is P with its diagonal set to zero."""
    A = -(P - np.diag(np.diag(P)))
    return 0.5 * (A + A.T)
