"""Weighted matrix factorization of Hu et al. (2008), fitted by alternating least squares."""

import numpy as np

from tea.base import make_fit

CONFIDENCE = 40.0
ITERATIONS = 15


def _solve(factors, weights, K, ridge):
    gram = factors.T @ factors
    out = np.zeros((weights.shape[0], K), dtype=np.float32)
    for r in range(weights.shape[0]):
        w = weights[r]
        nz = np.nonzero(w)[0]
        if nz.size == 0:
            continue
        F = factors[nz]
        A = gram + F.T @ (F * w[nz][:, None]) + ridge
        b = (F * (w[nz] + 1.0)[:, None]).sum(0)
        out[r] = np.linalg.solve(A, b)
    return out


def fit(X, params, K=None, with_topics=False, seed=0):
    """Fit WMF with confidence 1 + 40 x and return the scores U V^T."""
    rng = np.random.RandomState(seed)
    n_users, n_items = X.shape
    k = int(params["factors"])
    weights = (CONFIDENCE * X).astype(np.float32)
    U = (0.01 * rng.randn(n_users, k)).astype(np.float32)
    V = (0.01 * rng.randn(n_items, k)).astype(np.float32)
    ridge = float(params["regularization"]) * np.eye(k, dtype=np.float32)
    for _ in range(ITERATIONS):
        U = _solve(V, weights, k, ridge)
        V = _solve(U, weights.T, k, ridge)
    return make_fit((U @ V.T).astype(np.float32))
