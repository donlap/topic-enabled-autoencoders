"""Topic-Enabled Autoencoders (TEA).

TEA is a closed-form linear autoencoder with three changes to EASE.
The item normalization weights item i by d_i^(-alpha), where d_i is the number of users of item i.
The user normalization weights user u by q_u = (d_u - 1) / (d_u (d_u + tau)), where d_u is the
number of items of user u. The constant tau is estimated from the data, so tau needs no tuning.
The dropout regularization is Lambda = p / (1 - p) diag(G) + lambda I, where G = X^T X.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from tea import topics
from tea.base import binarize, make_fit


def _power(op, n, shift=0.0, iters=300):
    v = np.ones(n) / np.sqrt(n)
    lam = 0.0
    for _ in range(iters):
        w = op.matvec(v) + shift * v
        nrm = float(np.linalg.norm(w))
        if nrm <= 0:
            return shift
        v = w / nrm
        lam = float(v @ (op.matvec(v) + shift * v))
    return lam


def tau_hat(Xb):
    """Estimate tau as the largest item share of all interactions divided by the top eigenvalue s.

    The value s is the top eigenvalue of the average over users of H(x_u x_u^T) / (d_u (d_u - 1)).
    The eigenvalue solver starts from the vector of ones, so the estimate is the same in every run.
    If the solver does not converge, a shifted power iteration estimates s instead.
    """
    n = Xb.shape[0]
    d = np.maximum(Xb.sum(1), 1.0)
    t_hat = float(Xb.sum(0).max() / max(Xb.sum(), 1.0))
    pair = np.maximum(d * (d - 1.0), 1.0)
    Z = Xb / np.sqrt(pair)[:, None]
    d0 = (Xb / pair[:, None]).sum(0)
    op = LinearOperator((Xb.shape[1], Xb.shape[1]),
                        matvec=lambda v: (Z.T @ (Z @ v) - d0 * v) / n, dtype=np.float64)
    try:
        s1 = float(eigsh(op, k=1, which="LA", v0=np.ones(Xb.shape[1]), return_eigenvectors=False,
                         maxiter=5000, tol=1e-8)[0])
    except Exception:
        mu = _power(op, Xb.shape[1])
        s1 = _power(op, Xb.shape[1], shift=abs(mu)) - abs(mu)
    return t_hat / max(s1, 1e-12)


def user_weights(Xb, tau):
    """Return the user weights q_u = (d_u - 1) / (d_u (d_u + tau)), scaled to mean one."""
    d = np.maximum(Xb.sum(1), 1.0)
    w = np.maximum(d - 1.0, 0.0) / (d * (d + tau))
    if w.sum() <= 0:
        w = np.ones_like(d)
    return w / np.mean(w)


def weights(Xb, alpha, p, lam):
    """Return the weight matrix B of TEA and the inverse P of its regularized Gram matrix."""
    Xw = Xb * np.sqrt(user_weights(Xb, tau_hat(Xb)))[:, None]
    r = np.power(np.maximum(Xb.sum(0), 1.0), -float(alpha))
    G = r[:, None] * (Xw.T @ Xw) * r[None, :]
    Lam = (p / (1.0 - p)) * np.diag(Xb.T @ Xb) + lam if p > 0 else np.full(Xb.shape[1], lam)
    P = np.linalg.inv(G + np.diag(Lam))
    M = P @ G
    B = M - P * (np.diag(M) / np.diag(P))[None, :]
    np.fill_diagonal(B, 0.0)
    return B, P


def fit(X, params, K, with_topics=True, seed=0):
    """Fit TEA on the binarized matrix X and extract K topics."""
    Xb = binarize(X)
    B, P = weights(Xb, params["alpha"], params["p"], float(params["lambda"]))
    beta = topics.extract(topics.negated_offdiagonal(P), K) if with_topics else None
    return make_fit(Xb @ B, beta)
