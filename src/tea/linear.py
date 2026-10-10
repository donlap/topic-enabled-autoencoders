"""Closed-form linear autoencoders: EASE, RDLAE, DAN, and the RW and Symmetric normalizations.

Each function returns the weight matrix B and the inverse P of the regularized Gram matrix.
The scores of the users are X B.
The topics are extracted from -H(P), where H(P) is P with its diagonal set to zero.
"""

import numpy as np

from tea import topics
from tea.base import binarize, make_fit


def ease(X, lam):
    """EASE of Steck (2019): B = I - P diag(1 / diag(P)) with a zero diagonal, where P = (X^T X + lam I)^-1."""
    X = X.astype(np.float64)
    G = X.T @ X
    diag = np.diag_indices_from(G)
    G[diag] += float(lam)
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    B[diag] = 0.0
    return B.astype(np.float32).astype(np.float64), P


def rdlae(X, lam, p, xi):
    """RDLAE of Moon et al. (2023): dropout regularization and the constraint diag(B) <= xi."""
    X = X.astype(np.float64)
    G = X.T @ X
    Lam = (float(p) / (1.0 - float(p))) * np.diag(G) + float(lam)
    P = np.linalg.inv(G + np.diag(Lam))
    p_diag = np.diag(P)
    xi = float(xi)
    mu = np.where(1.0 - p_diag * Lam <= xi, 0.0, (1.0 - xi) / p_diag - Lam)
    return np.eye(X.shape[1]) - P * (Lam + mu)[None, :], P


def dan(X, lam, alpha, user_exponent):
    """DAN of Park et al. (2025) on the Gram matrix D_I^-(1-alpha) X^T D_U^-b X D_I^-alpha, with b the user exponent."""
    X = X.astype(np.float64)
    Xb = (X > 0).astype(np.float64)
    d_i = np.maximum(np.asarray(Xb.sum(0)).reshape(-1), 1.0)
    d_u = np.maximum(np.asarray(Xb.sum(1)).reshape(-1), 1.0)
    a, b = float(alpha), float(user_exponent)
    Xw = X * np.power(d_u, -b / 2.0)[:, None]
    G = np.power(d_i, -(1.0 - a))[:, None] * (Xw.T @ Xw) * np.power(d_i, -a)[None, :]
    P = np.linalg.inv(G + float(lam) * np.eye(X.shape[1]))
    M = P @ G
    B = M - P * (np.diag(M) / np.diag(P))[None, :]
    np.fill_diagonal(B, 0.0)
    return B, P


def _fit(weights, X, K, with_topics):
    Xb = binarize(X)
    B, P = weights(Xb)
    beta = topics.extract(topics.negated_offdiagonal(P), K) if with_topics else None
    return make_fit(Xb @ B, beta)


def fit_ease(X, params, K, with_topics=True, seed=0):
    """Fit EASE with the hyperparameter lambda and extract K topics."""
    return _fit(lambda Xb: ease(Xb, params["lambda"]), X, K, with_topics)


def fit_rdlae(X, params, K, with_topics=True, seed=0):
    """Fit RDLAE with the hyperparameters lambda, p, and xi, and extract K topics."""
    return _fit(lambda Xb: rdlae(Xb, params["lambda"], params["p"], params["xi"]), X, K, with_topics)


def fit_dan(X, params, K, with_topics=True, seed=0):
    """Fit DAN with the hyperparameters lambda, alpha, and user_exponent, and extract K topics."""
    return _fit(lambda Xb: dan(Xb, params["lambda"], params["alpha"], params["user_exponent"]), X, K, with_topics)


def fit_rw(X, params, K, with_topics=True, seed=0):
    """Fit the random-walk normalization, which is DAN with alpha = 0 and user exponent 1, and extract K topics."""
    return _fit(lambda Xb: dan(Xb, params["lambda"], 0.0, 1.0), X, K, with_topics)


def fit_symmetric(X, params, K, with_topics=True, seed=0):
    """Fit the symmetric normalization, which is DAN with alpha = 1/2 and user exponent 1, and extract K topics."""
    return _fit(lambda Xb: dan(Xb, params["lambda"], 0.5, 1.0), X, K, with_topics)
