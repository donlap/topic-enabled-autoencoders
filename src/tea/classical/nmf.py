"""Two topic models from nonnegative matrix factorization (NMF) of the item co-occurrence matrix G = X^T X.

NMF factorizes G itself and takes the topics from its second factor.
NMF-ppSPA factorizes the off-diagonal part of G and takes the topics from ppSPA on its first factor.
"""

import numpy as np
from sklearn.decomposition import NMF

from tea import ppspa
from tea.base import make_fit

ITERATIONS = 400


def _factorize(G, params, K, seed):
    model = NMF(n_components=int(K), init=params["init"], random_state=int(seed),
                alpha_W=float(params["alpha"]), alpha_H=float(params["alpha"]),
                l1_ratio=float(params["l1_ratio"]), max_iter=ITERATIONS, tol=1e-4)
    W = model.fit_transform(G)
    return model, W


def _scores(X, W, beta):
    beta = beta / beta.sum(axis=1, keepdims=True).clip(min=1e-12)
    theta = X @ W
    theta = theta / theta.sum(axis=1, keepdims=True).clip(min=1e-12)
    scores = np.log(theta @ beta + 1e-12)
    return make_fit(scores.astype(np.float32), beta.astype(np.float32))


def fit(X, params, K, with_topics=True, seed=0):
    """Fit NMF to G, with the rows of the second factor as the topics."""
    X = X.astype(np.float64)
    model, W = _factorize(X.T @ X, params, K, seed)
    return _scores(X, W, model.components_)


def fit_ppspa(X, params, K, with_topics=True, seed=0):
    """Fit NMF to the off-diagonal part of G, with the ppSPA loadings of the first factor as the topics."""
    X = X.astype(np.float64)
    G = X.T @ X
    _, W = _factorize(G - np.diag(np.diag(G)), params, K, seed)
    return _scores(X, W, ppspa.loadings(W).T)
