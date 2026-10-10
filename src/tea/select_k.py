"""Selection of the number of topics K by validation perplexity.

The users are split into three folds, as in the hyperparameter search.
For each fold, the users outside the fold form a build set.
The topics at each K are extracted from the operator -H((X^T X + 250 I)^-1) of EASE fitted on the build set.
The topic proportions of each user of the build set are fitted to the training items of the user by multinomial EM.
The perplexity is then computed on the validation items of the same users.
The selected K minimizes the mean log-perplexity over the three build sets.
"""

import numpy as np

from tea import ppspa
from tea.data.split import user_folds

GRID = (5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 30, 50, 100, 150, 200, 300)
LAMBDA = 250.0
SMOOTH = 0.01
EM_ITERATIONS = 150
EM_TOLERANCE = 1e-7


def ease_operator(X):
    """Return the eigenvalues and eigenvectors of -H((X^T X + 250 I)^-1)."""
    G = X.T @ X
    P = np.linalg.inv(G + LAMBDA * np.eye(G.shape[0]))
    A = -(P - np.diag(np.diag(P)))
    return np.linalg.eigh(0.5 * (A + A.T))


def topics_at(w, V, K):
    """Return the K topics extracted by ppSPA from the top K eigenvectors."""
    idx = np.argsort(w)[::-1][:int(K)]
    beta = ppspa.loadings(V[:, idx]).T
    return beta / beta.sum(axis=1, keepdims=True).clip(min=1e-12)


def fit_theta(X, beta, iterations=EM_ITERATIONS, tolerance=EM_TOLERANCE, eps=1e-12):
    """Maximize the sum over users and items of x_ui log (theta beta)_ui over theta, by multinomial EM."""
    n, K = X.shape[0], beta.shape[0]
    theta = np.full((n, K), 1.0 / K)
    prev = None
    for _ in range(iterations):
        recon = theta @ beta + eps
        theta = theta * ((X / recon) @ beta.T)
        theta = theta / theta.sum(axis=1, keepdims=True).clip(min=eps)
        ll = float((X * np.log(theta @ beta + eps)).sum())
        if prev is not None and abs(ll - prev) <= tolerance * abs(prev):
            break
        prev = ll
    return theta


def perplexity(theta, beta, held_out, eps=1e-12):
    """Return exp(-log-likelihood / number of held-out items) of the held-out items."""
    tokens = float(held_out.sum())
    if tokens <= 0:
        return float("inf")
    return float(np.exp(-float((held_out * np.log(theta @ beta + eps)).sum()) / tokens))


def curves(split, grid=GRID):
    """Return, for each build set, the validation perplexity at each K in the grid smaller than the number of items."""
    X = np.asarray(split["X_train"], dtype=np.float32).astype(np.float64)
    reference = np.asarray(split["train_pos"], dtype=np.float32).astype(np.float64)
    held_out = np.asarray(split["val_pos"], dtype=np.float64)
    n, m = X.shape
    ks = [k for k in grid if k < m]
    out = []
    for build, _ in user_folds(n):
        w, V = ease_operator(X[build])
        row = {}
        for K in ks:
            beta = (1.0 - SMOOTH) * topics_at(w, V, K) + SMOOTH / m
            row[K] = perplexity(fit_theta(reference[build], beta), beta, held_out[build])
        out.append(row)
    return out


def select(per_build_set):
    """Return the K with the smallest mean log-perplexity over the build sets, and the mean perplexity at each K."""
    ks = sorted(per_build_set[0])
    log_ppl = {K: float(np.mean([np.log(c[K]) for c in per_build_set])) for K in ks}
    best = min(ks, key=lambda k: log_ppl[k])
    return int(best), {K: float(np.exp(v)) for K, v in log_ppl.items()}
