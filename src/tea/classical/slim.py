"""SLIMElastic of Ning and Karypis (2011) and ADMMSLIM of Steck et al. (2020).

Both follow the RecBole implementations.
"""

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from tea.base import make_fit


def fit_elastic(X, params, K=None, with_topics=False, seed=0):
    """Fit one non-negative elastic net per item, with the item itself hidden, and return the scores X B."""
    Xs = sp.csr_matrix(X.astype(np.float32))
    X_lil = Xs.tolil()
    model = ElasticNet(alpha=float(params["alpha"]), l1_ratio=float(params["l1_ratio"]), positive=True,
                       fit_intercept=False, copy_X=False, precompute=True, selection="random",
                       max_iter=100, tol=1e-4, random_state=int(seed))
    cols = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for j in range(Xs.shape[1]):
            r = X_lil[:, j].toarray().ravel()
            saved = X_lil[:, j].copy()
            X_lil[:, j] = 0
            model.fit(X_lil.tocsr(), r)
            cols.append(sp.csr_matrix(model.sparse_coef_))
            X_lil[:, j] = saved
    B = sp.vstack(cols).T.toarray().astype(np.float32)
    return make_fit((X.astype(np.float32) @ B).astype(np.float32))


def fit_admm(X, params, K=None, with_topics=False, seed=0, alpha=0.5, rho=4000.0, iterations=100):
    """Fit ADMMSLIM with an L1 penalty lambda1, a popularity-scaled L2 penalty lambda2, and non-negative weights."""
    lambda1, lambda2 = float(params["lambda1"]), float(params["lambda2"])
    X = X.astype(np.float32)
    item_means = X.mean(axis=0).astype(np.float64)
    G = (X.T @ X).astype(np.float64)
    m = G.shape[0]
    reg = lambda2 * np.diag(np.power(np.maximum(item_means, 1e-12), alpha)) + rho * np.eye(m)
    P = np.linalg.inv(G + reg).astype(np.float32)
    B_aux = (P @ G.astype(np.float32)).astype(np.float32)
    Gamma = np.zeros((m, m), dtype=np.float32)
    C = np.zeros((m, m), dtype=np.float32)
    diag_P = np.diag(P).astype(np.float32)
    for _ in range(int(iterations)):
        B_tilde = B_aux + P @ (rho * C - Gamma)
        gamma = np.diag(B_tilde) / (diag_P + 1e-7)
        B = B_tilde - P * gamma
        T = B + Gamma / rho
        C = (np.abs(T) > lambda1 / rho).astype(np.float32) * (np.abs(T) - lambda1 / rho) * np.sign(T)
        C = np.maximum(C, 0.0)
        Gamma += rho * (B - C)
    return make_fit((X @ C).astype(np.float32))
