"""NCE-PLRec of Wu et al. (2019), following the RecBole implementation."""

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

from tea.base import make_fit


def fit(X, params, K=None, with_topics=False, seed=0, beta=1.0):
    """Fit a ridge regression of X on the projection X V*, where V* comes from an SVD of the NCE-weighted matrix."""
    R = sp.csr_matrix(X.astype(np.float32))
    n_users = R.shape[0]
    popularity = np.asarray(R.sum(axis=0)).ravel().astype(np.float64)
    rows, cols = R.nonzero()
    vals = np.maximum(np.log(n_users / np.power(np.maximum(popularity[cols], 1.0), float(beta))), 0.0)
    D = sp.coo_matrix((vals, (rows, cols)), shape=R.shape).tocsr()
    rank = int(min(int(params["rank"]), min(D.shape) - 1))
    _, sigma, Vt = randomized_svd(D, n_components=rank, n_iter="auto", power_iteration_normalizer="QR",
                                  random_state=int(seed))
    V = (Vt.T * np.sqrt(sigma).astype(np.float32)).astype(np.float32)
    Q = (R @ V).astype(np.float32)
    A = Q.T @ Q + float(params["regularization"]) * np.eye(rank, dtype=np.float32)
    W = np.linalg.solve(A, Q.T @ R.toarray().astype(np.float32)).astype(np.float32)
    return make_fit((Q @ W).astype(np.float32))
