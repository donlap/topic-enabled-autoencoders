"""The simulation behind Figure 1.

The topic matrix beta has K = 5 topics over m = 400 items, with one anchor item per topic and the
other columns drawn from a Dirichlet distribution with parameter 0.3. The factor f_u of each user u is drawn from a
Dirichlet distribution with parameter 2 gamma, where gamma_k is proportional to 3^(-(k-1)/(K-1)).
Each interaction of user u with item i is drawn as Bernoulli(0.1 <beta_i, f_u>).
EASE is fitted with lambda = 250.
"""

import numpy as np
from numpy.linalg import eigh, eigvalsh, inv, norm

K = 5
M = 400
LAMBDA = 250.0
SPREAD = 3.0
SCALE = 0.10
SEEDS = (0, 1, 2, 3)
N_GRID = (500, 1500, 5000, 15000, 50000, 150000, 500000, 1500000, 5000000, 10000000)
N_EIGENVALUES = 150000
CHUNK = 200_000


def make_beta(rng):
    """Draw the K x m topic matrix with one anchor item per topic."""
    beta = np.zeros((K, M))
    for k in range(K):
        beta[k, k] = 1.0
    beta[:, K:] = rng.dirichlet(np.full(K, 0.3), size=M - K).T
    return beta


def topic_prior():
    """Return gamma, the topic proportions with ratio 3 between the first and the last topic."""
    pi = SPREAD ** (-np.arange(K) / (K - 1))
    return pi / pi.sum()


def draw(beta, n, rng):
    """Draw n users and return G = X^T X, P^T P, and the column sums of P, where P holds the interaction probabilities.

    The users are drawn in chunks of at most 200000 users to limit the memory.
    """
    alpha = 2.0 * topic_prior()
    G = np.zeros((M, M))
    PP = np.zeros((M, M))
    ps = np.zeros(M)
    for c in range(0, n, CHUNK):
        size = min(CHUNK, n - c)
        P = SCALE * (rng.dirichlet(alpha, size=size) @ beta)
        X = (rng.random(P.shape) < P).astype(np.float32)
        G += (X.T @ X).astype(np.float64)
        PP += P.T @ P
        ps += P.sum(0)
    return G, PP, ps


def operator(G, lam=LAMBDA):
    """Return the symmetric part of A = -H((G + lam I)^-1)."""
    P = inv(G + lam * np.eye(M))
    A = -(P - np.diag(np.diag(P)))
    return 0.5 * (A + A.T)


def top_eigenvectors(G, lam=LAMBDA):
    """Return the K leading eigenvectors of -H((G + lam I)^-1)."""
    w, U = eigh(operator(G, lam))
    return U[:, np.argsort(w)[::-1][:K]]


def s_star(EG, D, lam=LAMBDA):
    """Return the symmetric part of S* = (D + lam I)^-1 - (G* + lam I)^-1."""
    S = np.diag(1.0 / (D + lam)) - inv(EG + lam * np.eye(M))
    return 0.5 * (S + S.T)


def relative_error(S, V):
    """Return min_C ||S - V C|| / ||S|| in the spectral norm, which is attained at C = V^T S."""
    return float(norm(S - V @ (V.T @ S), 2) / norm(S, 2))


def eigenvalues(beta, n=N_EIGENVALUES, seed=0):
    """Return the 2K largest eigenvalues of A-hat divided by the largest, for one draw of n users."""
    G, _, _ = draw(beta, n, np.random.default_rng(seed))
    ev = np.sort(eigvalsh(operator(G)))[::-1]
    return (ev[: 2 * K] / ev[0]).tolist()


def recovery(beta, n_grid=N_GRID, seeds=SEEDS):
    """Return the relative error of the K leading eigenvectors of A-hat at each n, for each seed.

    The matrix G* is the expectation of G given the drawn user factors.
    """
    out = []
    for n in n_grid:
        errors = []
        for s in seeds:
            G, PP, ps = draw(beta, n, np.random.default_rng(1000 + s))
            EG = PP.copy()
            np.fill_diagonal(EG, ps)
            errors.append(relative_error(s_star(EG, ps - np.diag(PP)), top_eigenvectors(G)))
        out.append(errors)
    return out


def limit(beta):
    """Return the limit of the relative error with G* in place of G as n grows, from the Dirichlet moments."""
    alpha = 2.0 * topic_prior()
    a0 = alpha.sum()
    moments = (np.outer(alpha, alpha) + np.diag(alpha)) / (a0 * (a0 + 1))
    C = SCALE ** 2 * beta.T @ moments @ beta
    e1 = SCALE * beta.T @ (alpha / a0)
    EG = C.copy()
    np.fill_diagonal(EG, e1)
    return relative_error(s_star(EG, e1 - np.diag(C), 0.0), top_eigenvectors(EG, 0.0))
