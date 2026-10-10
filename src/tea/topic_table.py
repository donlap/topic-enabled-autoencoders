"""The ranking of the topics shown in Table 2.

Each topic k is scored by TC_k log(n_eff_k) / log(n), where TC_k is the average NPMI over the pairs of
its top ten items, n_eff_k is the effective number of users of these items, and n is the number of users.
"""

import numpy as np

from tea import metrics

EPS = 1e-12


def npmi(items, X):
    """Return the average NPMI over the pairs of the given items, with probabilities over the users of the matrix X."""
    sub = X[:, items]
    U = X.shape[0]
    ps = sub.mean(0)
    co = (sub.T @ sub) / U
    values = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            pij = co[a, b]
            values.append(0.0 if pij <= 0 else np.log(pij / (ps[a] * ps[b] + EPS) + EPS) / (-np.log(pij + EPS)))
    return float(np.mean(values))


def rank(beta, X, n_items=10):
    """Return the topics sorted by score, each with its top items, TC, effective number of users, and score."""
    n = X.shape[0]
    out = []
    for k, items in enumerate(metrics.top_items(beta, n_items)):
        items = [int(i) for i in items]
        tc, neff = npmi(items, X), metrics.effective_users(items, X)
        out.append(dict(topic=k, items=items, tc=tc, neff=neff,
                        score=tc * (np.log(neff) / np.log(n)) if neff > 0 else 0.0))
    return sorted(out, key=lambda t: -t["score"])
