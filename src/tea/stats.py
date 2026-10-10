"""The significance tests behind the bold entries of Table 1.

NDCG@10 and recall@10 are compared by paired two-sided t-tests over the test users.
TQ is not an average over users, so TQ is compared by a paired bootstrap over the held-out users.
In each column, a method is bold if the method has the best value or is not significantly lower than the best method
at the 5% level.
"""

import numpy as np
from scipy import stats

LEVEL = 0.05
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0
EPS = 1e-12


def not_lower_by_t_test(per_user, order):
    """Return the methods whose per-user values are not significantly lower than those of the best method.

    per_user maps each method to its per-user values. The best method has the largest mean.
    A tie is broken by the order of the methods.
    """
    means = {m: float(np.mean(per_user[m])) for m in order if m in per_user}
    top = max(means.values())
    best = [m for m in order if means.get(m) == top]
    bold = list(best)
    x = np.asarray(per_user[best[0]], float)
    for m in order:
        if m not in means or m in best:
            continue
        y = np.asarray(per_user[m], float)
        d = x - y
        p = 1.0 if np.all(d == 0) else float(stats.ttest_rel(x, y)[1])
        if not (d.mean() > 0 and p < LEVEL):
            bold.append(m)
    return [m for m in order if m in bold]


def coherence_fast(top, X):
    """Return TC of the topics whose top items are the rows of top, on the binary matrix X.

    The probabilities are computed only on the columns of these items.
    """
    U = X.shape[0]
    union = np.unique(top)
    pos = {int(it): k for k, it in enumerate(union)}
    sub = X[:, union].astype(np.float64)
    ps = sub.mean(0)
    co = (sub.T @ sub) / U
    values = []
    for items in top:
        idx = [pos[int(i)] for i in items]
        v = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                pij = co[idx[a], idx[b]]
                v.append(0.0 if pij <= 0 else np.log(pij / (ps[idx[a]] * ps[idx[b]] + EPS) + EPS) / (-np.log(pij + EPS)))
        values.append(np.mean(v))
    return float(np.mean(values))


def bootstrap_tq(tops, diversities, reference, users, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Return, for each method, the TQ of its topics on each bootstrap resample of the held-out users.

    One resample of the users is shared by all methods in each replicate, so the comparisons are paired.
    """
    rng = np.random.RandomState(seed)
    boot = {m: np.empty(samples) for m in tops}
    for j in range(samples):
        resample = reference[rng.choice(users, size=len(users), replace=True)]
        for m in tops:
            boot[m][j] = coherence_fast(tops[m], resample) * diversities[m]
    return boot


def two_sided_p(d):
    """Return the two-sided bootstrap p-value of the differences d."""
    return float(2 * min((d <= 0).mean(), (d >= 0).mean()))


def not_lower_by_bootstrap(tq, boot, order):
    """Return the methods whose TQ is not significantly lower than the best TQ."""
    methods = [m for m in order if m in tq]
    best = max(methods, key=lambda m: tq[m])
    bold = [best] + [m for m in methods if m != best and two_sided_p(boot[best] - boot[m]) >= LEVEL]
    return [m for m in order if m in bold]
