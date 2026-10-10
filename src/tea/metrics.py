"""Ranking metrics and topic metrics.

The ranking metrics are NDCG@k and recall@k with binary relevance.
Each metric is computed for every user with at least one target item.
The items in the exclusion mask are never recommended.
The topic metrics are the coherence TC, the diversity TD, and the effective number of users.
"""

import numpy as np

NEG_INF = -1e30


def _top_k(scores, exclude, k, chunk=512):
    out = np.empty((scores.shape[0], min(k, scores.shape[1])), dtype=np.intp)
    for a in range(0, scores.shape[0], chunk):
        b = min(a + chunk, scores.shape[0])
        s = np.array(scores[a:b], dtype=scores.dtype, copy=True)
        s[exclude[a:b]] = NEG_INF
        out[a:b] = np.argsort(-s, axis=1)[:, :k]
    return out


def recall_per_user(scores, target, exclude, k):
    """Return recall@k of each user with a target item, with min(number of targets, k) as the denominator."""
    n_pos = np.asarray(target).sum(axis=1).astype(np.int64)
    keep = n_pos > 0
    if not keep.any():
        return np.zeros(0)
    top = _top_k(scores[keep], np.asarray(exclude)[keep], k)
    hits = np.take_along_axis(np.asarray(target)[keep], top, axis=1).sum(axis=1)
    return hits / np.minimum(n_pos[keep], k)


def ndcg_per_user(scores, target, exclude, k):
    """Return NDCG@k of each user with a target item, with the ideal ranking over min(number of targets, k) items."""
    n_pos = np.asarray(target).sum(axis=1).astype(np.int64)
    keep = n_pos > 0
    if not keep.any():
        return np.zeros(0)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    top = _top_k(scores[keep], np.asarray(exclude)[keep], k)
    rel = np.take_along_axis(np.asarray(target)[keep], top, axis=1).astype(float)
    dcg = (rel * discounts[:rel.shape[1]]).sum(axis=1)
    table = np.array([discounts[:j].sum() for j in range(k + 1)])
    idcg = table[np.minimum(n_pos[keep], k)]
    return np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), 0.0)


def top_items(beta, n):
    """Return the indices of the n items with the largest weights in each row of beta."""
    return np.argsort(-beta, axis=1)[:, :n]


def coherence(beta, reference, n=10, eps=1e-12):
    """Return TC, the average NPMI over the pairs of the top n items of each topic.

    The probabilities are fractions of the users in the binary matrix reference.
    A pair that no user has in common contributes 0.
    """
    X = np.asarray(reference, dtype=np.float64)
    U = X.shape[0]
    p_single = X.mean(axis=0)
    top = top_items(beta, n)
    n_topics, width = top.shape
    ia, ib = np.triu_indices(width, 1)
    if len(ia) == 0:
        return 0.0
    kept = np.zeros((n_topics, len(ia)))
    for k, items in enumerate(top):
        sub = X[:, items]
        p_ij = (sub.T @ sub)[ia, ib] / U
        ok = p_ij > 0
        pi, pj = p_single[items[ia]], p_single[items[ib]]
        pmi = np.log(p_ij / (pi * pj + eps) + eps)
        kept[k] = np.where(ok, pmi / (-np.log(p_ij + eps)), 0.0)
    return float(kept.mean(axis=1).mean())


def diversity(beta, n=25):
    """Return TD, the number of distinct items among the top n items of all topics, divided by n times the number of topics."""
    top = top_items(beta, n)
    return float(len(np.unique(top)) / (top.shape[0] * n))


def topic_quality(beta, reference, n_coherence=10, n_diversity=25):
    """Return TC, TD, and TQ = TC x TD of the topics beta against the binary matrix reference."""
    tc = coherence(beta, reference, n_coherence)
    td = diversity(beta, n_diversity)
    return dict(tc=tc, td=td, tq=tc * td)


def effective_users(items, reference):
    """Return the effective number of users of a set of items.

    User u contributes c(c - 1) / 2 pairs, where c is the number of the items that u has.
    The effective number is the inverse Simpson index of the users' shares of all pairs.
    The effective number is 1 when one user provides every pair and 0 when no pair occurs.
    """
    c = np.asarray(reference)[:, items].sum(1)
    C = c * (c - 1) / 2.0
    if C.sum() <= 0:
        return 0.0
    w = C / C.sum()
    return float(1.0 / (w ** 2).sum())
