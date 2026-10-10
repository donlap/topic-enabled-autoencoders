"""The temporal split of each user's interactions and the partitions of the users.

For each user, the items of the last timestamp are the test items.
The items of the second-to-last timestamp are the validation items. The remaining items are the training items.
The validation items exclude the training items of the user.
The test items exclude both the training and the validation items.
"""

import numpy as np

FOLDS = 3
FOLD_SEED = 0
HOLDOUT_FRACTION = 1.0 / 3.0


def _user_rows(interactions, n_users):
    order = interactions[:, 0].astype(np.int64)
    bounds = np.searchsorted(order, np.arange(n_users + 1))
    for u in range(n_users):
        s, e = bounds[u], bounds[u + 1]
        if e > s:
            yield u, interactions[s:e]


def temporal_split(prepared):
    """Split the interactions of a prepared dataset by time.

    Returns the binary training matrix X_train and the boolean matrices train_pos, val_pos,
    and test_pos, all of shape (number of users, number of items).
    """
    n_users, n_items = prepared["n_users"], prepared["n_items"]
    counts = np.zeros((n_users, n_items), dtype=np.float32)
    val_pos = np.zeros((n_users, n_items), dtype=bool)
    test_pos = np.zeros((n_users, n_items), dtype=bool)
    for u, rows in _user_rows(prepared["interactions"], n_users):
        ts = rows[:, 2].astype(np.int64)
        items = rows[:, 1].astype(np.int64)
        weights = rows[:, 3].astype(np.float32)
        unique_ts = np.unique(ts)
        if len(unique_ts) < 3:
            for it, c in zip(items, weights):
                counts[u, it] += c
            continue
        train_ts, val_ts, test_ts = set(unique_ts[:-2]), {unique_ts[-2]}, {unique_ts[-1]}
        train_items = set()
        for it, t, c in zip(items, ts, weights):
            if t in train_ts:
                counts[u, it] += c
                train_items.add(it)
        val_items = {it for it, t in zip(items, ts) if t in val_ts} - train_items
        test_items = {it for it, t in zip(items, ts) if t in test_ts} - train_items - val_items
        for it in val_items:
            val_pos[u, it] = True
        for it in test_items:
            test_pos[u, it] = True
    train_pos = counts > 0
    return dict(X_train=(counts > 0).astype(np.float32), train_pos=train_pos,
                val_pos=val_pos, test_pos=test_pos)


def user_folds(n_users, folds=FOLDS, seed=FOLD_SEED):
    """Partition the users into disjoint blocks, and return one pair of masks per block.

    In each pair, the build mask marks the users outside the block, and the score mask marks the users in the block.
    """
    order = np.random.RandomState(seed).permutation(n_users)
    out = []
    for block in np.array_split(order, folds):
        score = np.zeros(n_users, dtype=bool)
        score[block] = True
        out.append((~score, score))
    return out


def holdout_users(n_users, fraction=HOLDOUT_FRACTION, seed=FOLD_SEED):
    """Return the (build, score) masks of the users held out to measure topic quality.

    The held-out users are the first third of the same permutation that user_folds uses.
    """
    order = np.random.RandomState(seed).permutation(n_users)
    score = np.zeros(n_users, dtype=bool)
    score[order[:int(round(fraction * n_users))]] = True
    return ~score, score
