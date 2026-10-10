"""The hyperparameter search by validation NDCG@10.

The users are split into three folds. For each candidate value and each fold, a model is fitted on the training
items of the users outside the fold. The model is scored on the validation items of the same users.
The selected value has the highest mean validation NDCG@10 over the three folds.
A tie is broken in favor of the value that comes first in the grid.
"""

import json
import os

import numpy as np

from tea import metrics, registry
from tea.data.split import user_folds

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def grid(method, root=ROOT):
    """Return the candidate hyperparameters of a method, in the order of the search."""
    with open(os.path.join(root, "configs", "grids.json")) as f:
        spec = json.load(f)[method]
    out = [{}]
    for name, values in spec["axes"]:
        out = [dict(d, **{name: v}) for d in out for v in values]
    if spec["zero_alpha_has_no_l1"]:
        seen, unique = set(), []
        for d in out:
            d = dict(d, l1_ratio=0.0) if d["alpha"] == 0.0 else d
            key = json.dumps(d, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(d)
        out = unique
    return out


def validation_ndcg(method, split, params, K, seed=0):
    """Return the mean validation NDCG@10 over the three folds, and the value of each fold."""
    m = registry.get(method)
    X = split["X_train"].astype(np.float64)
    per_fold = []
    for build, _ in user_folds(X.shape[0]):
        sub = {k: split[k][build] for k in ("X_train", "train_pos", "val_pos", "test_pos")}
        extra = {"split": sub} if m.kind == "neural" and m.topics else {}
        scores = m.fit(X[build], params, K, with_topics=False, seed=seed, **extra).scores
        values = metrics.ndcg_per_user(scores, sub["val_pos"], sub["train_pos"], 10)
        per_fold.append(float(np.mean(values)) if values.size else float("nan"))
    return float(np.nanmean(per_fold)), per_fold


def search(method, split, K, candidates=None):
    """Return the selected hyperparameters and the validation NDCG@10 of every candidate."""
    candidates = grid(method) if candidates is None else candidates
    curve = [dict(params=p, ndcg=validation_ndcg(method, split, p, K)[0]) for p in candidates]
    best = int(np.nanargmax([c["ndcg"] for c in curve]))
    return dict(selected=candidates[best], curve=curve)
