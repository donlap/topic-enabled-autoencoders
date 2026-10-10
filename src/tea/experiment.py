"""Fit one method on one dataset at its selected hyperparameters and store the results.

The ranking metrics come from a fit on all users, with the test items as targets and the training
and validation items excluded. The topics come from a fit on two thirds of the users.
TQ is computed on the training items of the remaining third.
"""

import json
import os

import numpy as np

from tea import metrics, registry
from tea.data import split as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOPK = 10


def config(name, root=ROOT):
    """Load a JSON file from the configs folder."""
    with open(os.path.join(root, "configs", f"{name}.json")) as f:
        return json.load(f)


def _subset(split, users):
    return {k: split[k][users] for k in ("X_train", "train_pos", "val_pos", "test_pos")}


def run(method, dataset, split, params, K, seed=0):
    """Fit a method and return its per-user NDCG@10 and recall@10, and its topics and TQ if the method has topics.

    The entry named finite is true when every score and every topic weight is a finite number.
    """
    m = registry.get(method)
    X = split["X_train"].astype(np.float64)
    extra = {"split": split} if m.kind == "neural" and m.topics else {}
    full = m.fit(X, params, K, with_topics=False, seed=seed, **extra)
    exclude, target = split["train_pos"] | split["val_pos"], split["test_pos"]
    out = dict(ndcg=metrics.ndcg_per_user(full.scores, target, exclude, TOPK),
               recall=metrics.recall_per_user(full.scores, target, exclude, TOPK),
               finite=bool(np.isfinite(full.scores).all()))
    if m.topics:
        build, score = S.holdout_users(X.shape[0])
        sub = _subset(split, build)
        extra = {"split": sub} if m.kind == "neural" else {}
        beta = m.fit(X[build], params, K, with_topics=True, seed=seed, **extra).beta
        out["beta"] = beta
        out["finite"] = out["finite"] and bool(np.isfinite(beta).all())
        out.update(metrics.topic_quality(beta, split["train_pos"].astype(np.float64)[score]))
    return out


def result_path(results, dataset, method):
    """Return the path of the stored result of a method on a dataset."""
    return os.path.join(results, dataset, f"{method}.npz")


def save(results, dataset, method, out, params, K):
    """Store a result together with the hyperparameters and K that produced the result.

    The arrays go to an npz file. The other values and the means of NDCG@10 and recall@10 go to a JSON file.
    Each file is written under a temporary name and then renamed, and the JSON file is written last.
    """
    path = result_path(results, dataset, method)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    values = {k: v if isinstance(v, bool) else float(v) for k, v in out.items() if not isinstance(v, np.ndarray)}
    record = dict(method=method, dataset=dataset, params=params, K=K, ndcg_mean=float(np.mean(out["ndcg"])),
                  recall_mean=float(np.mean(out["recall"])), **values)
    with open(path + ".partial", "wb") as f:
        np.savez(f, **arrays)
    os.replace(path + ".partial", path)
    with open(path[:-4] + ".json.partial", "w") as f:
        json.dump(record, f, indent=1)
    os.replace(path[:-4] + ".json.partial", path[:-4] + ".json")


def check(result, dataset, method, root=ROOT):
    """Return a message if a stored result has non-finite values or other hyperparameters or K than the configs."""
    hyper, K = config("hyperparameters", root), config("k", root)
    if not result["finite"]:
        return f"{dataset} {method}: the scores or the topics are not finite"
    if result["params"] != hyper[method][dataset] or result["K"] != K[dataset]:
        return (f"{dataset} {method}: stored with other hyperparameters or another K than the configs. "
                "Refit this cell with scripts/fit.py --force.")
    return None


def load(results, dataset, method):
    """Load a stored result as a dictionary of arrays and values, in which the arrays take precedence."""
    path = result_path(results, dataset, method)
    with open(path[:-4] + ".json") as f:
        out = json.load(f)
    with np.load(path) as z:
        out.update({k: z[k] for k in z.files})
    return out
