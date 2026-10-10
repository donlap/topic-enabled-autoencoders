"""Compare the effective number of users of the topics of TEA and EASE on each dataset.

Both methods are fitted on the training items of all users at their selected hyperparameters.
For each topic, the script computes the effective number of users of the top ten items of the topic.
The script then prints the median over the topics for each method.
Usage: python scripts/neff.py
"""

import json
import os

import numpy as np

from tea import experiment, metrics, registry
from tea.data import build, sources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def effective_users_of_topics(method, split, params, K):
    """Return the effective number of users of the top ten items of each topic."""
    beta = registry.get(method).fit(split["X_train"].astype(np.float64), params, K).beta
    reference = split["train_pos"].astype(np.float64)
    return [metrics.effective_users([int(i) for i in items], reference) for items in metrics.top_items(beta, 10)]


def main():
    """Print the median effective number of users of the topics of TEA and EASE, and store all values as JSON."""
    hyper, K = experiment.config("hyperparameters"), experiment.config("k")
    out = {}
    for ds in sources.DATASETS:
        split = build.load(ds, DATA)
        out[ds] = {m: effective_users_of_topics(m, split, hyper[m][ds], K[ds]) for m in ("TEA", "EASE")}
        print(f"{ds}: median n_eff TEA {np.median(out[ds]['TEA']):.1f}, EASE {np.median(out[ds]['EASE']):.1f}",
              flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "neff.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
