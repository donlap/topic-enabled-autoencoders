"""Print the rows of Table 1 from the stored results of scripts/fit.py.

The bold entries come from paired t-tests for NDCG@10 and recall@10 and from a paired bootstrap for TQ.
Usage: python scripts/table1.py [--out FILE]
"""

import argparse
import json
import os
import sys

import numpy as np

from tea import experiment, metrics, render, stats
from tea.data import build, split as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "cells")
DATA = os.path.join(ROOT, "data")


def bold_sets(dataset, results):
    """Return the bold methods of each metric on one dataset."""
    bold = {}
    for metric in ("ndcg", "recall"):
        bold[metric] = stats.not_lower_by_t_test({m: r[metric] for m, r in results.items()}, render.ORDER)
    topics = {m: r for m, r in results.items() if "beta" in r}
    reference = build.load(dataset, DATA)["train_pos"].astype(np.float64)
    _, score = S.holdout_users(reference.shape[0])
    tops = {m: metrics.top_items(r["beta"], 10) for m, r in topics.items()}
    diversities = {m: metrics.diversity(r["beta"], 25) for m, r in topics.items()}
    boot = stats.bootstrap_tq(tops, diversities, reference, np.where(score)[0])
    bold["tq"] = stats.not_lower_by_bootstrap({m: float(r["tq"]) for m, r in topics.items()}, boot, render.ORDER)
    return bold


def check(dataset, results):
    """Stop if a result has non-finite values or was fitted at other hyperparameters or another K than the configs."""
    for m, r in results.items():
        problem = experiment.check(r, dataset, m)
        if problem:
            sys.exit(problem)


def main():
    """Load every stored result, run the tests, and print the rows of Table 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cells, bold = {}, {}
    for ds in render.DATASETS:
        results = {m: experiment.load(RESULTS, ds, m) for m in render.ORDER}
        check(ds, results)
        cells[ds] = {m: dict(ndcg=r["ndcg"].mean(), recall=r["recall"].mean(), tq=r.get("tq")) for m, r in results.items()}
        bold[ds] = bold_sets(ds, results)
    rows = render.table1(cells, bold)
    print(rows, end="")
    if args.out:
        with open(args.out, "w") as f:
            f.write(rows)
        with open(os.path.splitext(args.out)[0] + "_bold.json", "w") as f:
            json.dump(bold, f, indent=1)


if __name__ == "__main__":
    main()
