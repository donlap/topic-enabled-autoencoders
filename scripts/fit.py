"""Fit the methods on the datasets at their selected hyperparameters, and store the results.

Each dataset runs in its own process with the BLAS thread count in configs/threads.json.
Usage: python scripts/fit.py [--datasets ...] [--methods ...] [--force]
"""

import argparse
import os
import sys

import numpy as np

from tea import experiment, registry, runtime
from tea.data import build

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "cells")


def fit_dataset(dataset, methods, force):
    """Fit the given methods on one dataset, skipping complete stored results unless force is set.

    A result is complete when its JSON file exists, because experiment.save writes that file last.
    """
    hyper, K = experiment.config("hyperparameters"), experiment.config("k")
    split = build.load(dataset, DATA)
    for name in methods or [m.name for m in registry.METHODS]:
        path = experiment.result_path(RESULTS, dataset, name)
        if os.path.exists(path) and os.path.exists(path[:-4] + ".json") and not force:
            continue
        params = hyper[name][dataset]
        out = experiment.run(name, dataset, split, params, K[dataset])
        experiment.save(RESULTS, dataset, name, out, params, K[dataset])
        tq = f" TQ {out['tq']:.3f}" if "tq" in out else ""
        print(f"{dataset} {name}: NDCG@10 {np.mean(out['ndcg']):.3f} recall@10 {np.mean(out['recall']):.3f}{tq}",
              flush=True)


def main():
    """Fit each requested dataset in a child process."""
    if runtime.is_child():
        dataset, rest = sys.argv[1], sys.argv[2:]
        force = "--force" in rest
        methods = [a for a in rest if a != "--force"]
        fit_dataset(dataset, methods, force)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=["books", "cds", "gowalla", "ml-1m", "retail", "steam"])
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extra = list(args.methods) + (["--force"] if args.force else [])
    codes = runtime.run_per_dataset(os.path.abspath(__file__), args.datasets, extra)
    if any(codes.values()):
        sys.exit(f"failed: {[d for d, c in codes.items() if c]}")


if __name__ == "__main__":
    main()
