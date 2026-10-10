"""Select the number of topics K of each dataset by validation perplexity, and store the curves.

Usage: python scripts/select_k.py [dataset ...]
"""

import argparse
import json
import os

from tea import select_k
from tea.data import build, sources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "k")


def main():
    """Compute the perplexity curves, print the selected K, and store the curves as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(sources.DATASETS))
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for ds in args.datasets:
        curves = select_k.curves(build.load(ds, DATA))
        K, mean = select_k.select(curves)
        with open(os.path.join(OUT, f"{ds}.json"), "w") as f:
            json.dump(dict(K=K, mean_perplexity=mean, per_build_set=curves), f, indent=1)
        print(f"{ds}: K = {K}", flush=True)


if __name__ == "__main__":
    main()
