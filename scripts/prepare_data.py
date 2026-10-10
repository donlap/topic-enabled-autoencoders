"""Download the raw data and build the split of each dataset.

Usage: python scripts/prepare_data.py [dataset ...]
"""

import argparse
import os

from tea.data import build, sources

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    """Build the splits of the given datasets, or of all six."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(sources.DATASETS))
    args = parser.parse_args()
    for ds in args.datasets:
        split = build.build(ds, DATA)
        n, m = split["X_train"].shape
        print(f"{ds}: {n} users, {m} items, {int(split['train_pos'].sum())} training interactions")


if __name__ == "__main__":
    main()
