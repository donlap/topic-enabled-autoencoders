"""Print the rows of Table 3, the selected hyperparameters of each method on each dataset.

Usage: python scripts/table3.py
"""

from tea import experiment, render


def main():
    """Print the rows of Table 3 from configs/hyperparameters.json."""
    print(render.table3(experiment.config("hyperparameters")), end="")


if __name__ == "__main__":
    main()
