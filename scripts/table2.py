"""Print the rows of Table 2, the top topics of TEA fitted on the training items of all users.

The topics are ranked by TC_k log(n_eff_k) / log(n), and the first five are shown.
The labels and the shortened titles in configs/topic_labels.json were written by hand.
The script checks that the top five items of each shown topic are the items listed there.
Usage: python scripts/table2.py
"""

import os

import numpy as np

from tea import experiment, render, tea, topic_table
from tea.data import build

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    """Fit TEA on each dataset, rank its topics, and print the rows of Table 2."""
    hyper, K, labels = experiment.config("hyperparameters"), experiment.config("k"), experiment.config("topic_labels")
    shown = {}
    for ds, _, _ in render.TABLE2:
        split = build.load(ds, DATA)
        beta = tea.fit(split["X_train"].astype(np.float64), hyper["TEA"][ds], K[ds]).beta
        ranked = topic_table.rank(beta, split["train_pos"].astype(np.float64))[:5]
        shown[ds] = []
        for t, label in zip(ranked, labels[ds]):
            items = [str(split["vocab"][i]) for i in t["items"][:5]]
            if items != label["items"]:
                raise SystemExit(f"{ds}: the top items of topic {label['label']!r} differ from configs/topic_labels.json")
            shown[ds].append(dict(label=label["label"], titles=label["titles"], neff=t["neff"]))
    print(render.table2(shown, K), end="")


if __name__ == "__main__":
    main()
