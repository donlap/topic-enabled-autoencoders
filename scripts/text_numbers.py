"""Print the numbers that the text of the paper quotes from Table 1 and Figure 2.

Every gap is computed from the means at full precision and rounded only when the number is printed.
From Table 1, the paper quotes three bounds. The best linear autoencoder is at most 0.001 below the best
collaborative-filtering baseline in NDCG@10 on every dataset. TEA is at most 0.004 below the best method in NDCG@10
on ml-1m and retail. TEA is within 0.01 of the best method in recall@10 on every dataset.
From Figure 2, the paper quotes the decrease of the best validation NDCG@10, relative to its maximum over alpha,
at alpha = -0.5 on ml-1m and at alpha = 0.5 on steam.
The script prints the value of each dataset and then the quoted numbers.
With --json, the script prints the five quoted values at full precision, which scripts/verify.py checks.
Usage: python scripts/text_numbers.py [--json]
"""

import json
import os
import sys

from tea import experiment, render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
BASELINES = ("WMF", "LightGCN", "SLIMElastic", "ADMMSLIM", "NCEPLRec", "RecVAE")
AUTOENCODERS = ("EASE", "RW", "Symmetric", "DAN", "RDLAE", "TEA")


def table1_gaps():
    """Return, for each dataset, the gaps of TEA to the best method and of the autoencoders to the baselines."""
    gaps = {}
    for ds in render.DATASETS:
        means = {}
        for m in render.ORDER:
            r = experiment.load(os.path.join(RESULTS, "cells"), ds, m)
            problem = experiment.check(r, ds, m)
            if problem:
                sys.exit(problem)
            means[m] = {k: float(r[k].mean()) for k in ("ndcg", "recall")}
        best = {k: max(v[k] for v in means.values()) for k in ("ndcg", "recall")}
        gaps[ds] = dict(tea_ndcg=best["ndcg"] - means["TEA"]["ndcg"], tea_recall=best["recall"] - means["TEA"]["recall"],
                        baseline_minus_autoencoder=max(means[m]["ndcg"] for m in BASELINES)
                        - max(means[m]["ndcg"] for m in AUTOENCODERS))
    return gaps


def figure2_decreases():
    """Return, for each dataset with a stored search, the decreases at alpha = -0.5 and at alpha = 0.5."""
    out = {}
    for ds in render.DATASETS:
        path = os.path.join(RESULTS, "search", f"{ds}_TEA.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            curve = json.load(f)["curve"]
        best = {}
        for c in curve:
            a = c["params"]["alpha"]
            best[a] = max(best.get(a, float("-inf")), c["ndcg"])
        top = max(best.values())
        out[ds] = (1 - best[-0.5] / top, 1 - best[0.5] / top)
    return out


def quoted(gaps, decreases):
    """Return the five quoted values at full precision."""
    return dict(baseline_minus_autoencoder=max(g["baseline_minus_autoencoder"] for g in gaps.values()),
                tea_ndcg_ml1m_retail=max(gaps[ds]["tea_ndcg"] for ds in ("ml-1m", "retail")),
                tea_recall=max(g["tea_recall"] for g in gaps.values()),
                decrease_ml1m=decreases["ml-1m"][0], decrease_steam=decreases["steam"][1])


def main():
    """Print the value of each dataset and then the numbers that the text quotes, or the quoted values as JSON."""
    gaps = table1_gaps()
    if "--json" in sys.argv[1:]:
        print(json.dumps(quoted(gaps, figure2_decreases()), indent=1))
        return
    for ds, g in gaps.items():
        print(f"{ds}: TEA below the best by {g['tea_ndcg']:.4f} in NDCG@10 and {g['tea_recall']:.4f} in recall@10. "
              f"Best baseline minus best linear autoencoder in NDCG@10: {g['baseline_minus_autoencoder']:+.4f}")
    decreases = figure2_decreases()
    for ds, (low, high) in decreases.items():
        print(f"{ds}: decrease at alpha = -0.5: {low:.1%}, at alpha = 0.5: {high:.1%}")
    print("Quoted in the text:")
    print(f"best linear autoencoder below the best baseline in NDCG@10 by at most "
          f"{max(max(g['baseline_minus_autoencoder'] for g in gaps.values()), 0.0):.3f}")
    print(f"TEA below the best method in NDCG@10 on ml-1m and retail by at most "
          f"{max(gaps[ds]['tea_ndcg'] for ds in ('ml-1m', 'retail')):.3f}")
    print(f"TEA below the best method in recall@10 by at most {max(g['tea_recall'] for g in gaps.values()):.2f}")
    if "ml-1m" in decreases and "steam" in decreases:
        print(f"decrease at alpha = -0.5 on ml-1m: {decreases['ml-1m'][0]:.0%}")
        print(f"decrease at alpha = 0.5 on steam: {decreases['steam'][1]:.0%}")


if __name__ == "__main__":
    main()
