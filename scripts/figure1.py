"""Compute the data of Figure 1 by simulation and draw the figure.

The left panel shows the largest eigenvalues of A-hat for one draw of 150000 users.
The right panel shows the relative error of the K leading eigenvectors of A-hat at each n,
as the mean and one standard deviation over four seeds, and the limit of this error when G* replaces G.
Usage: python scripts/figure1.py [--max-n N]
"""

import argparse
import json
import os

import matplotlib
import numpy as np

from tea import synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute(max_n):
    """Return the data of both panels for each n of the grid up to max_n."""
    beta = synthetic.make_beta(np.random.default_rng(12345))
    grid = [n for n in synthetic.N_GRID if n <= max_n]
    errors = synthetic.recovery(beta, grid)
    return dict(n=grid, eigenvalues=synthetic.eigenvalues(beta), errors=errors,
                mean=[float(np.mean(e)) for e in errors], std=[float(np.std(e)) for e in errors],
                limit=synthetic.limit(beta))


def draw(data, path):
    """Draw the two panels of Figure 1 to a PDF file."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "savefig.dpi": 300, "lines.linewidth": 1.6, "mathtext.fontset": "cm"})
    main, grey, accent = "#2A9D8F", "#8C8C8C", "#E76F51"
    K = synthetic.K
    ev = np.array(data["eigenvalues"])
    idx = np.arange(1, len(ev) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.95))
    ax[0].vlines(idx[:K], 0, ev[:K], color=main, lw=2.2)
    ax[0].plot(idx[:K], ev[:K], "o", color=main, ms=3.5)
    ax[0].vlines(idx[K:], 0, ev[K:], color=grey, lw=2.2)
    ax[0].plot(idx[K:], ev[K:], "o", color=grey, ms=3.5)
    ax[0].axvline(K + 0.5, color=accent, ls=":", lw=1.1)
    ax[0].set_title("$K$ dominant eigenvalues", fontsize=10, fontweight="bold")
    ax[0].set_xlabel("eigenvalue index")
    ax[0].set_ylabel(r"$\lambda_i(\widehat{A})/\lambda_1(\widehat{A})$")
    ax[0].set_ylim(0, 1.05)
    ax[0].annotate(f"$K={K}$", (K + 0.8, 0.88), color=accent, fontsize=9.5)
    ax[1].errorbar(np.array(data["n"], float), data["mean"], yerr=data["std"], color=main, marker="o", ms=3.6,
                   capsize=1.8, zorder=3)
    ax[1].axhline(data["limit"], ls=":", color=accent, lw=1.2, zorder=2)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xticks([1e3, 1e5, 1e7])
    ax[1].set_title("subspace recovery", fontsize=10, fontweight="bold")
    ax[1].set_xlabel("$n$")
    ax[1].set_ylabel(r"$\min_C\Vert S^{\star}-\widehat{V}C\Vert/\Vert S^{\star}\Vert$", fontsize=9)
    ax[1].annotate(r"$\widehat{G}=G^{\!\!\!\star}$", (0.08, 0.06), xycoords="axes fraction", color=accent, fontsize=9.5)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, metadata={"Creator": None, "Producer": None, "CreationDate": None})


def main():
    """Compute the data, store the data as JSON, and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=max(synthetic.N_GRID))
    args = parser.parse_args()
    data = compute(args.max_n)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "figure1.json"), "w") as f:
        json.dump(data, f, indent=1)
    draw(data, os.path.join(ROOT, "figures", "figure1.pdf"))
    print(f"limit {data['limit']:.6f}, relative error at n = {data['n'][0]}: {data['mean'][0]:.4f}, "
          f"at n = {data['n'][-1]}: {data['mean'][-1]:.4f}")


if __name__ == "__main__":
    main()
