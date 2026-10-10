"""Run the hyperparameter search of TEA on each dataset and draw Figure 2.

Each panel shows the validation NDCG@10 of every setting of (lambda, p) at each item exponent alpha,
the largest of these values at each alpha, and the selected alpha.
The search fits 180 settings on three folds per dataset.
With --subset, the script fits only the settings that differ from the selected one in a single hyperparameter,
16 per dataset, and stores their validation NDCG@10 for scripts/verify.py.
With --from-reference, the script draws the figure from the search results in reference/figure2.json.
Usage: python scripts/figure2.py [--subset] [--from-reference] [--plot-only] [--datasets ...]
"""

import argparse
import io
import json
import os
import sys

import matplotlib

from tea import experiment, runtime, search
from tea.data import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "search")
DATASETS = ("books", "cds", "gowalla", "ml-1m", "retail", "steam")
ALPHAS = (-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5)
FONTS = (
    ("LaTeX with the Libertine fonts", {"text.usetex": True, "font.family": "serif",
                                                   "text.latex.preamble": r"\usepackage{libertine}"
                                                                          r"\usepackage[libertine]{newtxmath}"}),
    ("LaTeX with Computer Modern", {"text.usetex": True, "font.family": "serif", "text.latex.preamble": ""}),
    ("the fonts of matplotlib", {"text.usetex": False, "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
                                 "mathtext.fontset": "cm"}),
)


def subset(selected):
    """Return the settings of the grid that differ from the selected setting in at most one hyperparameter."""
    return [p for p in search.grid("TEA") if sum(p[k] != selected[k] for k in selected) <= 1]


def run(dataset, only_subset):
    """Run the search of TEA on one dataset, over the full grid or the subset.

    The validation NDCG@10 of every setting is stored as JSON.
    """
    split = build.load(dataset, os.path.join(ROOT, "data"))
    candidates = subset(experiment.config("hyperparameters")["TEA"][dataset]) if only_subset else None
    result = search.search("TEA", split, experiment.config("k")[dataset], candidates)
    os.makedirs(OUT, exist_ok=True)
    name = f"{dataset}_TEA_subset.json" if only_subset else f"{dataset}_TEA.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(result, f, indent=1)
    print(f"{dataset}: selected {result['selected']} among {len(result['curve'])} settings", flush=True)


def curves(dataset, from_reference=False):
    """Return the validation NDCG@10 values at each alpha, their maximum at each alpha, and the selected alpha.

    The validation NDCG@10 of the selected setting is returned as well.
    """
    if from_reference:
        with open(os.path.join(ROOT, "reference", "figure2.json")) as f:
            result = json.load(f)[dataset]
    else:
        with open(os.path.join(OUT, f"{dataset}_TEA.json")) as f:
            result = json.load(f)
    by_alpha = {a: [c["ndcg"] for c in result["curve"] if c["params"]["alpha"] == a] for a in ALPHAS}
    best = max(result["curve"], key=lambda c: c["ndcg"])
    return by_alpha, [max(by_alpha[a]) for a in ALPHAS], result["selected"]["alpha"], best["ndcg"]


def font_settings(plt):
    """Return the first font setup of FONTS that renders a test figure.

    The setups are, in this order, LaTeX with the Libertine fonts, LaTeX with Computer Modern, and the fonts of matplotlib.
    """
    for name, settings in FONTS:
        try:
            with matplotlib.rc_context(settings):
                fig = plt.figure(figsize=(1, 1))
                fig.text(0.5, 0.5, r"$\alpha=0.25$, NDCG@10")
                fig.savefig(io.BytesIO(), format="pdf")
                plt.close(fig)
            return name, settings
        except Exception:
            plt.close("all")
    raise RuntimeError("no font setup renders the figure")


def draw(path, from_reference=False):
    """Draw the six panels of Figure 2 to a PDF file."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    name, settings = font_settings(plt)
    print(f"Figure 2 is drawn with {name}.", flush=True)
    matplotlib.rcParams.update(settings)
    matplotlib.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 12, "xtick.labelsize": 8,
                                "ytick.labelsize": 8, "legend.fontsize": 10, "axes.linewidth": 0.6,
                                "lines.linewidth": 1.2, "xtick.direction": "out", "ytick.direction": "out"})
    usetex = settings["text.usetex"]
    line, cloud, mark = "#1f3a93", "#9aa3ad", "#c0392b"
    fig, axes = plt.subplots(3, 2, figsize=(3.33, 3.8), sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.04, wspace=0.03)
    for ax, ds in zip(axes.ravel(), DATASETS):
        by_alpha, envelope, selected, value = curves(ds, from_reference)
        for a in ALPHAS:
            ax.scatter([a] * len(by_alpha[a]), by_alpha[a], s=4, color=cloud, alpha=0.45, edgecolors="none", zorder=1)
        ax.plot(ALPHAS, envelope, color=line, marker="o", ms=2.4, mfc=line, mec=line, zorder=3)
        ax.axvline(selected, color=mark, ls="--", lw=0.7, alpha=0.7, zorder=2)
        ax.scatter([selected], [value], marker="*", s=70, color=mark, edgecolors="white", linewidths=0.5, zorder=4)
        title = f"\\texttt{{{ds}}}" if usetex else ds
        ax.set_title(f"{title}, $\\alpha^\\star={selected:g}$", pad=3)
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_xticks([-0.25, 0.25], minor=True)
        ax.set_xticklabels([r"$-0.5$", r"$0$", r"$0.5$"])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.margins(x=0.05)
        ax.grid(True, color="0.9", lw=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, pad=1.5)
    fig.supylabel("validation NDCG@10", fontsize=12)
    fig.supxlabel(r"$\alpha$", fontsize=12)
    handles = [Line2D([], [], color=line, marker="o", ms=3.5, label=r"best over $\lambda,\,p$"),
               Line2D([], [], color=cloud, marker="o", ms=3.5, ls="none", label=r"each $(\lambda,\,p)$"),
               Line2D([], [], color=mark, marker="*", ms=8, ls="none", label=r"selected $\alpha^\star$")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, handlelength=1.3,
               handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
    fig.savefig(path, bbox_inches="tight", metadata={"Creator": None, "Producer": None, "CreationDate": None})


def main():
    """Run the searches in child processes, then draw the figure."""
    if runtime.is_child():
        run(sys.argv[1], "--subset" in sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--from-reference", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    if not (args.plot_only or args.from_reference):
        extra = ["--subset"] if args.subset else []
        codes = runtime.run_per_dataset(os.path.abspath(__file__), args.datasets, extra)
        if any(codes.values()):
            sys.exit(f"failed: {[d for d, c in codes.items() if c]}")
        if args.subset:
            return
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    draw(os.path.join(ROOT, "figures", "figure2.pdf"), args.from_reference)
    for ds in DATASETS:
        by_alpha, envelope, selected, _ = curves(ds, args.from_reference)
        top = max(envelope)
        drops = {a: 1.0 - e / top for a, e in zip(ALPHAS, envelope)}
        print(f"{ds}: selected alpha {selected:g}, decrease at alpha = -0.5: {drops[-0.5]:.1%}, "
              f"at alpha = 0.5: {drops[0.5]:.1%}")


if __name__ == "__main__":
    main()
