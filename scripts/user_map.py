"""Make a UMAP plot of the cds users from their TEA scores, colored by their topics.

Usage: python scripts/user_map.py
"""

import itertools
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

from tea import experiment, tea, topic_table
from tea.data import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "cds"
NEIGHBORS, MIN_DIST, SEED = 15, 0.05, 0
EXPONENT = 2
CHROMA, FIRST_HUE = 0.16, 25.0
THEMES = (("dark", "#0d1117", 0.76), ("light", "#ffffff", 0.62))


def topic_weights(X, beta):
    """Return the average topic shares of the items of each user."""
    shares = beta / beta.sum(axis=0, keepdims=True)
    return (X @ shares.T) / X.sum(axis=1, keepdims=True)


def hue_order(weights):
    """Assign evenly spaced hues to the topics so that topics that share users get distant hues."""
    K = weights.shape[1]
    shared = weights.T @ weights
    np.fill_diagonal(shared, 0.0)
    angles = 2 * np.pi * np.arange(K) / K
    best, best_cost = None, np.inf
    for rest in itertools.permutations(range(1, K)):
        order = np.array((0,) + rest)
        cost = float(np.sum(shared * np.cos(angles[order][:, None] - angles[order][None, :])))
        if cost < best_cost:
            best, best_cost = order, cost
    return best


def topic_colors(K, lightness, order):
    """Return the OKLab color of each topic."""
    angles = np.deg2rad(FIRST_HUE) + 2 * np.pi * order / K
    return np.stack([np.full(K, lightness), CHROMA * np.cos(angles), CHROMA * np.sin(angles)], axis=-1)


def oklab_to_srgb(lab):
    """Convert OKLab colors to sRGB (Ottosson, 2020)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    linear = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                       -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                       -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], axis=-1).clip(0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)


def label_position(xy, share, bins=120, sigma=3.0):
    """Return the place in the plot where the share of a topic is highest."""
    edges = [np.linspace(xy[:, i].min(), xy[:, i].max(), bins + 1) for i in (0, 1)]
    count = gaussian_filter(np.histogram2d(xy[:, 0], xy[:, 1], bins=edges)[0], sigma)
    total = gaussian_filter(np.histogram2d(xy[:, 0], xy[:, 1], bins=edges, weights=share)[0], sigma)
    score = np.where(count >= 0.25 * count.max(), total / np.maximum(count, 1e-12) * np.sqrt(count), 0.0)
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return 0.5 * (edges[0][i] + edges[0][i + 1]), 0.5 * (edges[1][j] + edges[1][j + 1])


def draw(xy, weights, labels, order, theme, page, lightness, path):
    """Draw the figure in one theme and save the figure as a PNG file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt
    lab = topic_colors(weights.shape[1], lightness, order)
    mix = weights ** EXPONENT
    mix /= mix.sum(axis=1, keepdims=True)
    colors, pure = oklab_to_srgb(mix @ lab), oklab_to_srgb(lab)
    on_top = np.argsort(mix.max(axis=1))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xy[on_top, 0], xy[on_top, 1], c=colors[on_top], s=7, linewidths=0, alpha=0.9)
    for k, label in labels.items():
        x, y = label_position(xy, mix[:, k])
        ax.text(x, y, label, color=pure[k], fontsize=13, fontweight="bold", ha="center", va="center",
                path_effects=[patheffects.withStroke(linewidth=3.5, foreground=page)])
    ax.set_axis_off()
    fig.savefig(path, dpi=160, bbox_inches="tight", transparent=True, metadata={"Software": None})
    plt.close(fig)
    print(f"The {theme} version is in {os.path.relpath(path, ROOT)}.")


def main():
    """Fit TEA on cds, place the users with UMAP, and draw both versions of the figure."""
    try:
        import umap
    except ImportError:
        sys.exit("This script needs the packages of requirements-figure.txt.")
    split = build.load(DATASET, os.path.join(ROOT, "data"))
    X = split["X_train"].astype(np.float64)
    fit = tea.fit(X, experiment.config("hyperparameters")["TEA"][DATASET], experiment.config("k")[DATASET])
    ranked = topic_table.rank(fit.beta, split["train_pos"].astype(np.float64))
    labels = {}
    for t, label in zip(ranked, experiment.config("topic_labels")[DATASET]):
        if [str(split["vocab"][i]) for i in t["items"][:5]] != label["items"]:
            sys.exit(f"The top items of topic {label['label']!r} differ from configs/topic_labels.json.")
        labels[t["topic"]] = label["label"]
    weights = topic_weights((X > 0).astype(np.float64), fit.beta)
    xy = umap.UMAP(n_neighbors=NEIGHBORS, min_dist=MIN_DIST, metric="cosine", random_state=SEED,
                   n_jobs=1).fit_transform(fit.scores.astype(np.float64))
    order = hue_order(weights ** EXPONENT / (weights ** EXPONENT).sum(axis=1, keepdims=True))
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    for theme, page, lightness in THEMES:
        draw(xy, weights, labels, order, theme, page, lightness, os.path.join(ROOT, "figures", f"user-map-{theme}.png"))


if __name__ == "__main__":
    main()
