"""Unit tests of the metrics, the split, the closed forms, ppSPA, the search, the selection of K, the significance
tests, the stored results, the grids, and the imports."""

import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

from tea import experiment, linear, metrics, ppspa, registry, search, select_k, stats, tea
from tea.data import split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ndcg_and_recall_on_a_small_example():
    """A hit at rank 1 out of two targets gives NDCG 1/(1 + 1/log2(3)) and recall 1/2."""
    scores = np.array([[3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    target = np.array([[True, False, False, True]])
    exclude = np.zeros_like(target)
    ndcg = metrics.ndcg_per_user(scores, target, exclude, 2)
    recall = metrics.recall_per_user(scores, target, exclude, 2)
    assert ndcg[0] == pytest.approx(1.0 / (1.0 + 1.0 / np.log2(3)))
    assert recall[0] == pytest.approx(0.5)


def test_excluded_items_are_never_recommended():
    """An excluded item with the highest score is skipped."""
    scores = np.array([[5.0, 1.0, 0.0]], dtype=np.float32)
    target = np.array([[False, True, False]])
    exclude = np.array([[True, False, False]])
    assert metrics.recall_per_user(scores, target, exclude, 1)[0] == 1.0


def test_effective_users():
    """One user gives 1, and three users with equal shares give 3."""
    one = np.array([[1, 1, 1], [0, 0, 1]], dtype=float)
    three = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=float)
    assert metrics.effective_users([0, 1, 2], one) == pytest.approx(1.0)
    assert metrics.effective_users([0, 1], three) == pytest.approx(3.0)


def test_diversity_counts_distinct_top_items():
    """Two topics with the same top item have a diversity of one half at n = 1."""
    beta = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0]])
    assert metrics.diversity(beta, 1) == pytest.approx(0.5)


def test_temporal_split():
    """The last timestamp is the test basket and the second-to-last is the validation basket."""
    rows = np.array([[0, 0, 1, 1.0], [0, 1, 2, 1.0], [0, 2, 3, 1.0], [0, 3, 4, 1.0]])
    parts = split.temporal_split(dict(interactions=rows, n_users=1, n_items=4))
    assert parts["train_pos"][0].tolist() == [True, True, False, False]
    assert parts["val_pos"][0].tolist() == [False, False, True, False]
    assert parts["test_pos"][0].tolist() == [False, False, False, True]


def test_user_folds_partition_the_users():
    """Every user is scored in exactly one fold."""
    folds = split.user_folds(10)
    assert np.array_equal(sum(score.astype(int) for _, score in folds), np.ones(10, dtype=int))
    build, score = split.holdout_users(9)
    assert score.sum() == 3 and not (build & score).any()


def test_ppspa_recovers_the_vertices_of_a_simplex():
    """Points on a simplex with five copies of each vertex have loadings equal to their barycentric weights."""
    rng = np.random.default_rng(0)
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    weights = np.vstack([np.repeat(np.eye(3), 5, axis=0), rng.dirichlet(np.ones(3), size=60)])
    points = weights @ vertices
    H = ppspa.loadings(np.column_stack([points, np.ones(len(points))]))
    assert min(np.abs(H[:, list(p)] - weights).max() for p in itertools.permutations(range(3))) < 1e-9


def toy_split(seed=0, users=60, items=20):
    """Return a random split with training, validation, and test items for every user."""
    rng = np.random.default_rng(seed)
    order = rng.random((users, items)).argsort(axis=1)
    train = np.zeros((users, items), dtype=bool)
    val = np.zeros_like(train)
    test = np.zeros_like(train)
    rows = np.arange(users)[:, None]
    train[rows, order[:, :6]] = True
    val[rows, order[:, 6:7]] = True
    test[rows, order[:, 7:8]] = True
    return dict(X_train=train.astype(np.float32), train_pos=train, val_pos=val, test_pos=test)


def test_ease_satisfies_its_optimality_condition():
    """(X^T X + lambda I)(B - I) is a diagonal matrix, and B has a zero diagonal."""
    X = toy_split()["X_train"].astype(np.float64)
    B, _ = linear.ease(X, 10.0)
    residual = (X.T @ X + 10.0 * np.eye(X.shape[1])) @ (B - np.eye(X.shape[1]))
    assert np.allclose(np.diag(B), 0.0)
    assert np.abs(residual - np.diag(np.diag(residual))).max() < 1e-4


def test_tea_returns_scores_and_topics_on_the_simplex():
    """TEA returns a score for every user-item pair and K topics that are distributions over the items."""
    X = toy_split()["X_train"].astype(np.float64)
    fit = tea.fit(X, {"alpha": 0.125, "p": 0.5, "lambda": 10.0}, K=3)
    assert fit.scores.shape == X.shape and fit.beta.shape == (3, X.shape[1])
    assert np.allclose(fit.beta.sum(axis=1), 1.0) and (fit.beta >= 0).all()


def test_search_selects_the_best_candidate_and_the_first_of_a_tie():
    """The search returns the candidate with the highest validation NDCG@10, and the first one of a tie."""
    data = toy_split()
    candidates = [{"lambda": 1.0}, {"lambda": 30.0}, {"lambda": 1000.0}]
    result = search.search("EASE", data, 3, candidates)
    values = [c["ndcg"] for c in result["curve"]]
    assert values == [search.validation_ndcg("EASE", data, c, 3)[0] for c in candidates]
    assert result["selected"] is candidates[int(np.argmax(values))]
    tied = [{"lambda": 10.0}, {"lambda": 10.0}]
    assert search.search("EASE", data, 3, tied)["selected"] is tied[0]


def test_selection_of_k_takes_the_smallest_mean_log_perplexity():
    """The selected K minimizes the mean of the log-perplexities over the build sets."""
    curves = [{5: 1.0, 6: 4.0}, {5: 100.0, 6: 4.0}]
    best, mean = select_k.select(curves)
    assert best == 6 and mean[5] == pytest.approx(10.0)


def test_coherence_of_items_that_always_appear_together_is_one():
    """Two items that the same half of the users have give NPMI 1."""
    reference = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    beta = np.array([[0.5, 0.4, 0.1]])
    assert metrics.coherence(beta, reference, 2) == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_keeps_a_method_with_the_same_topics_in_bold():
    """Two methods with identical topics share every bootstrap replicate, so neither is significantly lower."""
    rng = np.random.default_rng(0)
    reference = (rng.random((40, 12)) < 0.4).astype(float)
    top = np.array([[0, 1, 2], [3, 4, 5]])
    tops = {"A": top, "B": top.copy(), "C": np.array([[6, 7, 8], [9, 10, 11]])}
    boot = stats.bootstrap_tq(tops, {m: 1.0 for m in tops}, reference, np.arange(40), samples=50)
    assert np.array_equal(boot["A"], boot["B"])
    tq = {m: stats.coherence_fast(t, reference) for m, t in tops.items()}
    bold = stats.not_lower_by_bootstrap(tq, boot, ("A", "B", "C"))
    assert ("A" in bold) == ("B" in bold)


def test_two_sided_p():
    """A difference that is always positive has p-value 0, and a symmetric one has p-value 1."""
    assert stats.two_sided_p(np.ones(10)) == 0.0
    assert stats.two_sided_p(np.array([-1.0, 1.0])) == 1.0


def test_t_test_keeps_ties_and_drops_lower_methods():
    """A method equal to the best is bold, and a method lower for every user is not."""
    best = np.linspace(0.0, 1.0, 50)
    per_user = {"A": best, "B": best.copy(), "C": 0.5 * best}
    assert stats.not_lower_by_t_test(per_user, ("A", "B", "C")) == ["A", "B"]


def test_a_stored_result_loads_with_its_per_user_arrays(tmp_path):
    """A result of EASE saved and loaded again keeps its per-user arrays, its topics, its TQ, and its means."""
    rng = np.random.default_rng(0)
    X = (rng.random((40, 12)) < 0.3).astype(np.float32)
    held = rng.random(X.shape) < 0.1
    data = dict(X_train=X, train_pos=X > 0, val_pos=held & (X == 0), test_pos=~held & (X == 0) & (rng.random(X.shape) < 0.1))
    out = experiment.run("EASE", "toy", data, {"lambda": 10.0}, 2)
    experiment.save(str(tmp_path), "toy", "EASE", out, {"lambda": 10.0}, 2)
    back = experiment.load(str(tmp_path), "toy", "EASE")
    for key in ("ndcg", "recall", "beta"):
        assert isinstance(back[key], np.ndarray) and np.array_equal(back[key], out[key])
    assert back["tq"] == out["tq"] and back["finite"] is True
    assert back["ndcg_mean"] == float(np.mean(out["ndcg"])) and back["params"] == {"lambda": 10.0} and back["K"] == 2


def test_grid_sizes():
    """The grids have the sizes of the search of the paper."""
    sizes = {m.name: len(search.grid(m.name)) for m in registry.METHODS}
    assert sizes == {"WMF": 4, "LightGCN": 4, "SLIMElastic": 9, "ADMMSLIM": 9, "NCEPLRec": 6, "RecVAE": 4, "LDA": 18,
                     "ETM": 36, "ECRTM": 36, "Mult-VAE": 18, "NMF": 14, "NMF-ppSPA": 14, "EASE": 5, "RW": 5,
                     "Symmetric": 5, "DAN": 100, "RDLAE": 75, "TEA": 180}


def test_non_neural_modules_do_not_import_torch():
    """The registry and the closed-form and classical methods load without torch."""
    code = ("import sys, tea.registry, tea.linear, tea.tea, tea.classical.nmf, tea.classical.lda, "
            "tea.experiment, tea.stats; sys.exit('torch' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_scripts_do_not_shadow_the_standard_library():
    """No script has the name of a standard-library module, which would hide that module from the other scripts."""
    names = {f[:-3] for f in os.listdir(os.path.join(ROOT, "scripts")) if f.endswith(".py")}
    assert not names & set(sys.stdlib_module_names)
