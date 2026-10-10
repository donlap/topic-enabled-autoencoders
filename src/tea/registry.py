"""The 18 methods of the paper, in the order of Table 1.

Each method has a fit function with the signature fit(X, params, K, with_topics, seed) that returns a Fit.
The neural methods are imported only when they are fitted, so that this module does not import torch.
"""

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    """A method of the paper.

    A method has a name, a kind, the location of its fit function, and a flag that tells whether the method has topics.
    """

    name: str
    kind: str
    target: str
    topics: bool

    def fit(self, X, params, K, with_topics=True, seed=0, **extra):
        """Fit the method on the training matrix X with the given hyperparameters and number of topics K."""
        module, function = self.target.split(":")
        fn = getattr(importlib.import_module(module), function)
        return fn(X, params, K, with_topics=with_topics and self.topics, seed=seed, **extra)


METHODS = (
    Method("WMF", "classical", "tea.classical.wmf:fit", False),
    Method("LightGCN", "neural", "tea.neural.lightgcn:fit", False),
    Method("SLIMElastic", "classical", "tea.classical.slim:fit_elastic", False),
    Method("ADMMSLIM", "classical", "tea.classical.slim:fit_admm", False),
    Method("NCEPLRec", "classical", "tea.classical.nceplrec:fit", False),
    Method("RecVAE", "neural", "tea.neural.recvae:fit", False),
    Method("LDA", "classical", "tea.classical.lda:fit", True),
    Method("ETM", "neural", "tea.neural.topic_models:fit_etm", True),
    Method("ECRTM", "neural", "tea.neural.topic_models:fit_ecrtm", True),
    Method("Mult-VAE", "neural", "tea.neural.topic_models:fit_multvae", True),
    Method("NMF", "classical", "tea.classical.nmf:fit", True),
    Method("NMF-ppSPA", "classical", "tea.classical.nmf:fit_ppspa", True),
    Method("EASE", "linear", "tea.linear:fit_ease", True),
    Method("RW", "linear", "tea.linear:fit_rw", True),
    Method("Symmetric", "linear", "tea.linear:fit_symmetric", True),
    Method("DAN", "linear", "tea.linear:fit_dan", True),
    Method("RDLAE", "linear", "tea.linear:fit_rdlae", True),
    Method("TEA", "linear", "tea.tea:fit", True),
)

BY_NAME = {m.name: m for m in METHODS}


def get(name):
    """Return the method with the given name."""
    return BY_NAME[name]
