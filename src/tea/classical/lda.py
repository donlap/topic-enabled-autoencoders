"""Latent Dirichlet allocation of Blei et al. (2003), fitted by scikit-learn."""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from tea.base import make_fit


def fit(X, params, K, with_topics=True, seed=0):
    """Fit LDA with K topics and score each item for a user by log(theta beta).

    An unset prior keeps the scikit-learn default of 1/K.
    """
    lda = LatentDirichletAllocation(n_components=int(K), max_iter=int(params["max_iter"]),
                                    doc_topic_prior=params["doc_topic_prior"],
                                    topic_word_prior=params["topic_word_prior"],
                                    learning_method="batch", random_state=seed)
    theta = lda.fit_transform(X)
    theta = (theta / theta.sum(1, keepdims=True).clip(min=1e-12)).astype(np.float32)
    beta = lda.components_
    beta = (beta / beta.sum(1, keepdims=True).clip(min=1e-12)).astype(np.float32)
    return make_fit(np.log(theta @ beta + 1e-12), beta)
