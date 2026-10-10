"""LightGCN of He et al. (2020), trained with the BPR loss."""

import numpy as np
import torch

from tea.base import make_fit
from tea.neural.device import default_device

DIMENSION = 64
EPOCHS = 30
BATCH_SIZE = 4096
WEIGHT_DECAY = 1e-4


def fit(X, params, K=None, with_topics=False, seed=0):
    """Train LightGCN with embedding dimension 64 and return the scores of the propagated embeddings."""
    device = default_device()
    rng = np.random.RandomState(seed)
    n_users, n_items = X.shape
    positives = [np.nonzero(X[u])[0] for u in range(n_users)]
    row, col = [], []
    for u, items in enumerate(positives):
        row.extend([u] * len(items))
        col.extend(items.tolist())
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    n_pos = row.shape[0]
    src = np.concatenate([row, col + n_users])
    dst = np.concatenate([col + n_users, row])
    deg = np.bincount(src, minlength=n_users + n_items).astype(np.float64)
    d_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1.0, None))
    vals = d_inv_sqrt[src] * d_inv_sqrt[dst]
    indices = torch.from_numpy(np.stack([src, dst])).long().to(device)
    values = torch.from_numpy(vals).float().to(device)
    A = torch.sparse_coo_tensor(indices, values, (n_users + n_items, n_users + n_items)).coalesce()
    gen = torch.Generator().manual_seed(seed)
    E0 = torch.nn.Parameter((0.1 * torch.randn(n_users + n_items, DIMENSION, generator=gen)).to(device))
    opt = torch.optim.Adam([E0], lr=float(params["learning_rate"]), weight_decay=WEIGHT_DECAY)
    positive_sets = [set(items.tolist()) for items in positives]
    layers = int(params["layers"])

    def _propagate():
        emb = [E0]
        for _ in range(layers):
            emb.append(torch.sparse.mm(A, emb[-1]))
        return torch.stack(emb, 0).mean(0)

    steps = max(1, n_pos // BATCH_SIZE)
    for _ in range(EPOCHS):
        order = rng.permutation(n_pos)
        for s in range(steps):
            sel = order[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            u, i = row[sel], col[sel]
            j = rng.randint(0, n_items, size=u.shape[0])
            for k in range(u.shape[0]):
                while j[k] in positive_sets[u[k]]:
                    j[k] = rng.randint(0, n_items)
            full = _propagate()
            U, V = full[:n_users], full[n_users:]
            ut = torch.from_numpy(u).long().to(device)
            it = torch.from_numpy(i).long().to(device)
            jt = torch.from_numpy(j).long().to(device)
            xui = (U[ut] * V[it]).sum(-1)
            xuj = (U[ut] * V[jt]).sum(-1)
            loss = -torch.nn.functional.logsigmoid(xui - xuj).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        full = _propagate()
        scores = (full[:n_users] @ full[n_users:].T).detach().cpu().numpy()
    return make_fit(scores.astype(np.float32))
