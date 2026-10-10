"""Three neural topic models with a shared encoder: ETM, ECRTM, and Mult-VAE.

ETM of Dieng et al. (2020) has topics softmax(rho alpha^T) over the items.
ECRTM of Wu et al. (2023) has topics from the distances between topic and item embeddings, a
batch-normalized decoder, and an embedding-clustering regularizer.
Mult-VAE of Liang et al. (2018) has a linear decoder from the latent vector to the items.
ETM follows the code of the authors at https://github.com/adjidieng/ETM.
ECRTM follows the implementation of the authors at https://github.com/BobXWu/ECRTM.
All three are trained with full-batch Adam and keep the epoch with the highest validation NDCG@10.
"""

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tea import metrics
from tea.base import make_fit
from tea.neural.device import default_device

EMBEDDING_SIZE = 300
WEIGHT_DECAY = 1.2e-6
EPOCHS = 200
PATIENCE = 30
PRIOR_ALPHA = 1.0
DEFAULT_TEMPERATURE = 0.2


class TopicModel(nn.Module):
    """The encoder and decoder of ETM, ECRTM, or Mult-VAE, selected by arch."""

    def __init__(self, arch, n_topics, n_items, hidden, dropout, temperature=DEFAULT_TEMPERATURE, ecr_weight=0.0):
        super().__init__()
        self.arch = arch
        self.temperature = temperature
        self.ecr_weight = ecr_weight
        self.n_topics = n_topics
        self.n_items = n_items
        self.dropout = dropout
        self.t_drop = nn.Dropout(dropout)
        self.theta_act = nn.ReLU()
        self.rho = nn.Linear(EMBEDDING_SIZE, n_items, bias=False)
        self.alphas = nn.Linear(EMBEDDING_SIZE, n_topics, bias=False)
        self.q_theta = nn.Sequential(nn.Linear(n_items, hidden), self.theta_act,
                                     nn.Linear(hidden, hidden), self.theta_act)
        self.mu_q_theta = nn.Linear(hidden, n_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(hidden, n_topics, bias=True)
        if arch == "ecrtm":
            topic_emb = torch.empty(n_topics, EMBEDDING_SIZE)
            nn.init.trunc_normal_(topic_emb, std=0.1)
            self.topic_embeddings = nn.Parameter(F.normalize(topic_emb, dim=1))
            self.decoder_bn = nn.BatchNorm1d(n_items, affine=True)
            self.decoder_bn.weight.requires_grad = False
            pa = PRIOR_ALPHA * torch.ones(n_topics)
            self.register_buffer("ecr_mu0", pa.log() - pa.log().mean())
            self.register_buffer("ecr_var0", (1.0 / pa) * (1 - 2.0 / n_topics) + (1.0 / pa).sum() / n_topics ** 2)
        elif arch == "multvae":
            self.mv_decoder = nn.Linear(n_topics, n_items, bias=True)
            nn.init.xavier_uniform_(self.mv_decoder.weight)
            nn.init.zeros_(self.mv_decoder.bias)

    def encode(self, bows):
        """Return the posterior mean, the posterior log-variance, and the KL divergence to the prior."""
        q = self.q_theta(bows)
        if self.dropout > 0:
            q = self.t_drop(q)
        mu = self.mu_q_theta(q)
        logsigma = self.logsigma_q_theta(q)
        if self.arch == "ecrtm":
            var = logsigma.exp()
            kl = (var / self.ecr_var0 + (mu - self.ecr_mu0).pow(2) / self.ecr_var0
                  + self.ecr_var0.log() - logsigma)
            return mu, logsigma, (0.5 * (kl.sum(-1) - self.n_topics)).mean()
        return mu, logsigma, -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=-1).mean()

    def theta(self, normalized_bows):
        """Return the topic proportions of each user and the KL divergence to the prior."""
        mu, logsigma, kl = self.encode(normalized_bows)
        if self.training:
            std = torch.exp(0.5 * logsigma)
            z = torch.randn_like(std).mul_(std).add_(mu)
        else:
            z = mu
        return (z if self.arch == "multvae" else F.softmax(z, dim=-1)), kl

    def _distances(self):
        return (self.topic_embeddings.unsqueeze(1) - self.rho.weight.unsqueeze(0)).pow(2).sum(-1)

    def beta(self):
        """Return the K x m topic matrix."""
        if self.arch == "ecrtm":
            return F.softmax(-self._distances() / self.temperature, dim=0)
        if self.arch == "multvae":
            return F.softmax(self.mv_decoder.weight.t(), dim=1)
        return F.softmax(self.alphas(self.rho.weight), dim=0).transpose(1, 0)

    def decode(self, theta, beta):
        """Return the logits of the items for each user."""
        if self.arch == "ecrtm":
            return self.decoder_bn(torch.mm(theta, beta))
        if self.arch == "multvae":
            return self.mv_decoder(theta)
        res = torch.mm(theta, beta)
        return torch.log(res.add(torch.full_like(res, 1e-6)))

    def ecr_loss(self):
        """Return the Sinkhorn transport cost between the topic and the item embeddings.

        The Sinkhorn iterations follow the implementation of the authors of ECRTM at https://github.com/BobXWu/ECRTM.
        """
        cost = self._distances()
        a = torch.ones(self.n_topics, device=cost.device) / self.n_topics
        b = torch.ones(self.n_items, device=cost.device) / self.n_items
        kernel = torch.exp(-cost * 20.0)
        u = torch.ones_like(a) / self.n_topics
        v = None
        for it in range(1000):
            v = b / (kernel.t() @ u + 1e-16)
            u = a / (kernel @ v + 1e-16)
            if it % 50 == 1:
                if (v * (kernel.t() @ u) - b).abs().sum().item() < 0.5e-2:
                    break
        return (u.unsqueeze(-1) * (kernel * v) * cost).sum()

    def forward(self, bows, normalized_bows, mask):
        """Return the masked multinomial reconstruction loss and the KL divergence."""
        theta, kl = self.theta(normalized_bows)
        preds = self.decode(theta, self.beta())
        loss = (-(mask * bows * F.log_softmax(preds, dim=1)).sum(1)).mean()
        if self.arch == "ecrtm" and self.ecr_weight > 0:
            loss = loss + self.ecr_weight * self.ecr_loss()
        return loss, kl

    def scores(self, normalized_bows):
        """Return the item logits, the topic proportions, and the topics, in evaluation mode."""
        self.eval()
        theta, _ = self.theta(normalized_bows)
        beta = self.beta()
        return self.decode(theta, beta), theta, beta


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(arch, X, train_pos, val_pos, test_pos, K, lr, kl_weight, hidden=800, dropout=0.0,
          temperature=DEFAULT_TEMPERATURE, ecr_weight=0.0, seed=0, epochs=EPOCHS):
    """Train a topic model and return its scores and topics at the epoch with the highest validation NDCG@10.

    The validation ranking excludes the training items. The loss ignores the validation and test items.
    """
    _seed(seed)
    device = default_device()
    X = np.asarray(X, dtype=np.float32)
    bows = torch.from_numpy(X).float().to(device)
    norm = torch.from_numpy(X / np.clip(X.sum(axis=1, keepdims=True), 1e-6, None)).float().to(device)
    mask = torch.from_numpy((~(val_pos | test_pos)).astype(np.float32)).float().to(device)
    model = TopicModel(arch, K, X.shape[1], hidden, dropout, temperature, ecr_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    best_val, best_state, bad = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, kl = model(bows, norm, mask)
        loss = recon + kl_weight * kl
        loss.backward()
        optimizer.step()
        logits, _, _ = model.scores(norm)
        v = float(np.mean(metrics.ndcg_per_user(logits.detach().cpu().numpy(), val_pos, train_pos, 10)))
        if v > best_val + 1e-6:
            best_val, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    logits, _, beta = model.scores(norm)
    return logits.detach().cpu().numpy(), beta.detach().cpu().numpy()


def _fit(arch, X, K, split, seed, **kwargs):
    scores, beta = train(arch, X, split["train_pos"], split["val_pos"], split["test_pos"], K, seed=seed, **kwargs)
    return make_fit(scores, beta)


def fit_etm(X, params, K, with_topics=True, seed=0, split=None):
    """Train ETM with the given learning rate, hidden size, dropout probability, and KL weight."""
    return _fit("etm", X, K, split, seed, lr=params["learning_rate"], kl_weight=params["kl_weight"],
                hidden=int(params["hidden_size"]), dropout=params["dropout"])


def fit_ecrtm(X, params, K, with_topics=True, seed=0, split=None):
    """Train ECRTM with the given learning rate, temperature, ECR weight, and KL weight."""
    return _fit("ecrtm", X, K, split, seed, lr=params["learning_rate"], kl_weight=params["kl_weight"],
                temperature=params["temperature"], ecr_weight=params["ecr_weight"])


def fit_multvae(X, params, K, with_topics=True, seed=0, split=None):
    """Train Mult-VAE with the given learning rate, KL weight, and dropout probability."""
    return _fit("multvae", X, K, split, seed, lr=params["learning_rate"], kl_weight=params["kl_weight"],
                dropout=params["dropout"])
