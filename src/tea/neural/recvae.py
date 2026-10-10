"""RecVAE of Shenbin et al. (2020), following the RecBole implementation."""

import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tea.base import make_fit
from tea.neural.device import default_device

HIDDEN = 600
DROPOUT = 0.5
ENCODER_STEPS = 3
DECODER_STEPS = 1
EPOCHS = 50
LEARNING_RATE = 5e-4
MIXTURE = (0.15, 0.75, 0.1)


def _swish(x):
    return x * torch.sigmoid(x)


def _log_normal(x, mu, logvar):
    return -0.5 * (logvar + math.log(2 * math.pi) + (x - mu).pow(2) / logvar.exp())


class Encoder(nn.Module):
    """A five-layer encoder with dense skip connections, the swish activation, and layer normalization."""

    def __init__(self, n_items, latent):
        super().__init__()
        self.fc1 = nn.Linear(n_items, HIDDEN)
        self.ln1 = nn.LayerNorm(HIDDEN, eps=1e-1)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.ln2 = nn.LayerNorm(HIDDEN, eps=1e-1)
        self.fc3 = nn.Linear(HIDDEN, HIDDEN)
        self.ln3 = nn.LayerNorm(HIDDEN, eps=1e-1)
        self.fc4 = nn.Linear(HIDDEN, HIDDEN)
        self.ln4 = nn.LayerNorm(HIDDEN, eps=1e-1)
        self.fc5 = nn.Linear(HIDDEN, HIDDEN)
        self.ln5 = nn.LayerNorm(HIDDEN, eps=1e-1)
        self.fc_mu = nn.Linear(HIDDEN, latent)
        self.fc_logvar = nn.Linear(HIDDEN, latent)

    def forward(self, x, dropout):
        """Return the mean and the log-variance of the posterior of each user."""
        x = F.normalize(x)
        x = F.dropout(x, dropout, training=self.training)
        h1 = self.ln1(_swish(self.fc1(x)))
        h2 = self.ln2(_swish(self.fc2(h1) + h1))
        h3 = self.ln3(_swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(_swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(_swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class CompositePrior(nn.Module):
    """A mixture of three normal distributions with weights 0.15, 0.75, and 0.1.

    The three distributions are the standard normal distribution, the posterior of the encoder from the previous epoch,
    and a normal distribution with log-variance 10.
    """

    def __init__(self, n_items, latent):
        super().__init__()
        self.mu_prior = nn.Parameter(torch.zeros(1, latent), requires_grad=False)
        self.logvar_prior = nn.Parameter(torch.zeros(1, latent), requires_grad=False)
        self.logvar_uniform_prior = nn.Parameter(torch.full((1, latent), 10.0), requires_grad=False)
        self.encoder_old = Encoder(n_items, latent)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        """Return the log-density of z under the mixture prior."""
        post_mu, post_logvar = self.encoder_old(x, 0)
        parts = [_log_normal(z, self.mu_prior, self.logvar_prior), _log_normal(z, post_mu, post_logvar),
                 _log_normal(z, self.mu_prior, self.logvar_uniform_prior)]
        return torch.logsumexp(torch.stack([g + math.log(w) for g, w in zip(parts, MIXTURE)], dim=-1), dim=-1)


def _build(n_items, latent, seed):
    torch.manual_seed(int(seed))
    encoder = Encoder(n_items, latent)
    prior = CompositePrior(n_items, latent)
    decoder = nn.Linear(latent, n_items)
    for m in list(encoder.modules()) + list(decoder.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return encoder, prior, decoder


def _step(encoder, prior, decoder, X, gamma, optimizer, encode):
    if encode:
        encoder.train()
        decoder.eval()
        dropout = DROPOUT
    else:
        encoder.eval()
        decoder.train()
        dropout = 0.0
    optimizer.zero_grad()
    mu, logvar = encoder(X, dropout)
    if encoder.training:
        eps = torch.zeros_like(mu).normal_(mean=0.0, std=0.01)
        z = mu + eps * (0.5 * logvar).exp()
    else:
        z = mu
    log_pred = F.log_softmax(decoder(z), dim=-1)
    mll = (log_pred * X).sum(dim=-1).mean()
    kl_weight = gamma * X.sum(dim=-1)
    log_q = -0.5 * (logvar + math.log(2 * math.pi) + (z - mu).pow(2) / logvar.exp())
    kld = ((log_q - prior(X, z)).sum(dim=-1) * kl_weight).mean()
    loss = -(mll - kld)
    loss.backward()
    optimizer.step()


def fit(X, params, K=None, with_topics=False, seed=0):
    """Train RecVAE with alternating encoder and decoder updates and return the decoded scores."""
    device = default_device()
    X_t = torch.from_numpy(X.astype(np.float32)).float().to(device)
    encoder, prior, decoder = _build(X.shape[1], int(params["latent_dim"]), seed)
    encoder.to(device)
    prior.to(device)
    decoder.to(device)
    opt_encoder = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
    opt_decoder = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    gamma = float(params["gamma"])
    for _ in range(EPOCHS):
        for _ in range(ENCODER_STEPS):
            _step(encoder, prior, decoder, X_t, gamma, opt_encoder, True)
        for _ in range(DECODER_STEPS):
            _step(encoder, prior, decoder, X_t, gamma, opt_decoder, False)
        prior.encoder_old.load_state_dict(deepcopy(encoder.state_dict()))
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        mu, _ = encoder(X_t, 0.0)
        scores = decoder(mu).cpu().numpy().astype(np.float32)
    return make_fit(scores)
