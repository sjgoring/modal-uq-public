
import numpy as np
from .base import ModelBase
from ..registry import register

@register('model','mdn')
class MDN(ModelBase):
    def __init__(self, input_dim=None, hidden_sizes=(64,64), num_components=5, lr=1e-3, epochs=200, batch_size=128, marginalization=None):
        super().__init__(marginalization=marginalization)
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.num_components = num_components
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self._torch = None; self._net = None
        self._y_min = None; self._y_max = None

    def _ensure_torch(self):
        if self._torch is None:
            import torch, torch.nn as nn
            self._torch = (torch, nn)
        return self._torch

    def fit(self, X, y, X_val=None, y_val=None):
        torch, nn = self._ensure_torch()
        input_dim = X.shape[1] if self.input_dim is None else self.input_dim
        layers = []
        d = input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        head = nn.Linear(d, 3*self.num_components)
        net = nn.Sequential(*layers, head)
        self._net = net
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1,1)
        def split(out):
            K = self.num_components
            pi = torch.softmax(out[:, :K], dim=-1)
            mu = out[:, K:2*K]
            sigma = torch.exp(out[:, 2*K:3*K]).clamp_min(1e-6)
            return pi, mu, sigma
        for _ in range(self.epochs):
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i:i+self.batch_size]; yb = y_t[i:i+self.batch_size]
                out = net(xb)
                pi, mu, sigma = split(out)
                comp = torch.distributions.Normal(mu, sigma)  # mu,sigma: [B,K]
                yb_expanded = yb.expand_as(mu)                # [B,K]
                log_comp = comp.log_prob(yb_expanded)         # [B,K]
                log_prob = torch.logsumexp(torch.log(pi + 1e-12) + log_comp, dim=1)  # [B]
                loss = -log_prob.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self._y_min = float(y.min()); self._y_max = float(y.max())

    def predict_density(self, X, y_grid, context='predict'):
        # Deterministic model: context parameter is ignored
        torch, nn = self._ensure_torch()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            out = self._net(X_t)
            K = self.num_components
            pi = torch.softmax(out[:, :K], dim=-1)
            mu = out[:, K:2*K]
            sigma = torch.exp(out[:, 2*K:3*K]).clamp_min(1e-6)
            Yg = torch.tensor(y_grid, dtype=torch.float32)[None, None, :]
            mu = mu[:, :, None]; sigma = sigma[:, :, None]; pi = pi[:, :, None]
            comp = torch.distributions.Normal(mu, sigma)
            dens = torch.sum(pi * torch.exp(comp.log_prob(Yg)), dim=1)
            return dens.numpy()

    def predict_mixture_params(self, X):
        torch, nn = self._ensure_torch()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            out = self._net(X_t)
            K = self.num_components
            pi = torch.softmax(out[:, :K], dim=-1).numpy()
            mu = out[:, K:2*K].numpy()
            sigma2 = torch.exp(2*out[:, 2*K:3*K]).clamp_min(1e-12).numpy()
        return pi, mu, sigma2
    
    def get_member_parameters(self):
        """Return member indices as 'parameter samples' for meta-QUEST."""
        return np.arange(len(self.members)).reshape(-1, 1)
