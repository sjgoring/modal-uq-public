
import numpy as np
from .base import ModelBase
from ..registry import register

@register('model','mdn_mc_dropout')
class MDN_MCDropout(ModelBase):
    def __init__(self, input_dim=None, hidden_sizes=(64,64), dropout_p=0.1,
                 num_components=5, lr=1e-3, epochs=200, batch_size=128):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout_p = dropout_p
        self.num_components = num_components
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self._torch = None; self._net = None
        self._y_min = None; self._y_max = None

    def _ensure_torch(self):
        if self._torch is None:
            import torch, torch.nn as nn
            self._torch = (torch, nn)
        return self._torch

    def _build_net(self, input_dim):
        torch, nn = self._ensure_torch()
        layers = []
        d = input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh(), nn.Dropout(p=self.dropout_p)]
            d = h
        head = nn.Linear(d, 3*self.num_components)
        net = nn.Sequential(*layers, head)
        return net

    def fit(self, X, y, X_val=None, y_val=None):
        torch, nn = self._ensure_torch()
        input_dim = X.shape[1] if self.input_dim is None else self.input_dim
        self._net = self._build_net(input_dim)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
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
                out = self._net(xb)
                pi, mu, sigma = split(out)
                comp = torch.distributions.Normal(mu, sigma)
                log_prob = torch.logsumexp(torch.log(pi + 1e-12) + comp.log_prob(yb).squeeze(2), dim=1)
                loss = -log_prob.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self._y_min = float(y.min()); self._y_max = float(y.max())

    def _forward_density(self, X, y_grid):
        torch, nn = self._ensure_torch()
        with torch.no_grad():
            self._net.train()  # activate dropout
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

    def predict_density(self, X, y_grid):
        torch, nn = self._ensure_torch()
        with torch.no_grad():
            self._net.eval()
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

    def predict_density_samples(self, X, y_grid, n_samples: int = 20):
        import numpy as np
        samples = []
        for _ in range(n_samples):
            samples.append(self._forward_density(X, y_grid))
        return np.stack(samples, axis=0)
