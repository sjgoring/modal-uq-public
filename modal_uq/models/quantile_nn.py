
import numpy as np
from .base import ModelBase
from ..registry import register

@register('model','quantile_nn')
class QuantileNN(ModelBase):
    def __init__(self, quantiles=(0.1,0.5,0.9), hidden_sizes=(128,128), lr=1e-3, epochs=200, batch_size=128):
        self.quantiles = list(quantiles)
        self.hidden_sizes = hidden_sizes
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
        d = X.shape[1]
        layers = []
        for h in self.hidden_sizes:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        head = nn.Linear(d, len(self.quantiles))
        net = nn.Sequential(*layers, head)
        self._net = net
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1,1)
        q = torch.tensor(self.quantiles, dtype=torch.float32)
        for _ in range(self.epochs):
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i:i+self.batch_size]; yb = y_t[i:i+self.batch_size]
                pred = net(xb)
                e = yb - pred
                loss = (q[None,:]*torch.clamp(e, min=0) + (1-q[None,:])*torch.clamp(-e, min=0)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self._y_min = float(y.min()); self._y_max = float(y.max())

    def predict_density(self, X, y_grid):
        torch, nn = self._ensure_torch()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            qy = self._net(X_t).numpy()  # [N,Q]
        N, G = X.shape[0], len(y_grid)
        pdf = np.zeros((N, G))
        for i in range(N):
            xs = list(qy[i]); ps = self.quantiles
            xs, ps = zip(*sorted(zip(xs, ps)))
            xs = np.array(xs); ps = np.array(ps)
            xg = y_grid
            cdf = np.interp(xg, xs, ps, left=0.0, right=1.0)
            pdf[i] = np.gradient(cdf, xg)
            pdf[i] = np.maximum(pdf[i], 0)
        return pdf
