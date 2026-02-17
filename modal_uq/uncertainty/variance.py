
import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','variance')
class PredictiveVariance(UncertaintyBase):
    def score(self, model, X, y_true=None):
        if hasattr(model, 'predict_mixture_params'):
            pi, mu, sigma2 = model.predict_mixture_params(X)
            m1 = (pi * mu).sum(axis=1)
            m2 = (pi * (sigma2 + mu**2)).sum(axis=1)
            return (m2 - m1**2)
        else:
            y_grid = model.default_y_grid(X)
            dens = model.predict_density(X, y_grid)
            dens /= (np.trapz(dens, y_grid, axis=1, keepdims=True) + 1e-12)
            Ey  = np.trapz(dens * y_grid[None,:], y_grid, axis=1)
            Ey2 = np.trapz(dens * (y_grid[None,:]**2), y_grid, axis=1)
            return (Ey2 - Ey**2)
