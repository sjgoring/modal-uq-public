
import numpy as np
from .base import UncertaintyBase
from ..registry import register
import scipy.integrate as integrate

@register('uncertainty','nll')
class NLLScore(UncertaintyBase):
    def score(self, model, X, y_true=None):
        if y_true is None:
            raise ValueError('NLL requires y_true')
        y_grid = model.default_y_grid(X)
        dens = model.predict_density(X, y_grid)
        idx = np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)
        p = dens[np.arange(len(y_true)), idx] + 1e-12
        return -np.log(p)

@register('uncertainty','crps_proxy')
class CRPSProxy(UncertaintyBase):
    def score(self, model, X, y_true=None):
        y_grid = model.default_y_grid(X)
        dens = model.predict_density(X, y_grid)
        cdf = np.cumsum(dens, axis=1); cdf /= (cdf[:,-1][:,None] + 1e-12)
        return integrate.trapezoid(np.abs(cdf - 0.5), y_grid, axis=1)
