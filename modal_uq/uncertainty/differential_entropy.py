
import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','differential_entropy')
class DifferentialEntropy(UncertaintyBase):
    def __init__(self, grid_points=512, y_pad=1.0, **kwargs):
        super().__init__(grid_points=grid_points, y_pad=y_pad, **kwargs)
    def score(self, model, X, y_true=None):
        y_grid = model.default_y_grid(X, grid_points=self.kwargs['grid_points'], y_pad=self.kwargs['y_pad'])
        p = model.predict_density(X, y_grid)
        p = p / (np.trapz(p, y_grid, axis=1, keepdims=True) + 1e-12)
        H = -np.trapz(p * (np.log(p + 1e-12)), y_grid, axis=1)
        return H
