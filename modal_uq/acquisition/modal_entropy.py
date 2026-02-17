
import numpy as np
from .base import AcquisitionBase
from ..registry import register

@register('acquisition','modal_entropy')
class ModalEntropy(AcquisitionBase):
    def __init__(self, grid_points=512, y_pad=1.0):
        self.grid_points = grid_points; self.y_pad = y_pad
    def score(self, model, X_pool):
        y_grid = model.default_y_grid(X_pool, grid_points=self.grid_points, y_pad=self.y_pad)
        dens = model.predict_density(X_pool, y_grid)
        dens /= (np.trapz(dens, y_grid, axis=1, keepdims=True) + 1e-12)
        H = -np.trapz(dens * (np.log(dens + 1e-12)), y_grid, axis=1)
        return H
