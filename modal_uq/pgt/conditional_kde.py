
import numpy as np
from sklearn.neighbors import KernelDensity
from .base import PGTBase
from ..registry import register

@register('pgt','conditional_kde')
class ConditionalKDE(PGTBase):
    def __init__(self, bandwidth=0.5, kernel='gaussian', y_grid_points=512, y_pad=1.0):
        self.bandwidth = bandwidth; self.kernel = kernel
        self.y_grid_points = y_grid_points; self.y_pad = y_pad
        self._kde = None; self._y_bounds = None

    def fit(self, X, y):
        XY = np.hstack([X, y.reshape(-1,1)])
        self._kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(XY)
        self._y_bounds = (float(y.min()), float(y.max()))

    def _y_grid(self):
        lo, hi = self._y_bounds
        pad = self.y_pad * (hi - lo + 1e-6)
        return np.linspace(lo - pad, hi + pad, self.y_grid_points)

    def conditional_mode(self, x_query):
        y_grid = self._y_grid()
        XY = np.hstack([np.repeat(np.asarray(x_query).reshape(1,-1), len(y_grid), axis=0), y_grid.reshape(-1,1)])
        logp = self._kde.score_samples(XY)
        return float(y_grid[int(np.argmax(logp))])
