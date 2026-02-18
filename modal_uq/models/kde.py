
import numpy as np
from sklearn.neighbors import KernelDensity
from .base import ModelBase
from ..registry import register

@register('model','kde')
class ConditionalKDEModel(ModelBase):
    def __init__(self, bandwidth=0.5, kernel='gaussian', marginalization=None):
        super().__init__(marginalization=marginalization)
        self.bandwidth = bandwidth; self.kernel = kernel
        self._kde = None
        self._y_min = None; self._y_max = None

    def fit(self, X, y, X_val=None, y_val=None):
        XY = np.hstack([X, y.reshape(-1,1)])
        self._kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(XY)
        self._y_min = float(y.min()); self._y_max = float(y.max())

    def predict_density(self, X, y_grid, context='predict'):
        # Deterministic model: context parameter is ignored
        N, G = X.shape[0], len(y_grid)
        out = np.zeros((N, G))
        for i in range(N):
            XY = np.hstack([np.repeat(X[i:i+1,:], G, axis=0), y_grid.reshape(-1,1)])
            out[i] = np.exp(self._kde.score_samples(XY))
        return out
