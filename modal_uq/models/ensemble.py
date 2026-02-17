
import numpy as np
from .base import ModelBase
from ..registry import register, build

@register('model','ensemble')
class Ensemble(ModelBase):
    def __init__(self, base_model='mdn', base_params=None, n_members=5, bootstrap=True, seed=42):
        self.base_model = base_model
        self.base_params = base_params or {}
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.seed = seed
        self.members = []
        self._y_min = None; self._y_max = None

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        self.members = []
        N = len(X)
        from copy import deepcopy
        for _ in range(self.n_members):
            model = build('model', self.base_model, **deepcopy(self.base_params))
            if self.bootstrap:
                idx = rng.integers(0, N, size=N)
                X_m, y_m = X[idx], y[idx]
            else:
                X_m, y_m = X, y
            model.fit(X_m, y_m, X_val, y_val)
            self.members.append(model)
        self._y_min = float(y.min()); self._y_max = float(y.max())

    def predict_density(self, X, y_grid):
        dens = [m.predict_density(X, y_grid) for m in self.members]
        return np.mean(np.stack(dens, axis=0), axis=0)

    def predict_density_samples(self, X, y_grid, n_samples: int = None):
        dens = [m.predict_density(X, y_grid) for m in self.members]
        return np.stack(dens, axis=0)
