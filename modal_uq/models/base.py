
from abc import ABC, abstractmethod
import numpy as np

class ModelBase(ABC):
    """Base API for predictive density models.

    Optional stochastic extensions:
      - predict_density_samples: return multiple parameter-sampled densities [S,N,G]
      - predict_mixture_params: return deterministic mixture params (for MDN)
    """
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None): ...

    @abstractmethod
    def predict_density(self, X, y_grid): ...

    def predict_mode(self, X, y_grid):
        dens = self.predict_density(X, y_grid)
        idx = dens.argmax(axis=1)
        return y_grid[idx]

    def default_y_grid(self, X, grid_points=512, y_pad=1.0):
        lo, hi = -1.0, 1.0
        try:
            lo, hi = float(self._y_min), float(self._y_max)
        except Exception:
            pass
        pad = y_pad * (hi - lo + 1e-6)
        return np.linspace(lo - pad, hi + pad, grid_points)

    def predict_density_samples(self, X, y_grid, n_samples: int = 20):
        dens = self.predict_density(X, y_grid)
        return np.repeat(dens[None, ...], n_samples, axis=0)

    def predict_moments(self, X, y_grid):
        dens = self.predict_density(X, y_grid)
        dens = dens / (np.trapz(dens, y_grid, axis=1, keepdims=True) + 1e-12)
        Ey  = np.trapz(dens * y_grid[None,:], y_grid, axis=1)
        Ey2 = np.trapz(dens * (y_grid[None,:]**2), y_grid, axis=1)
        Var = Ey2 - Ey**2
        return Ey, Var
