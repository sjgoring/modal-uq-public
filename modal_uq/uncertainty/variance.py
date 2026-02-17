
import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','variance')
class PredictiveVariance(UncertaintyBase):
    """Predictive variance with optional decomposition.

    - total:      Var[Y|x]
    - aleatoric:  E_θ[ Var(Y|x,θ) ]
    - epistemic:  Var_θ[ E(Y|x,θ) ]

    If the model exposes stochastic parameter sampling via
    `predict_density_samples(X, y_grid, n_samples) -> [S,N,G]`, we use Monte Carlo
    to decompose variance. Otherwise we fall back to a deterministic density
    (aleatoric only; epistemic=0).
    """
    def __init__(self, decomposition='total', grid_points=512, y_pad=1.0, n_param_samples=20):
        assert decomposition in {'total','aleatoric','epistemic'}
        self.decomposition = decomposition
        self.grid_points = grid_points
        self.y_pad = y_pad
        self.n_param_samples = n_param_samples

    def _moments_from_density(self, dens, y_grid):
        dens = dens / (np.trapz(dens, y_grid, axis=-1, keepdims=True) + 1e-12)
        Ey  = np.trapz(dens * y_grid[None,None,:] if dens.ndim==3 else dens * y_grid[None,:], y_grid, axis=-1)
        Ey2 = np.trapz(dens * (y_grid[None,None,:] if dens.ndim==3 else y_grid[None,:])**2, y_grid, axis=-1)
        Var = Ey2 - Ey**2
        return Ey, Var

    def score(self, model, X, y_true=None):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        try:
            dens_s = model.predict_density_samples(X, y_grid, n_samples=self.n_param_samples)  # [S,N,G]
            if dens_s.ndim != 3:
                raise Exception('Bad shape from predict_density_samples')
            Ey_s, Var_s = self._moments_from_density(dens_s, y_grid)  # [S,N]
            aleatoric  = Var_s.mean(axis=0)
            epistemic  = Ey_s.var(axis=0)
            total      = aleatoric + epistemic
        except Exception:
            dens = model.predict_density(X, y_grid)
            Ey, Var = self._moments_from_density(dens, y_grid)
            aleatoric = Var
            epistemic = np.zeros_like(aleatoric)
            total = aleatoric

        if self.decomposition == 'aleatoric':
            return aleatoric
        elif self.decomposition == 'epistemic':
            return epistemic
        else:
            return total
