import numpy as np
from .base import UncertaintyBase
from ..registry import register
import scipy.integrate as integrate

@register('uncertainty','variance')
class PredictiveVariance(UncertaintyBase):
    """
    Predictive variance with optional decomposition.

    Uses dual inferential_choice contexts for uncertainty decomposition:
    
    - total:      Var[Y | x] computed from predict context
    - aleatoric:  E_θ[ Var(Y | x, θ) ] from approximate context (true DGP with known params)
    - epistemic:  total - aleatoric (difference from approximate to predict)

    The approximate context represents the true data generating process with point-estimate
    parameters (minimal epistemic uncertainty), while predict includes parameter uncertainty.
    
    Uses model.predict_density(X, y_grid, context='predict'|'approximate') that returns
    either [N,G] (single density) or [S,N,G] (many densities).

    Aggregation semantics:
    - single density: score is computed directly on that density
    - many densities: score is computed per density then averaged over S
    """
    def __init__(self, decomposition='total', grid_points=512, y_pad=1.0, n_param_samples=20):
        assert decomposition in {'total','aleatoric','epistemic'}
        self.decomposition = decomposition
        self.grid_points = grid_points
        self.y_pad = y_pad
        self.n_param_samples = n_param_samples

    @staticmethod
    def _normalize_last_axis(arr, y_grid):
        """
        Normalize densities along the last axis (G), regardless of arr being [N,G] or [S,N,G].
        """
        Z = integrate.trapezoid(arr, y_grid, axis=-1)     # shape: [N] or [S,N]
        Z = np.expand_dims(Z, axis=-1)         # -> [N,1] or [S,N,1]
        return arr / (Z + 1e-12)

    @staticmethod
    def _moments_from_density(dens, y_grid):
        """
        Compute E[Y] and Var[Y] from densities given on a shared y_grid.
        dens can be [N,G] or [S,N,G]. Returns (Ey, Var) with shape [N] or [S,N].
        """
        dens = PredictiveVariance._normalize_last_axis(dens, y_grid)  # preserve original ndim

        # Broadcast y and y^2 to match dens shape
        y = y_grid
        if dens.ndim == 3:   # [S,N,G]
            y = y[None, None, :]        # -> [1,1,G]
        else:                # [N,G]
            y = y[None, :]              # -> [1,G]

        Ey  = integrate.trapezoid(dens * y,          y_grid, axis=-1)  # [N] or [S,N]
        Ey2 = integrate.trapezoid(dens * (y**2),     y_grid, axis=-1)  # [N] or [S,N]
        Var = Ey2 - Ey**2                                   # [N] or [S,N]
        return Ey, Var

    def _compute_total(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        _, Var_pred_s = self._moments_from_density(dens_pred, y_grid)
        return Var_pred_s.mean(axis=0)

    def _compute_aleatoric(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_approx = self._predict_density_collection(model, X, y_grid, context='approximate')
        _, Var_approx_s = self._moments_from_density(dens_approx, y_grid)
        return Var_approx_s.mean(axis=0)

    def _compute_epistemic(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        Ey_pred, _ = self._moments_from_density(dens_pred, y_grid)

        if Ey_pred.ndim == 2:
            return np.var(Ey_pred, axis=0)

        return np.zeros(len(X), dtype=float)
        
    def score_total(self, model, X, y_true=None):
        return self._compute_total(model, X)

    def score_aleatoric(self, model, X, y_true=None):
        return self._compute_aleatoric(model, X)

    def score_epistemic(self, model, X, y_true=None):
        return self._compute_epistemic(model, X)

    def score(self, model, X, y_true=None):
        """Dispatch to total/aleatoric/epistemic score by decomposition."""
        if self.decomposition == 'total':
            return self.score_total(model, X, y_true=y_true)
        if self.decomposition == 'aleatoric':
            return self.score_aleatoric(model, X, y_true=y_true)
        if self.decomposition == 'epistemic':
            return self.score_epistemic(model, X, y_true=y_true)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")