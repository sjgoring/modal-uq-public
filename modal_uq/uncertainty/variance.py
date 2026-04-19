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
    
    Requires model.predict_density_samples(X, y_grid, context='predict'|'approximate', n_samples)
    to return [S,N,G] densities. Falls back to deterministic density if unavailable.
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
        try:
            dens_pred = model.predict_density_samples(
                X, y_grid, context='predict', n_samples=self.n_param_samples
            )
            if dens_pred.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")
            Ey_pred_s, Var_pred_s = self._moments_from_density(dens_pred, y_grid)
            return Ey_pred_s.var(axis=0) + Var_pred_s.mean(axis=0)
        except Exception:
            # Deterministic density fallback: [N,G]
            dens_pred = model.predict_density(X, y_grid, context='predict')
            _, Var_pred = self._moments_from_density(dens_pred, y_grid)
            return Var_pred

    def _compute_aleatoric(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        try:
            dens_approx = model.predict_density_samples(
                X, y_grid, context='approximate', n_samples=self.n_param_samples
            )
            if dens_approx.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")
            _, Var_approx_s = self._moments_from_density(dens_approx, y_grid)
            return Var_approx_s.mean(axis=0)
        except Exception:
            dens_approx = model.predict_density(X, y_grid, context='approximate')
            _, Var_approx = self._moments_from_density(dens_approx, y_grid)
            return Var_approx

    def _compute_epistemic(self, model, X):
        # Not implemented for variance under the current inferential_choice setup.
        aleatoric = self._compute_aleatoric(model, X)
        return np.zeros_like(aleatoric)

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