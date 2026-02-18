import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','variance')
class PredictiveVariance(UncertaintyBase):
    """
    Predictive variance with optional decomposition.

    - total:      Var[Y | x]
    - aleatoric:  E_θ[ Var(Y | x, θ) ]
    - epistemic:  Var_θ[ E(Y | x, θ) ]

    Uses model.predict_density_samples(X, y_grid, n_samples)-> [S,N,G] if available;
    otherwise falls back to deterministic density [N,G] (epistemic=0).
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
        Z = np.trapz(arr, y_grid, axis=-1)     # shape: [N] or [S,N]
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

        Ey  = np.trapz(dens * y,          y_grid, axis=-1)  # [N] or [S,N]
        Ey2 = np.trapz(dens * (y**2),     y_grid, axis=-1)  # [N] or [S,N]
        Var = Ey2 - Ey**2                                   # [N] or [S,N]
        return Ey, Var

    def score(self, model, X, y_true=None):
        # Build a default grid per model (shared across X in this batch)
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)

        try:
            # Monte-Carlo parameter samples: [S,N,G]
            dens_s = model.predict_density_samples(X, y_grid, n_samples=self.n_param_samples)
            if dens_s.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")

            Ey_s, Var_s = self._moments_from_density(dens_s, y_grid)  # [S,N] each

            # Decomposition:
            aleatoric = Var_s.mean(axis=0)    # E_theta[Var(Y | x, θ)]     -> [N]
            epistemic = Ey_s.var(axis=0)      # Var_theta[E(Y | x, θ)]     -> [N]
            total     = aleatoric + epistemic # Var(Y | x)                 -> [N]

        except Exception:
            # Deterministic density fallback: [N,G]   (epistemic = 0)
            dens = model.predict_density(X, y_grid)
            Ey, Var = self._moments_from_density(dens, y_grid)  # [N]
            aleatoric = Var
            epistemic = np.zeros_like(aleatoric)
            total = aleatoric

        if self.decomposition == 'aleatoric':
            return aleatoric
        elif self.decomposition == 'epistemic':
            return epistemic
        else:
            return total