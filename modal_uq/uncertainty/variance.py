import numpy as np
from .base import UncertaintyBase
from ..models.base import InferentialChoiceConfig
from ..registry import register
import scipy.integrate as integrate

@register('uncertainty','variance')
class PredictiveVariance(UncertaintyBase):
    """
    Predictive variance with optional decomposition.

    Uses dual inferential_choice contexts for uncertainty decomposition:
    
    - predict: candidate distributions, returned as [S,N,G] or [N,G]
    - approximate: posterior predictive reference, returned as [N,G] or [S,N,G]

    Variance decompositions:
    - total:      BMA of the squared loss around the posterior predictive mean
    - aleatoric:  BMA of the variance measure over all predict samples
    - epistemic:  BMA of the squared deviation of each candidate mean from the posterior predictive mean
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
    def _posterior_predictive_density(dens):
        """Collapse a density collection to its posterior predictive density."""
        dens = np.asarray(dens)
        if dens.ndim == 3:
            return dens.mean(axis=0)
        return dens

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

    @staticmethod
    def _validate_inferential_choices(model):
        cfg = model.get_inferential_choice_config()
        predict = InferentialChoiceConfig.canonicalize_strategy(cfg.predict)
        approximate = InferentialChoiceConfig.canonicalize_strategy(cfg.approximate)

        if predict != 'bma' or approximate != 'posterior_predictive':
            raise NotImplementedError(
                "PredictiveVariance requires inferential choices predict='bma' and "
                "approximate='posterior_predictive'. "
                f"Current settings: predict='{cfg.predict}', approximate='{cfg.approximate}'."
            )

    def _compute_total(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        dens_approx = self._predict_density_collection(model, X, y_grid, context='approximate')

        dens_ref = self._posterior_predictive_density(dens_approx)
        Ey_ref, _ = self._moments_from_density(dens_ref, y_grid)

        candidate_densities = [dens_pred, dens_approx]
        squared_losses = []

        for dens_collection in candidate_densities:
            if dens_collection.ndim == 2:
                dens_collection = dens_collection[None, ...]

            for dens_s in dens_collection:
                Ey_s, Var_s = self._moments_from_density(dens_s, y_grid)
                squared_losses.append(Var_s + (Ey_s - Ey_ref) ** 2)

        return np.mean(np.asarray(squared_losses), axis=0)

    def _compute_aleatoric(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        _, Var_pred_s = self._moments_from_density(dens_pred, y_grid)
        return Var_pred_s.mean(axis=0)

    def _compute_epistemic(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        dens_ref = self._posterior_predictive_density(
            self._predict_density_collection(model, X, y_grid, context='approximate')
        )
        Ey_pred, _ = self._moments_from_density(dens_pred, y_grid)
        Ey_ref, _ = self._moments_from_density(dens_ref, y_grid)

        if Ey_pred.ndim == 1:
            return np.zeros_like(Ey_pred)

        return np.mean((Ey_pred - Ey_ref[None, :]) ** 2, axis=0)
        
    def score_total(self, model, X, y_true=None):
        return self._compute_total(model, X)

    def score_aleatoric(self, model, X, y_true=None):
        return self._compute_aleatoric(model, X)

    def score_epistemic(self, model, X, y_true=None):
        return self._compute_epistemic(model, X)

    def score(self, model, X, y_true=None):
        self._validate_inferential_choices(model)
        """Dispatch to total/aleatoric/epistemic score by decomposition."""
        if self.decomposition == 'total':
            return self.score_total(model, X, y_true=y_true)
        if self.decomposition == 'aleatoric':
            return self.score_aleatoric(model, X, y_true=y_true)
        if self.decomposition == 'epistemic':
            return self.score_epistemic(model, X, y_true=y_true)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")