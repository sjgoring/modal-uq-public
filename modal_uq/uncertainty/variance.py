import numpy as np
from .base import UncertaintyBase
from ..registry import register
import scipy.integrate as integrate

@register('uncertainty','variance')
class PredictiveVariance(UncertaintyBase):
    """
    Predictive variance with optional decomposition.

    Uses dual marginalization contexts for uncertainty decomposition:
    
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

    def score(self, model, X, y_true=None):
        """Compute variance using predict and approximate contexts.
        
        Uncertainty components (NOT additive decomposition):
        - aleatoric:  Var computed from approximate context (true DGP with known params)
        - total:      Var computed from predict context (includes parameter uncertainty)
        - epistemic:  NOT DEFINED - decomposition depends on marginalization strategy choice
        
        Note: Epistemic uncertainty cannot be universally defined as (total - aleatoric) because
        the relationship between these quantities depends on the specific marginalization strategies
        chosen for predict vs. approximate contexts. Use separate measures if strict decomposition
        semantics are required.
        """
        # Build a default grid per model (shared across X in this batch)
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)

        try:
            # Sample from both prediction and approximation contexts
            dens_pred = model.predict_density_samples(X, y_grid, context='predict', n_samples=self.n_param_samples)  # [S,N,G]
            dens_approx = model.predict_density_samples(X, y_grid, context='approximate', n_samples=self.n_param_samples)  # [S,N,G]
            
            if dens_pred.ndim != 3 or dens_approx.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")

            # Compute variances from both contexts
            Ey_pred_s, Var_pred_s = self._moments_from_density(dens_pred, y_grid)  # [S,N] each
            Ey_approx_s, Var_approx_s = self._moments_from_density(dens_approx, y_grid)  # [S,N] each

            # Variance components:
            aleatoric = Var_approx_s.mean(axis=0)    # E_theta[Var(Y | x, θ)] from approximate  -> [N]
            total = Ey_pred_s.var(axis=0) + Var_pred_s.mean(axis=0)  # Var(Y | x) from predict -> [N]

        except Exception:
            # Deterministic density fallback: [N,G]
            dens_pred = model.predict_density(X, y_grid, context='predict')
            dens_approx = model.predict_density(X, y_grid, context='approximate')
            
            Ey_pred, Var_pred = self._moments_from_density(dens_pred, y_grid)  # [N]
            Ey_approx, Var_approx = self._moments_from_density(dens_approx, y_grid)  # [N]
            
            aleatoric = Var_approx
            total = Var_pred

        if self.decomposition == 'aleatoric':
            return aleatoric
        elif self.decomposition == 'epistemic':
            # Epistemic variance: parameter uncertainty in our model (predict context)
            # Var_θ_p[E[Y | x, θ_p]] - variance of predicted means across parameter samples
            #
            # The computation depends on marginalization strategies:
            # Case 1: No expectation (both point/bma) → epistemic = 0
            # Case 2: 1 expectation (predict posterior) → epistemic = var(Ey_pred_s)
            # Case 3: 1 expectation (approx posterior) → epistemic = 0 (no parameter variance in predict)
            # Case 4: 2 expectations (both posterior) → epistemic = var(Ey_pred_s)
            #
            # Note: epistemic variance depends on predict context (parameter uncertainty in our model),
            # not on approximate context (which only affects aleatoric).
            
            ## Not implemented!



            # try:
            #     S_pred = Ey_pred_s.shape[0]
                
            #     if S_pred > 1:
            #         # Posterior_weighted: compute variance across parameter samples
            #         epistemic = Ey_pred_s.var(axis=0)  # Var_θ_p[E[Y | x, θ_p]] -> [N]
            #     else:
            #         # Point estimate or bma_expected: no parameter variance
            #         epistemic = np.zeros_like(aleatoric)
                    
            # except NameError:
            #     # Fallback for deterministic case: no parameter uncertainty
            epistemic = np.zeros_like(aleatoric)
            return epistemic
        else:
            return total