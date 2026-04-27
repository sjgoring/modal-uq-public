import numpy as np
from cgmm import ConditionalGMMRegressor

from .base import ModelBase
from ..registry import register

@register('model','condgmm')

class CondGMM(ModelBase):
    """Conditional Gaussian Mixture Model regressor using cgmm package."""
    def __init__(self, inferential_choice=None, n_components=5, **kwargs):
        super().__init__(inferential_choice)
        self.model = ConditionalGMMRegressor(n_components=n_components, **kwargs)
        self._y_min = None
        self._y_max = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self._y_min = float(np.min(y))
        self._y_max = float(np.max(y))

    def predict_density(self, X, y_grid, context='predict'):
        strategy = self.resolve_inferential_choice(context=context)
        if strategy == 'bma':
            raise NotImplementedError(
                "BMA inferential choice is not implemented for CondGMM because it is deterministic."
            )
        if strategy == 'posterior_predictive':
            # CondGMM is deterministic, so we return the same density regardless of the inferential choice.
            pass

        dens = np.zeros((X.shape[0], y_grid.shape[0]))
        gmms = self.model.condition(X)  # Returns list of sklearn GaussianMixture for each X.
        for gmm, idx in zip(gmms, range(X.shape[0])):
            dens[idx] = gmm.score_samples(y_grid.reshape(-1, 1))  # log density
            dens[idx] = np.exp(dens[idx])  # convert log density to density
            # Normalize numerically over the provided grid for stable downstream metrics.
            integral = np.trapz(dens[idx], y_grid)
            if not np.isfinite(integral) or integral <= 0:
                raise ValueError(
                    f"Density for sample {idx} has invalid integral ({integral})."
                )
            if not np.isclose(integral, 1.0, atol=1e-3):
                dens[idx] /= integral
        return dens

    # Deterministic models should not override this method, as they do not provide a second-order distribution. The default implementation raises NotImplementedError to indicate that this functionality is not available for CondGMM.
    # def get_second_order_distribution(self, X, y_grid, context='predict'):
        
    #     raise NotImplementedError(
    #         "CondGMM is deterministic and does not provide a second-order distribution."
    #     )
    def get_params(self, X):
        """Return parameters of the fitted model at X, for empirical second-order distribution creation."""
        if self._y_min is None or self._y_max is None:
            raise ValueError("Model must be fitted before getting parameters.")
        
        # Dictionary of parameters
        gmm = self.model.condition(X)
        weights = gmm.weights_         # (n_components,)
        means = gmm.means_             # (n_components, n_targets)
        covariances = gmm.covariances_ # (n_components, n_targets, n_targets)
        # return as a single array weights, means, covariances.
        return np.stack([weights, means.flatten(), covariances.flatten()])