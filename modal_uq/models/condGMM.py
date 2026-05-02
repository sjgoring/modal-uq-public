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
        gmms = self.model.condition(X)
        weights = []
        means = []
        covariances = []
        for gmm in gmms:
            weights.append(gmm.weights_)         # (n_components,)
            means.append(gmm.means_.flatten())           # (n_components, n_targets)
            covariances.append(gmm.covariances_.flatten()) # (n_components, n_targets, n_targets)
            # note: n_targets is 1 for univariate regression, so means and covariances will have a trailing dimension of size 1. We can flatten these for simplicity in downstream processing.
        # Build per-sample parameter vectors and return shape [n_params, n_X]
        per_sample_params = []
        for w, m, c in zip(weights, means, covariances):
            vec = np.concatenate([np.asarray(w).ravel(), np.asarray(m).ravel(), np.asarray(c).ravel()], axis=0)
            per_sample_params.append(vec)
        # Stack as columns: [n_params, n_X]
        params = np.stack(per_sample_params, axis=1)
        return params
    
    def sample_output(self, X, n_samples, rng):
        """
        Sample outputs from the conditional GMM for each input in X.

        Returns array of shape (n_samples, N, d).
        """
        gmms = self.model.condition(X)
        N = X.shape[0]
        # Determine output dimensionality from GMM means
        d = gmms[0].means_.shape[1]
        samples = np.zeros((n_samples, N, d))
        for i, gmm in enumerate(gmms):
            s, _ = gmm.sample(n_samples)
            samples[:, i, :] = s.reshape(n_samples, d)
        return samples

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        """
        Build per-input axis-aligned bounds by sampling the conditional GMM and using quantiles.

        Returns bounds shaped (N, d, 2).
        """
        rng = np.random.default_rng(rng)
        samples = self.sample_output(X, n_samples, rng)
        # samples shape: (n_samples, N, d)
        lows = np.quantile(samples, q_low, axis=0)  # (N, d)
        highs = np.quantile(samples, q_high, axis=0)  # (N, d)
        pad = (highs - lows) * pad_frac
        bounds = np.stack([lows - pad, highs + pad], axis=-1)  # (N, d, 2)
        return bounds

    def density_function_for_input(self, X):
        """
        Return a density callable: density_func(points, input_index)

        `points` shape: (M, d)
        Returns: (M,) densities for the specified input_index.
        """
        gmms = self.model.condition(X)

        def density_func(points, input_index=0):
            gmm = gmms[input_index]
            logp = gmm.score_samples(np.asarray(points).reshape(-1, gmm.means_.shape[1]))
            return np.exp(logp)

        return density_func
