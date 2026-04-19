import numpy as np
import warnings
from skmdn import MixtureDensityEstimator

from .base import ModelBase
from ..registry import register

@register('model','mdn')
class MixtureDensityModel(ModelBase):
    """Mixture Density network model wrapper.
        kwargs are passed to the underlying MixtureDensityEstimator from scikit-mdn.
    """
    def __init__(self, inferential_choice=None, **kwargs):
        super().__init__(inferential_choice)
        self.mdn = MixtureDensityEstimator(**kwargs)
        self._y_min = None
        self._y_max = None

    def fit(self, X, y, X_val=None, y_val=None):
        # Expanding y dim if required.
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)

        self.mdn.fit(X, y)
        self._y_min = np.min(y)
        self._y_max = np.max(y)

    def predict_density(self, X, y_grid, context='predict'):
        strategy = self.resolve_inferential_choice(context=context)
        if strategy == 'bma':
            raise NotImplementedError(
                "BMA inferential choice is not implemented for MixtureDensityModel because it is deterministic."
            )

        warnings.warn(
            "MixtureDensityModel is deterministic; using posterior_predictive as a compatibility path "
            "(this is not a true posterior predictive over parameter uncertainty).",
            UserWarning,
            stacklevel=2,
        )

        # Predict mean and std for each X
        # mu, std = self.mdn.predict(X, return_std=True)
        # std = np.maximum(std, 1e-6)  # Avoid zero std
        # # Compute Gaussian density for each y_grid
        # dens = np.exp(-0.5 * ((y_grid[None, :] - mu[:, None]) ** 2) / (std[:, None] ** 2))
        # dens /= (np.sqrt(2 * np.pi) * std[:, None])
        # return dens
        y_min = min(y_grid)
        y_max = max(y_grid)
        res = y_grid.shape[0]
        out = self.mdn.pdf(X, resolution=res, y_min=y_min, y_max=y_max)[0]
        # print(out)
        return out
        # The above works on the assumption the y_grid is the same as the one used in the MDN fit. If we want to allow arbitrary y_grids, we may need to compute the density manually from the predicted mixture parameters.

    def get_second_order_distribution(self, X, y_grid, context='predict'):
        raise NotImplementedError(
            "MixtureDensityModel is deterministic and does not provide a second-order distribution."
        )

    # def predict_mixture_params(self, X):
    #     mu, std = self.gp.predict(X, return_std=True)
    #     return mu, std
