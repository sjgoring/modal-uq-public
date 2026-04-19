import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from .base import ModelBase

class GaussianProcessModel(ModelBase):
    """Gaussian Process Regression model wrapper."""
    def __init__(self, kernel=None, inferential_choice=None):
        super().__init__(inferential_choice)
        if kernel is None:
            kernel = C(1.0, (1e-4, 1e6)) * RBF(1.0, (1e-6, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self._y_min = None
        self._y_max = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.gp.fit(X, y)
        self._y_min = np.min(y)
        self._y_max = np.max(y)

    def predict_density(self, X, y_grid, context='predict'):
        # Predict mean and std for each X
        mu, std = self.gp.predict(X, return_std=True)
        std = np.maximum(std, 1e-6)  # Avoid zero std
        # Compute Gaussian density for each y_grid
        dens = np.exp(-0.5 * ((y_grid[None, :] - mu[:, None]) ** 2) / (std[:, None] ** 2))
        dens /= (np.sqrt(2 * np.pi) * std[:, None])
        return dens

    def predict_mixture_params(self, X):
        mu, std = self.gp.predict(X, return_std=True)
        return mu, std
