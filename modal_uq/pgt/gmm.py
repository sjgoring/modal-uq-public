
import numpy as np
from sklearn.mixture import GaussianMixture
from .base import PGTBase
from ..registry import register

@register('pgt','gmm')
class JointGMM(PGTBase):
    def __init__(self, n_components=2, covariance_type='full', random_state=0, y_grid_points=512, y_pad=1.0):
        self.model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=random_state)
        self.y_grid_points = y_grid_points; self.y_pad = y_pad
        self._y_bounds = None; self._dimx = None

    def fit(self, X, y):
        XY = np.hstack([X, y.reshape(-1,1)])
        self._dimx = X.shape[1]
        self.model.fit(XY)
        self._y_bounds = (float(y.min()), float(y.max()))

    def _y_grid(self):
        lo, hi = self._y_bounds
        pad = self.y_pad * (hi - lo + 1e-6)
        return np.linspace(lo - pad, hi + pad, self.y_grid_points)

    def conditional_mode(self, x_query):
        y_grid = self._y_grid()
        from scipy.stats import multivariate_normal
        weights = self.model.weights_; means = self.model.means_; covs = self.model.covariances_
        xy = np.hstack([np.repeat(np.asarray(x_query).reshape(1,-1), len(y_grid), axis=0), y_grid.reshape(-1,1)])
        dens = np.zeros(len(y_grid))
        for wk, mk, Sk in zip(weights, means, covs):
            dens += wk * multivariate_normal.pdf(xy, mean=mk, cov=Sk)
        return float(y_grid[int(dens.argmax())])
