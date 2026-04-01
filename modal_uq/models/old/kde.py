
import numpy as np
from ..base import ModelBase
from ...registry import register
from conditional_kde import ConditionalGaussianKernelDensity

# Wrapper class for conditional_kde's ConditionalGaussianKernelDensity, which is a conditional kernel density estimator that can be used for regression tasks. It estimates the conditional density of the target variable given the input features, allowing for uncertainty quantification in predictions.

@register('model','kde')
class ConditionalKDEModel(ModelBase):
    def __init__(self, bandwidth=0.5, kernel='gaussian', marginalization=None):
        super().__init__(marginalization=marginalization)
        self.bandwidth = bandwidth; self.kernel = kernel
        self._y_min = None; self._y_max = None
        self._kde = ConditionalGaussianKernelDensity(
         whitening_algorithm = "rescale",
            bandwidth = "optimized",
            steps = 10,
            cv_fold = 5,
            n_jobs = -1,
            verbose = 1,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        XY = np.hstack((X, y.reshape(-1, 1)))
        self._kde.fit(XY, features = list(range(X.shape[0])).append( "y"))        

    def predict_density(self, X, y_grid, context='predict'):
        # Deterministic model: context parameter is ignored]
        self._kde.score_samples(X, y_grid)
        

if __name__ == "__main__":
    # Testing
    # Aiming to recreate results from https://conditional-kde.readthedocs.io/en/latest/notebooks/ConditionalKDE.html
    data = np.random.rand(100, 2)

    data_xy = np.concatenate(
    (
        np.random.multivariate_normal(mean1, cov1, 10000),
        np.random.multivariate_normal(mean2, cov2, 10000)
    ),
    axis = 0
)