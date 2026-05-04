
import numpy as np
from .base import UncertaintyBase
from ..registry import register
import scipy.integrate as integrate

@register('uncertainty','nll')
class NLLScore(UncertaintyBase):
    def __init__(self, decomposition='total', n_jobs=None):
        assert decomposition in {'total', 'aleatoric', 'epistemic'}
        self.decomposition = decomposition
        self.n_jobs = n_jobs  # Number of parallel jobs for compute-heavy operations

    def score_total(self, model, X, y_true=None):
        if y_true is None:
            raise ValueError('NLL requires y_true')
        y_grid = model.default_y_grid(X)
        dens = self._predict_density_collection(model, X, y_grid, context='predict')
        idx = np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)
        nll_scores = []
        for s in range(dens.shape[0]):
            p = dens[s, np.arange(len(y_true)), idx] + 1e-12
            nll_scores.append(-np.log(p))
        return np.mean(np.stack(nll_scores, axis=0), axis=0)

    def score_aleatoric(self, model, X, y_true=None):
        raise NotImplementedError('NLL does not define an aleatoric decomposition.')

    def score_epistemic(self, model, X, y_true=None):
        raise NotImplementedError('NLL does not define an epistemic decomposition.')

    def score(self, model, X, y_true=None):
        if self.decomposition == 'total':
            return self.score_total(model, X, y_true=y_true)
        if self.decomposition == 'aleatoric':
            return self.score_aleatoric(model, X, y_true=y_true)
        if self.decomposition == 'epistemic':
            return self.score_epistemic(model, X, y_true=y_true)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")

@register('uncertainty','crps_proxy')
class CRPSProxy(UncertaintyBase):
    def __init__(self, decomposition='total', n_jobs=None):
        assert decomposition in {'total', 'aleatoric', 'epistemic'}
        self.decomposition = decomposition
        self.n_jobs = n_jobs  # Number of parallel jobs for compute-heavy operations

    def score_total(self, model, X, y_true=None):
        y_grid = model.default_y_grid(X)
        dens = self._predict_density_collection(model, X, y_grid, context='predict')
        crps_scores = []
        for s in range(dens.shape[0]):
            cdf = np.cumsum(dens[s], axis=1)
            cdf /= (cdf[:, -1][:, None] + 1e-12)
            crps_scores.append(integrate.trapezoid(np.abs(cdf - 0.5), y_grid, axis=1))
        return np.mean(np.stack(crps_scores, axis=0), axis=0)

    def score_aleatoric(self, model, X, y_true=None):
        raise NotImplementedError('CRPS proxy does not define an aleatoric decomposition.')

    def score_epistemic(self, model, X, y_true=None):
        raise NotImplementedError('CRPS proxy does not define an epistemic decomposition.')

    def score(self, model, X, y_true=None):
        if self.decomposition == 'total':
            return self.score_total(model, X, y_true=y_true)
        if self.decomposition == 'aleatoric':
            return self.score_aleatoric(model, X, y_true=y_true)
        if self.decomposition == 'epistemic':
            return self.score_epistemic(model, X, y_true=y_true)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")
