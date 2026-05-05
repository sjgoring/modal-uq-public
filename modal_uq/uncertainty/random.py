import numpy as np
import scipy.integrate as integrate

from .base import UncertaintyBase
from ..models.base import InferentialChoiceConfig
from ..registry import register

@register('uncertainty','random')
class RandomUncertainty(UncertaintyBase):
    """
    Baseline random uncertainty measure that ignores the model and returns random scores.

    """
    def __init__(self, decomposition="total",**kwargs):
        super().__init__()
        self.decomposition = decomposition

    def score_total(self, model, X, y_true=None):
        return np.random.rand(X.shape[0])

    def score_aleatoric(self, model, X, y_true=None):
        return np.random.rand(X.shape[0])

    def score_epistemic(self, model, X, y_true=None):
        return np.random.rand(X.shape[0])

    def score(self, model, X, y_true=None):
        if self.decomposition == 'total':
            return self.score_total(model, X, y_true=y_true)
        if self.decomposition == 'aleatoric':
            return self.score_aleatoric(model, X, y_true=y_true)
        if self.decomposition == 'epistemic':
            return self.score_epistemic(model, X, y_true=y_true)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")