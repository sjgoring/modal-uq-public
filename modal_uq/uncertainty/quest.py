
import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','quest')
class QUESTUncertainty(UncertaintyBase):
    def score(self, model, X, y_true=None):
        y_grid = model.default_y_grid(X)
        dens = model.predict_density(X, y_grid)
        # TODO: implement your measure
        return np.zeros(X.shape[0])
