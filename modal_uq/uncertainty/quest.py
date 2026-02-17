
import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','quest')
class QUESTUncertainty(UncertaintyBase):
    """Stub for QUEST (to be specified in paper). Returns zeros until implemented."""
    def score(self, model, X, y_true=None):
        y_grid = model.default_y_grid(X)
        dens = model.predict_density(X, y_grid)
        # TODO: replace with your bespoke computation around conditional mode(s)
        return np.zeros(X.shape[0])
