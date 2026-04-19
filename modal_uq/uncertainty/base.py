
from abc import ABC, abstractmethod
import numpy as np

class UncertaintyBase(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def score(self, model, X, y_true=None):
        ...

    @abstractmethod
    def score_total(self, model, X, y_true=None):
        ...

    @abstractmethod
    def score_aleatoric(self, model, X, y_true=None):
        ...

    @abstractmethod
    def score_epistemic(self, model, X, y_true=None):
        ...

    @staticmethod
    def _as_density_collection(dens):
        """Normalize model densities to [S,N,G]."""
        dens = np.asarray(dens)
        if dens.ndim == 2:
            return dens[None, ...]
        if dens.ndim == 3:
            return dens
        raise ValueError("predict_density must return [N,G] or [S,N,G].")

    def _predict_density_collection(self, model, X, y_grid, context='predict'):
        dens = model.predict_density(X, y_grid, context=context)
        return self._as_density_collection(dens)
