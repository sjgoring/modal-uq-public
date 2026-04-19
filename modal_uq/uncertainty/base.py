
from abc import ABC, abstractmethod

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
