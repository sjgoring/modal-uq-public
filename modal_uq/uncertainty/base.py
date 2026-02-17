
from abc import ABC, abstractmethod

class UncertaintyBase(ABC):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    @abstractmethod
    def score(self, model, X, y_true=None): ...
