
from abc import ABC, abstractmethod

class AcquisitionBase(ABC):
    @abstractmethod
    def score(self, model, X_pool): ...
