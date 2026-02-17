
from abc import ABC, abstractmethod

class PGTBase(ABC):
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def conditional_mode(self, x_query): ...
