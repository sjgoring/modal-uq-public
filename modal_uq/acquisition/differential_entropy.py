
from ..uncertainty.differential_entropy import DifferentialEntropy
from ..registry import register

@register('acquisition','differential_entropy')
class DifferentialEntropyAcq(DifferentialEntropy):
    def score(self, model, X_pool):
        return super().score(model, X_pool)
