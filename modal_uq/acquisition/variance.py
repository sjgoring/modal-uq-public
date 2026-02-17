
from ..uncertainty.variance import PredictiveVariance
from ..registry import register

@register('acquisition','variance')
class VarianceAcq(PredictiveVariance):
    def score(self, model, X_pool):
        return super().score(model, X_pool)
