
from ..uncertainty.random import RandomUncertainty
from ..registry import register

@register('acquisition','random')
class RandomAcq(RandomUncertainty):
    def score(self, model, X_pool):
        return super().score(model, X_pool)
