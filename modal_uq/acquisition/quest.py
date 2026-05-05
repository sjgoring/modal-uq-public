
from ..uncertainty.quest import QUESTUncertainty
from ..registry import register

@register('acquisition','quest')
class QUESTAcq(QUESTUncertainty):
    def score(self, model, X_pool):
        return super().score(model, X_pool)
