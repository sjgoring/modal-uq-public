
from .base import AcquisitionBase
from ..registry import register

@register('acquisition','mixture_ambiguity')
class MixtureAmbiguity(AcquisitionBase):
    def score(self, model, X_pool):
        if not hasattr(model, 'predict_mixture_params'):
            from .modal_entropy import ModalEntropy
            return ModalEntropy().score(model, X_pool)
        pi, mu, sigma2 = model.predict_mixture_params(X_pool)
        return 1.0 - pi.max(axis=1)
