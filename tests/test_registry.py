
from modal_uq.registry import register, build

def test_registry_roundtrip():
    @register('dataset','_dummy')
    class D: 
        def __init__(self, **kwargs): pass
    obj = build('dataset','_dummy')
    assert obj is not None


def test_integrated_volume_kde_1d_and_2d():
    import numpy as np
    from modal_uq.uncertainty.quest import QUESTUncertainty

    class _DummyModel:
        @staticmethod
        def mdn_parameter_log_density(theta):
            # Isotropic Gaussian log-density up to an additive constant.
            return -0.5 * np.sum(theta ** 2, axis=1)

    rs = np.random.RandomState(0)
    # 1D samples
    theta1 = rs.normal(loc=0.0, scale=1.0, size=(500, 1))
    q = QUESTUncertainty(alpha=0.05)
    meta1 = q.integrated_volume_from_parameter_samples(theta1, model=_DummyModel())
    assert isinstance(meta1, float)
    assert meta1 >= 0.0

    # 2D samples
    theta2 = rs.normal(size=(800, 2))
    meta2 = q.integrated_volume_from_parameter_samples(theta2, model=_DummyModel())
    assert isinstance(meta2, float)
    assert meta2 >= 0.0


def test_integrated_volume_requires_model():
    import numpy as np
    from modal_uq.uncertainty.quest import QUESTUncertainty

    rs = np.random.RandomState(1)
    theta = rs.normal(size=(200, 4))
    q = QUESTUncertainty(alpha=0.05)
    try:
        q.integrated_volume_from_parameter_samples(theta)
        raised = False
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when model is missing"


def test_mdn_bma_not_implemented():
    import numpy as np
    from modal_uq.models.mdn import MixtureDensityModel

    model = MixtureDensityModel(inferential_choice={'predict': 'bma', 'approximate': 'posterior_predictive'})
    raised = False
    try:
        model.predict_density(np.zeros((2, 1)), np.linspace(-1, 1, 16), context='predict')
    except NotImplementedError:
        raised = True
    assert raised, "Expected NotImplementedError for deterministic model with bma inferential choice"


def test_variance_supports_density_collection():
    import numpy as np
    from modal_uq.uncertainty.variance import PredictiveVariance

    class _DummyModel:
        def default_y_grid(self, X, grid_points=32, y_pad=1.0):
            return np.linspace(-1, 1, 32)

        def predict_density(self, X, y_grid, context='predict'):
            n = len(X)
            base = np.exp(-0.5 * ((y_grid[None, :] - 0.0) / 0.2) ** 2)
            base = np.repeat(base, n, axis=0)
            if context == 'predict':
                return np.stack([base, 1.1 * base], axis=0)
            return base

    scores = PredictiveVariance(decomposition='total').score(_DummyModel(), np.zeros((4, 1)))
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
