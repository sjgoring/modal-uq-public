
from modal_uq.registry import register, build

def test_registry_roundtrip():
    @register('dataset','_dummy')
    class D: 
        def __init__(self, **kwargs): pass
    obj = build('dataset','_dummy')
    assert obj is not None


def test_meta_quest_kde_1d_and_2d():
    import numpy as np
    from modal_uq.uncertainty.quest import QUESTUncertainty

    rs = np.random.RandomState(0)
    # 1D samples
    theta1 = rs.normal(loc=0.0, scale=1.0, size=(500, 1))
    q = QUESTUncertainty(alpha=0.05)
    meta1 = q.meta_quest_from_parameter_samples(theta1, grid_points_per_dim=128)
    assert isinstance(meta1, float)
    assert meta1 >= 0.0

    # 2D samples
    theta2 = rs.normal(size=(800, 2))
    meta2 = q.meta_quest_from_parameter_samples(theta2, grid_points_per_dim=48)
    assert isinstance(meta2, float)
    assert meta2 >= 0.0


def test_meta_quest_high_dim_raises():
    import numpy as np
    from modal_uq.uncertainty.quest import QUESTUncertainty

    rs = np.random.RandomState(1)
    theta = rs.normal(size=(200, 4))
    q = QUESTUncertainty(alpha=0.05)
    try:
        q.meta_quest_from_parameter_samples(theta, grid_points_per_dim=16)
        raised = False
    except NotImplementedError:
        raised = True
    assert raised, "Expected NotImplementedError for P>3"
