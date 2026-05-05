import numpy as np
import pandas as pd

from modal_uq.datasets.synthetic_constant_var import SyntheticConstantVarDataset
from modal_uq.experiments.selective import SelectivePrediction
from modal_uq.metrics.mode_errors import likelihood_ratio_measure


def test_synthetic_constant_var_dataset_returns_padded_y_grid():
    ds = SyntheticConstantVarDataset(n_samples=20, y_min=-2.0, y_max=2.0, y_grid_size=41, y_pad=1.0, seed=0)

    X, y, global_mode, mode_ids, y_densities, y_grid = ds.get_data()

    assert X.shape == (20, 1)
    assert y.shape == (20,)
    assert global_mode.shape == (2,)
    assert mode_ids.shape == (20,)
    assert y_densities.shape == (20, 41)
    assert y_grid.shape == (41,)
    assert y_grid[0] < ds.y_min
    assert y_grid[-1] > ds.y_max
    assert np.all(np.isin(y, y_grid))


def test_selective_curve_uses_cached_y_grid(monkeypatch):
    y_grid = np.array([-2.0, 0.0, 2.0])
    y_true = np.array([-2.0, 2.0])
    y_pred = np.array([0.0, 0.0])
    uncertainty = np.array([0.9, 0.1])
    true_dens = np.array([
        [0.3, 0.4, 0.3],
        [0.2, 0.5, 0.3],
    ])
    est_dens = np.array([
        [0.2, 0.5, 0.3],
        [0.3, 0.4, 0.3],
    ])

    captured = {}

    def fake_risk_coverage(y_true_arg, y_pred_arg, uncertainty_arg, risk_fn, risk_fn_kwargs, steps=20, true_dens=None, est_dens=None, y_grid=None):
        captured["y_grid"] = y_grid
        captured["y_true"] = y_true_arg.copy()
        captured["y_pred"] = y_pred_arg.copy()
        return np.array([1.0]), np.array([0.5])

    class DummyModel:
        def default_y_grid(self, X, grid_points=512, y_pad=1.0):
            raise AssertionError("default_y_grid should not be called when a cached y_grid is available")

    exp = SelectivePrediction.__new__(SelectivePrediction)
    exp.model = DummyModel()
    exp.X_test = np.zeros((2, 1))
    exp.y_test = y_true
    exp.y_grid = y_grid
    exp.results = {
        "y_mode_pred": y_pred,
        "true_dens": true_dens,
        "est_dens": est_dens,
        "y_grid": y_grid,
    }
    exp._get_risk_fn = lambda metric_name: (likelihood_ratio_measure, {"reference_dist": "true"})

    monkeypatch.setattr("modal_uq.experiments.selective.risk_coverage", fake_risk_coverage)

    curve = SelectivePrediction.compute_selective_curve(
        exp,
        pd.DataFrame({"uncertainty": uncertainty}),
        "uncertainty",
        metric_name="lr_true",
        steps=0,
    )

    assert np.array_equal(captured["y_grid"], y_grid)
    assert np.array_equal(captured["y_true"], y_true)
    assert np.array_equal(captured["y_pred"], y_pred)
    assert curve["risk"].shape == (1,)
    assert curve["coverages"].shape == (1,)
    assert curve["aurc"] == 0.0


def test_selective_curve_uses_mode_aligned_truth_for_lr(monkeypatch):
    y_grid = np.array([-2.0, 0.0, 2.0])
    y_mode_true = np.array([-2.0, 2.0])
    y_test = np.array([-1.5, 1.5])
    y_pred = np.array([0.0, 0.0])
    uncertainty = np.array([0.9, 0.1])
    true_dens = np.array([
        [0.3, 0.4, 0.3],
        [0.2, 0.5, 0.3],
    ])
    est_dens = np.array([
        [0.2, 0.5, 0.3],
        [0.3, 0.4, 0.3],
    ])

    captured = {}

    def fake_risk_coverage(y_true_arg, y_pred_arg, uncertainty_arg, risk_fn, risk_fn_kwargs, steps=20, true_dens=None, est_dens=None, y_grid=None):
        captured["y_true"] = y_true_arg.copy()
        captured["y_pred"] = y_pred_arg.copy()
        captured["y_grid"] = y_grid
        return np.array([1.0]), np.array([0.5])

    class DummyModel:
        def default_y_grid(self, X, grid_points=512, y_pad=1.0):
            raise AssertionError("default_y_grid should not be called when a cached y_grid is available")

    exp = SelectivePrediction.__new__(SelectivePrediction)
    exp.model = DummyModel()
    exp.X_test = np.zeros((2, 1))
    exp.y_test = y_test
    exp.y_grid = y_grid
    exp.results = {
        "y_mode_true": y_mode_true,
        "y_mode_pred": y_pred,
        "true_dens": true_dens,
        "est_dens": est_dens,
        "y_grid": y_grid,
    }
    exp._get_risk_fn = lambda metric_name: (likelihood_ratio_measure, {"reference_dist": "true"})

    monkeypatch.setattr("modal_uq.experiments.selective.risk_coverage", fake_risk_coverage)

    curve = SelectivePrediction.compute_selective_curve(
        exp,
        pd.DataFrame({"uncertainty": uncertainty}),
        "uncertainty",
        metric_name="lr_true",
        steps=0,
    )

    assert np.array_equal(captured["y_true"], y_mode_true)
    assert not np.array_equal(captured["y_true"], y_test)
    assert np.array_equal(captured["y_pred"], y_pred)
    assert np.array_equal(captured["y_grid"], y_grid)
    assert curve["risk"].shape == (1,)



def test_likelihood_ratio_collapses_bma_est_dens_before_argmax_check():
    """Regression: LR should canonicalize BMA estimated densities before argmax checks.

    Construct est_dens as a stack [M, N, G] where at least one member's argmax disagrees
    with the ensemble mean argmax. LR must succeed by collapsing est_dens to the
    ensemble mean before validation.
    """
    y_grid = np.array([0.0, 1.0, 2.0])

    # y_true must be the argmax of true_dens (LR contract)
    y_true = np.array([0.0, 2.0])

    # Two members disagree on the mode; ensemble mean peaks at 1.0
    est_member0 = np.array([[0.6, 0.4, 0.0],
                            [0.6, 0.4, 0.0]])
    est_member1 = np.array([[0.0, 0.4, 0.6],
                            [0.0, 0.4, 0.6]])
    est_dens_stack = np.stack([est_member0, est_member1], axis=0)

    est_mean = est_dens_stack.mean(axis=0)
    y_mode_pred = y_grid[est_mean.argmax(axis=1)]
    assert np.array_equal(y_mode_pred, np.array([1.0, 1.0]))

    # Guard: the old member-wise argmax check would fail
    assert not all(np.all(y_mode_pred == y_grid[est_dens_stack[m].argmax(axis=1)])
                   for m in range(est_dens_stack.shape[0]))

    true_dens = np.array([[0.7, 0.2, 0.1],
                          [0.1, 0.2, 0.7]])

    lr = likelihood_ratio_measure(
        y_true,
        y_mode_pred,
        kwargs_dict={
            'true_dens': true_dens,
            'est_dens': est_dens_stack,
            'y_grid': y_grid,
            'reference_dist': 'true',
        },
    )

    assert np.isclose(lr, 3.5)
