import numpy as np
import pytest
from scipy import integrate

from modal_uq.metrics.mode_errors import (
    likelihood_ratio_measure,
    modal_absolute_error,
    modal_coverage_measure,
    modal_squared_error,
)


def _kwargs(true_dens, est_dens, y_grid, reference_dist):
    return {
        "true_dens": true_dens,
        "est_dens": est_dens,
        "y_grid": y_grid,
        "reference_dist": reference_dist,
    }


def _trapz_normalize(dens, y_grid):
    area = integrate.trapezoid(dens, y_grid, axis=-1)
    return dens / area[..., None]


def test_modal_absolute_and_squared_error_basic():
    y_true = np.array([0.0, 2.0, -1.0])
    y_mode_pred = np.array([1.0, 1.0, -2.0])

    assert modal_absolute_error(y_true, y_mode_pred, {}) == pytest.approx(1.0)
    assert modal_squared_error(y_true, y_mode_pred, {}) == pytest.approx(1.0)


def test_likelihood_ratio_measure_true_reference_basic():
    y_grid = np.array([0.0, 1.0, 2.0])
    y_true = np.array([0.0])
    y_mode_pred = np.array([1.0])

    true_dens = np.array([[0.2, 0.5, 0.3]])
    est_dens = np.array([[0.3, 0.4, 0.3]])

    result = likelihood_ratio_measure(
        y_true,
        y_mode_pred,
        _kwargs(true_dens, est_dens, y_grid, "true"),
    )

    expected = 0.5 / 0.2
    assert result == pytest.approx(expected)


def test_likelihood_ratio_measure_est_reference_basic():
    y_grid = np.array([0.0, 1.0, 2.0])
    y_true = np.array([0.0])
    y_mode_pred = np.array([2.0])

    true_dens = np.array([[0.2, 0.5, 0.3]])
    est_dens = np.array([[0.3, 0.1, 0.6]])

    result = likelihood_ratio_measure(
        y_true,
        y_mode_pred,
        _kwargs(true_dens, est_dens, y_grid, "est"),
    )

    expected = 0.3 / 0.6
    assert result == pytest.approx(expected)


def test_likelihood_ratio_measure_averages_over_batch():
    y_grid = np.array([0.0, 1.0, 2.0])
    y_true = np.array([0.0, 2.0])
    y_mode_pred = np.array([1.0, 1.0])

    true_dens = np.array([
        [0.2, 0.5, 0.3],
        [0.1, 0.6, 0.3],
    ])
    est_dens = true_dens.copy()

    result = likelihood_ratio_measure(
        y_true,
        y_mode_pred,
        _kwargs(true_dens, est_dens, y_grid, "true"),
    )

    expected = np.mean([0.5 / 0.2, 0.6 / 0.3])
    assert result == pytest.approx(expected)


def test_modal_coverage_measure_true_reference_basic():
    y_grid = np.array([0.0, 1.0, 2.0])
    y_true = np.array([1.0])
    y_mode_pred = np.array([0.0])

    true_dens = np.array([[0.1, 0.6, 0.3]])
    est_dens = np.array([[0.2, 0.5, 0.3]])

    result = modal_coverage_measure(
        y_true,
        y_mode_pred,
        _kwargs(true_dens, est_dens, y_grid, "true"),
    )

    norm_true = _trapz_normalize(true_dens, y_grid)[0]
    threshold = norm_true[np.abs(y_grid - y_mode_pred[0]).argmin()]
    indicator = (norm_true > threshold).astype(float)
    expected = integrate.trapezoid(norm_true * indicator, y_grid)
    assert result == pytest.approx(expected)


def test_modal_coverage_measure_est_reference_nonuniform_grid_uses_trapezoidal_mass():
    y_grid = np.array([0.0, 1.0, 3.0])
    y_true = np.array([0.0])
    y_mode_pred = np.array([1.0])

    true_dens = np.array([[0.1, 0.6, 0.3]])
    est_dens = np.array([[0.2, 0.6, 0.2]])

    result = modal_coverage_measure(
        y_true,
        y_mode_pred,
        _kwargs(true_dens, est_dens, y_grid, "est"),
    )

    norm_est = _trapz_normalize(est_dens, y_grid)[0]
    threshold = norm_est[np.abs(y_grid - y_true[0]).argmin()]
    indicator = (norm_est > threshold).astype(float)
    expected = integrate.trapezoid(norm_est * indicator, y_grid)

    assert result == pytest.approx(expected)


def test_likelihood_ratio_measure_averages_over_member_stack():
    y_grid = np.array([0.0, 1.0, 2.0])
    y_true = np.array([0.0, 2.0])
    y_mode_pred = np.array([1.0, 1.0])

    true_dens = np.array([
        [0.2, 0.5, 0.3],
        [0.1, 0.6, 0.3],
    ])
    est_dens = np.stack([
        true_dens,
        np.array([
            [0.3, 0.4, 0.3],
            [0.2, 0.3, 0.5],
        ]),
    ], axis=0)

    result = likelihood_ratio_measure(
        y_true,
        y_mode_pred,
        _kwargs(true_dens, est_dens, y_grid, "true"),
    )

    expected = np.mean([
        likelihood_ratio_measure(y_true, y_mode_pred, _kwargs(true_dens, est_dens[0], y_grid, "true")),
        likelihood_ratio_measure(y_true, y_mode_pred, _kwargs(true_dens, est_dens[1], y_grid, "true")),
    ])
    assert result == pytest.approx(expected)


def test_modal_coverage_measure_averages_over_member_stack():
    y_grid = np.array([0.0, 1.0, 2.0])
    y_true = np.array([1.0, 0.0])
    y_mode_pred = np.array([0.0, 2.0])

    true_dens = np.array([
        [0.1, 0.6, 0.3],
        [0.2, 0.2, 0.6],
    ])
    est_dens = np.stack([
        true_dens,
        np.array([
            [0.3, 0.3, 0.4],
            [0.5, 0.2, 0.3],
        ]),
    ], axis=0)

    result = modal_coverage_measure(
        y_true,
        y_mode_pred,
        _kwargs(true_dens, est_dens, y_grid, "true"),
    )

    expected = np.mean([
        modal_coverage_measure(y_true, y_mode_pred, _kwargs(true_dens, est_dens[0], y_grid, "true")),
        modal_coverage_measure(y_true, y_mode_pred, _kwargs(true_dens, est_dens[1], y_grid, "true")),
    ])
    assert result == pytest.approx(expected)