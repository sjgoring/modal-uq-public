import numpy as np

from modal_uq.metrics.mode_errors import likelihood_ratio_measure
from modal_uq.metrics.selective import risk_coverage


def _kwargs(true_dens, est_dens, y_grid, reference_dist):
    return {
        "true_dens": true_dens,
        "est_dens": est_dens,
        "y_grid": y_grid,
        "reference_dist": reference_dist,
    }


def test_risk_coverage_supports_bma_density_stack():
    y_grid = np.array([0.0, 1.0, 2.0])
    y_true = np.array([0.0, 2.0])
    y_pred = np.array([1.0, 1.0])
    uncertainty = np.array([0.8, 0.2])

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

    coverages, risks = risk_coverage(
        y_true,
        y_pred,
        uncertainty,
        likelihood_ratio_measure,
        {"reference_dist": "true"},
        steps=0,
        true_dens=true_dens,
        est_dens=est_dens,
        y_grid=y_grid,
    )

    expected = np.mean([
        likelihood_ratio_measure(y_true, y_pred, _kwargs(true_dens, est_dens[0], y_grid, "true")),
        likelihood_ratio_measure(y_true, y_pred, _kwargs(true_dens, est_dens[1], y_grid, "true")),
    ])

    assert coverages.tolist() == [1.0]
    assert risks.shape == (1,)
    assert np.isfinite(risks[0])
    assert np.isclose(risks[0], expected)