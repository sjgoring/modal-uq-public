import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import numpy as np
import matplotlib
from unittest.mock import patch
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modal_uq.models.ensemble import Ensemble


class _FakeMember:
    def __init__(self, params):
        self._params = np.asarray(params, dtype=float)

    def get_params(self, X):
        return self._params


def _run_second_order_distribution_check(show_plot=False):
    print("[START] test_second_order_distribution_uses_member_parameter_slices")
    X = np.zeros((2, 1), dtype=float)
    theta_grid = np.array(
        [
            [0.0, 0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0, 3.5],
        ],
        dtype=float,
    )

    members = [
        _FakeMember([[0.7, 0.4], [1.0, 1.5], [2.0, 2.1]]),
        _FakeMember([[0.2, 0.5], [1.1, 1.6], [2.2, 2.3]]),
        _FakeMember([[0.1, 0.1], [0.9, 1.4], [1.8, 1.9]]),
    ]

    ensemble = Ensemble(
        base_model="condgmm",
        base_params={"n_components": 2},
        n_members=len(members),
        bootstrap=False,
        seed=0,
    )
    ensemble.members = members

    seen_samples = []

    def fake_gaussian_kde(samples):
        samples = np.asarray(samples, dtype=float)
        seen_samples.append(samples.copy())

        class _FakeKDE:
            def __call__(self, grid):
                grid = np.asarray(grid, dtype=float)
                # Return density proportional to grid size (last dimension)
                return np.linspace(0.25, 0.75, grid.shape[-1], dtype=float)

        return _FakeKDE()

    try:
        with patch("scipy.stats.gaussian_kde", fake_gaussian_kde):
            theta_dens_by_X, returned_grid = ensemble.get_second_order_distribution(X, theta_grid)

        assert returned_grid is theta_grid
        assert theta_dens_by_X.shape == (2, 4)

        expected_density = np.linspace(0.25, 0.75, 4, dtype=float)
        compare_dir = os.path.join("runs", "_ensemble_2nd_order")
        os.makedirs(compare_dir, exist_ok=True)
        compare_path = os.path.join(compare_dir, "test_second_order_distribution_compare.png")

        np.testing.assert_allclose(theta_dens_by_X[0], expected_density)
        np.testing.assert_allclose(theta_dens_by_X[1], expected_density)

        expected_first_sample = np.array(
            [
                [1.0, 1.1, 0.9],
                [2.0, 2.2, 1.8],
            ],
            dtype=float,
        )
        expected_second_sample = np.array(
            [
                [1.5, 1.6, 1.4],
                [2.1, 2.3, 1.9],
            ],
            dtype=float,
        )

        assert len(seen_samples) == 2
        np.testing.assert_allclose(seen_samples[0], expected_first_sample)
        np.testing.assert_allclose(seen_samples[1], expected_second_sample)

        # Create a finer grid for smooth 3D visualization
        # Expand by 4x in both axes while keeping same centering
        param0_grid = np.linspace(-2.25, 3.75, 20)
        param1_grid = np.linspace(-0.25, 5.75, 20)
        gx, gy = np.meshgrid(param0_grid, param1_grid, indexing="ij")
        
        fig = plt.figure(figsize=(14, 6))
        
        for sample_idx in range(2):
            ax = fig.add_subplot(1, 2, sample_idx + 1, projection="3d")
            
            # Select the correct expected sample for this subplot
            if sample_idx == 0:
                exp_params = expected_first_sample
            else:
                exp_params = expected_second_sample
            
            # Create synthetic smooth surface (bivariate Gaussian-like for visualization)
            # Center on the mean of expected samples for THIS sample
            center0 = np.mean(exp_params[0])
            center1 = np.mean(exp_params[1])
            Z = 0.5 * np.exp(-0.5 * ((gx - center0)**2 + (gy - center1)**2))
            
            surf = ax.plot_surface(gx, gy, Z, cmap="viridis", alpha=0.8, edgecolor="none")
            
            # Evaluate density at each expected parameter point
            z_expected = 0.5 * np.exp(-0.5 * ((exp_params[0] - center0)**2 + (exp_params[1] - center1)**2))
            ax.scatter(exp_params[0], exp_params[1], z_expected, color="red", s=100, marker="*", 
                      label="expected samples", zorder=10)
            
            ax.set_xlabel("parameter 0")
            ax.set_ylabel("parameter 1")
            ax.set_zlabel("density")
            ax.set_title(f"Joint 3D distribution (sample {sample_idx})")
            ax.legend()
            plt.colorbar(surf, ax=ax, label="density", shrink=0.5)
        
        fig.suptitle("Second-order distribution: 3D joint parameter space with expected samples")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(compare_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)
        print(f"[INFO] saved comparison plot to {compare_path}")
    except AssertionError as exc:
        print(f"[FAIL] test_second_order_distribution_uses_member_parameter_slices: {exc}")
        raise
    except Exception as exc:
        print(f"[ERROR] test_second_order_distribution_uses_member_parameter_slices: {exc}")
        raise
    else:
        print("[PASS] test_second_order_distribution_uses_member_parameter_slices")


def test_second_order_distribution_uses_member_parameter_slices():
    _run_second_order_distribution_check(show_plot=False)


if __name__ == "__main__":
    _run_second_order_distribution_check(show_plot=True)