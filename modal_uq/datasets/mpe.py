"""
MPE (Multi-Particle Environment) dataset.

Loads pre-simulated trajectories from a PettingZoo MPE environment where a
good agent avoids static adversaries to reach a goal landmark. X (n, 14) is
the initial world state; y (n, 50) contains the maximum lateral displacement
(w.r.t. the agent-goal unit vector) across 50 independent trajectories.

The `gt` method fits a KDE to the 50 y samples per row and returns the mode
of the KDE as the ground-truth conditional mode of y|x.
"""
from typing import Optional, Tuple
import numpy as np
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
from sklearn.model_selection import train_test_split
from ..registry import register
from ..utils.seed import resolve_seed


@register('dataset', 'mpe')
class MpeDataset:
    def __init__(
        self,
        n_samples: int = 1000,
        data_path: str = "data/raw/mpe.npz",
        seed: Optional[int] = 42,
        split_seed: Optional[int] = None,
    ):
        self.n_samples = n_samples
        self.data_path = data_path
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.needs_pseudo_ground_truth = False

        # Populate X_raw / y_raw and train/val/test splits
        self.get_data()
        self._setup_train_test_split(split_seed)

    def _setup_train_test_split(self, split_seed: Optional[int] = None):
        """Initialize train/test/val split from generated data for active_learning compatibility.

        Generates the full dataset and splits it into X_train, y_train, X_test, y_test
        (and X_val, y_val for DatasetSpec compatibility).
        """
        split_seed = resolve_seed(split_seed)
        X, y = self.X_raw, self.y_raw

        # Split: 60% train, 20% val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=split_seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=split_seed  # 0.25 * 0.8 = 0.2
        )

        # Store splits as attributes for active_learning and other experiments
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset from the npz file.

        Returns X, y, global_mode (array), mode_ids, y_densities (n, y_grid_size), and y_grid.
        """
        data = np.load(self.data_path)
        X = data['X']   # (n, 14)
        y = data['y']   # (n, 50)

        self.X_raw = X
        self.y_raw = y

        y_min, y_max = float(y.min()), float(y.max())
        y_grid = np.linspace(y_min, y_max, 200)

        y_densities = []
        modes = np.empty(X.shape[0])

        for i in range(X.shape[0]):
            samples = y[i]      # 50 trajectory samples
            kde = gaussian_kde(samples)
            dens = kde(y_grid)
            dens = dens / integrate.trapezoid(dens, y_grid)
            y_densities.append(dens)
            modes[i] = y_grid[np.argmax(dens)]

        # mode_ids: placeholder zeros (no discrete mixture components for MPE)
        mode_ids = np.zeros(X.shape[0], dtype=int)
        global_mode = np.array([modes.mean()])

        return X, y, global_mode, mode_ids, np.vstack(y_densities), y_grid

    def gt(self, X: np.ndarray) -> np.ndarray:
        """Return the KDE mode of y|x for each row in X.

        For each query row, the nearest stored row (by L1 distance) is located,
        its 50 trajectory samples are used to fit a KDE, and the argmax of the
        KDE on a fine grid is returned as the ground-truth conditional mode.
        """
        modes = np.empty(X.shape[0])
        for i, x in enumerate(X):
            idx = int(np.argmin(np.abs(self.X_raw - x).sum(axis=1)))
            samples = self.y_raw[idx]       # (50,)
            kde = gaussian_kde(samples)
            lo, hi = samples.min(), samples.max()
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            grid = np.linspace(lo - pad, hi + pad, 500)
            modes[i] = grid[np.argmax(kde(grid))]
        return modes

    def gt_dens(self, X: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
        """Return the KDE density on y_grid for each row in X."""
        y_densities = []
        for x in X:
            idx = int(np.argmin(np.abs(self.X_raw - x).sum(axis=1)))
            samples = self.y_raw[idx]
            kde = gaussian_kde(samples)
            dens = kde(y_grid)
            dens = dens / integrate.trapezoid(dens, y_grid)
            y_densities.append(dens)
        return np.vstack(y_densities)

    def sample_x(self) -> np.ndarray:
        raise NotImplementedError
