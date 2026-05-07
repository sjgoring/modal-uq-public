"""Simple MPE dataset adapter for deep_moe.

Loads the precomputed `.npz` used by the modal_uq implementation and exposes
the minimal interface needed by the active-learning runner:

- `X_raw`, `y_raw` : raw arrays loaded from file
- `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test` : deterministic splits
- `y_grid` : canonical evaluation grid
- `gt(X, y)` and `gt_dens(X, y, y_grid=None)` helpers using KDE

The class is intentionally small and local to `deep_moe` to avoid pulling in
the full `modal_uq` registry or dependencies.
"""
from typing import Optional, Tuple
import numpy as np
from scipy.stats import gaussian_kde
import scipy.integrate as integrate
from sklearn.model_selection import train_test_split


class MpeDataset:
    def __init__(
        self,
        data_path: str = "data/raw/mpe.npz",
        split_seed: Optional[int] = None,
        y_grid_size: int = 1000,
        y_pad: float = 1.0,
    ):
        self.data_path = data_path
        self.split_seed = split_seed
        self.y_grid_size = y_grid_size
        self.y_pad = y_pad

        # raw arrays
        self.X_raw: Optional[np.ndarray] = None
        self.y_raw: Optional[np.ndarray] = None

        # public splits
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        # canonical y grid and cached densities
        self.y_grid: Optional[np.ndarray] = None
        self._y_densities: Optional[np.ndarray] = None

        self._load_and_prepare()

    def _load_and_prepare(self):
        data = np.load(self.data_path)
        X = data["X"]
        y = data["y"]

        self.X_raw = X
        self.y_raw = y

        # canonical grid
        self.y_min, self.y_max = float(y.min()), float(y.max())
        self.y_grid = self._make_y_grid()

        # precompute KDE densities on canonical grid for each row
        y_densities = []
        for i in range(X.shape[0]):
            samples = y[i]
            kde = gaussian_kde(samples)
            dens = kde(self.y_grid)
            dens = dens / integrate.trapezoid(dens, self.y_grid)
            y_densities.append(dens)
        self._y_densities = np.vstack(y_densities)

        # deterministic train/val/test split: 60/20/20
        rs = int(self.split_seed) if self.split_seed is not None else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rs
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=rs
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def _make_y_grid(self) -> np.ndarray:
        y_span = self.y_max - self.y_min
        pad = self.y_pad * (y_span + 1e-6)
        return np.linspace(self.y_min - pad, self.y_max + pad, self.y_grid_size)

    def gt(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return KDE mode for each row in X using the provided per-row samples y.

        The caller should supply the per-row sample matrix `y` with the same
        row order as X (this mirrors the modal_uq semantics where the raw file
        contains a collection of sample vectors per row).
        """
        modes = np.empty(X.shape[0])
        for idx in range(X.shape[0]):
            samples = y[idx]
            kde = gaussian_kde(samples)
            modes[idx] = self.y_grid[np.argmax(kde(self.y_grid))]
        return modes

    def gt_dens(self, X: np.ndarray, y: np.ndarray, y_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """Return KDE density on `y_grid` (or canonical grid) for each row in X."""
        eval_grid = y_grid if y_grid is not None else self.y_grid
        y_densities = []
        for idx in range(X.shape[0]):
            samples = y[idx]
            kde = gaussian_kde(samples)
            dens = kde(eval_grid)
            dens = dens / integrate.trapezoid(dens, eval_grid)
            y_densities.append(dens)
        return np.vstack(y_densities)

    def sample_x(self) -> np.ndarray:
        raise NotImplementedError("Sampling helper not implemented for MpeDataset.")


def load_mpe_dataset(data_path: str = "data/raw/mpe.npz", split_seed: Optional[int] = None) -> MpeDataset:
    """Convenience loader returning an initialized `MpeDataset`."""
    return MpeDataset(data_path=data_path, split_seed=split_seed)
