# modal_uq/datasets/moons_synthetic.py
from typing import Optional, Tuple, Sequence, Callable
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from sklearn.datasets import make_moons


class MoonsSyntheticDataset:
    """
    Small wrapper around sklearn.datasets.make_moons.

    Parameters
    - n_samples: total samples to generate
    - noise: gaussian noise passed to make_moons
    - random_state: RNG seed
    - return_1d: if True, features returned as (n_samples,1) using X[:,0]
                 (useful for 1D -> 1D regression examples). If False, returns 2D X.
    - target: 'y' to use continuous second coordinate as regression target,
              'label' to return integer class labels.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        noise: float = 0.1,
        random_state: Optional[int] = None,
        return_1d: bool = True,
        target: str = "y",
    ):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.return_1d = return_1d
        if target not in ("y", "label"):
            raise ValueError("target must be 'y' or 'label'")
        self.target = target

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        X, labels = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        if self.return_1d:
            X_out = X[:, 0].reshape(-1, 1)
        else:
            X_out = X
        if self.target == "y":
            # use second coordinate as continuous target (sensible regression target)
            y_out = X[:, 1].copy()
        else:
            y_out = labels.copy()
        return X_out, y_out

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, y, global_mode, mode_ids)

        - X: (n_samples, n_features)
        - y: (n_samples,)
        - global_mode: centroid of the largest class (2D coordinates)
        - mode_ids: integer class labels (0/1) from make_moons
        """
        X_full, labels = make_moons(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        # construct outputs
        X_out = X_full[:, 0].reshape(-1, 1) if self.return_1d else X_full
        y_out = X_full[:, 1].copy() if self.target == "y" else labels.copy()
        mode_ids = labels.copy()
        # global_mode: centroid of majority class (in full 2D space)
        counts = np.bincount(labels)
        maj = int(np.argmax(counts))
        global_mode = X_full[labels == maj].mean(axis=0)
        return X_out, y_out, global_mode, mode_ids

    def get_feature_grid(self, X: np.ndarray, x_values: Optional[Sequence] = None):
        """
        For compatibility with other dataset wrappers:
        - If X has 1 feature: returns x_values array
        - If X has 2 features: returns (x1_values, x2_values, grid)
        """
        n_features = X.shape[1]
        if x_values is None:
            if n_features == 1:
                x_values = np.unique(np.sort(X[:, 0]))
                return x_values
            elif n_features == 2:
                x1_values = np.unique(np.sort(X[:, 0]))
                x2_values = np.unique(np.sort(X[:, 1]))
                grid = np.array(np.meshgrid(x1_values, x2_values)).reshape(2, -1).T
                return x1_values, x2_values, grid
        else:
            if n_features == 1:
                return np.array(x_values)
            elif n_features == 2:
                x1_values = np.array([v[0] for v in x_values])
                x2_values = np.array([v[1] for v in x_values])
                grid = np.array(np.meshgrid(x1_values, x2_values)).reshape(2, -1).T
                return x1_values, x2_values, grid
        raise ValueError("Grid generation only supported for 1D or 2D feature space.")

    def to_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        if X.ndim == 1 or X.shape[1] == 1:
            df = pd.DataFrame({"x1": X.ravel(), "y": y})
        else:
            cols = {f"x{i+1}": X[:, i] for i in range(X.shape[1])}
            df = pd.DataFrame(cols)
            df["y"] = y
        return df
    
    def plot_conditional_y_given_x(self, X, y, pi_fn=None, mu_fn=None, sigma_fn=None, x_values=None, mode_ids=None, bins=100, predictive_density=None):
        """
        Interactive plot of the conditional distribution y|x for selected x.
        Works for 1D or 2D X. If `pi_fn`, `mu_fn`, `sigma_fn` are provided, plots
        the mixture components (normalised then weighted by pi). Otherwise falls
        back to showing empirical density (histogram/KDE-like) and per-mode samples
        if `mode_ids` is provided.
        Args:
            X: (n_samples, n_features)
            y: (n_samples,)
            pi_fn, mu_fn, sigma_fn: optional mixture parameter callables matching
                signatures used elsewhere in the repo.
            x_values: optional grid or values for slider (defaults to unique X)
            mode_ids: optional integer mode labels for samples
            bins: number of points in y grid
            predictive_density: optional predicted densities (one row per grid point)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        n_features = X.shape[1]
        if n_features not in (1, 2):
            raise ValueError("plot_conditional_y_given_x supports 1D or 2D X only.")

        # feature grid helper (same logic as other datasets)
        if x_values is None:
            if n_features == 1:
                x_values = np.unique(np.sort(X[:, 0]))
            else:
                x1_values = np.unique(np.sort(X[:, 0]))
                x2_values = np.unique(np.sort(X[:, 1]))
                grid = np.array(np.meshgrid(x1_values, x2_values)).reshape(2, -1).T
        else:
            if n_features == 1:
                x_values = np.array(x_values)
            else:
                x1_values = np.array([v[0] for v in x_values])
                x2_values = np.array([v[1] for v in x_values])
                grid = np.array(np.meshgrid(x1_values, x2_values)).reshape(2, -1).T

        # y grid
        y_min, y_max = float(np.min(y)), float(np.max(y))
        y_grid = np.linspace(y_min, y_max, bins)

        # prepare figure
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.subplots_adjust(bottom=0.3 if n_features == 2 else 0.2)
        l_density, = ax.plot([], [], label='Mixture / Empirical Density')
        l_comp = []
        l_sampled_y = ax.scatter([], [], marker='v', color='red', s=60, label='Sampled y|x')
        l_pred = None
        if predictive_density is not None:
            l_pred, = ax.plot([], [], color='black', lw=2, label='Predictive Density')

        ax.set_xlabel('y|x')
        ax.set_ylabel('Density')
        ax.set_title('Conditional y|x')
        ax.legend()

        # sliders
        axcolor = 'lightgoldenrodyellow'
        if n_features == 1:
            ax_x = plt.axes([0.15, 0.05, 0.7, 0.05], facecolor=axcolor)
            step = np.unique(np.diff(x_values)).min() if len(x_values) > 1 else 1.0
            slider = Slider(ax_x, 'x', np.min(x_values), np.max(x_values), valinit=x_values[0], valstep=step)
        else:
            ax_x1 = plt.axes([0.15, 0.10, 0.7, 0.05], facecolor=axcolor)
            ax_x2 = plt.axes([0.15, 0.05, 0.7, 0.05], facecolor=axcolor)
            slider1 = Slider(ax_x1, 'x1', np.min(x1_values), np.max(x1_values), valinit=x1_values[0], valstep=np.unique(np.diff(x1_values)).min() if len(x1_values)>1 else 1.0)
            slider2 = Slider(ax_x2, 'x2', np.min(x2_values), np.max(x2_values), valinit=x2_values[0], valstep=np.unique(np.diff(x2_values)).min() if len(x2_values)>1 else 1.0)

        # Precompute mixture params if provided
        has_mixture = (pi_fn is not None and mu_fn is not None and sigma_fn is not None)
        if has_mixture:
            if n_features == 1:
                pi_all = pi_fn(x_values.reshape(-1, 1))
                mu_all = mu_fn(x_values.reshape(-1, 1))
                sigma_all = sigma_fn(x_values.reshape(-1, 1))
                n_modes = mu_all.shape[1]
            else:
                pi_all = pi_fn(grid)
                mu_all = mu_fn(grid)
                sigma_all = sigma_fn(grid)
                n_modes = mu_all.shape[1]
            # create component line placeholders
            l_comp = [ax.plot([], [], '--', label=f'Comp {k+1}')[0] for k in range(n_modes)]
        else:
            # if mode_ids present, set number of modes for plotting markers
            if mode_ids is not None:
                n_modes = int(np.max(mode_ids) + 1)
            else:
                n_modes = 0

        def mixture_density(y_grid, mu, sigma, pi):
            dens = np.zeros_like(y_grid)
            for k in range(len(pi)):
                dens += pi[k] * (1.0 / (np.sqrt(2.0 * np.pi) * sigma[k])) * np.exp(-0.5 * ((y_grid - mu[k]) / sigma[k]) ** 2)
            return dens

        def get_data_y_at_x(x_sel, tol=1e-8):
            if n_features == 1:
                return y[np.abs(X[:, 0] - x_sel) < tol]
            else:
                return y[(np.abs(X[:, 0] - x_sel[0]) < tol) & (np.abs(X[:, 1] - x_sel[1]) < tol)]

        def update(val=None):
            # select parameters for chosen x
            if n_features == 1:
                x_sel = slider.val
                ix = np.argmin(np.abs(x_values - x_sel))
                x_sel = x_values[ix]
                if has_mixture:
                    mu = mu_all[ix]
                    sigma = sigma_all[ix]
                    pi = pi_all[ix]
            else:
                x1v = slider1.val
                x2v = slider2.val
                ix1 = np.argmin(np.abs(x1_values - x1v))
                ix2 = np.argmin(np.abs(x2_values - x2v))
                x_sel = (x1_values[ix1], x2_values[ix2])
                # find index in flattened grid
                grid_idx = np.where((grid[:, 0] == x_sel[0]) & (grid[:, 1] == x_sel[1]))[0]
                if len(grid_idx) == 0:
                    return
                idx = grid_idx[0]
                if has_mixture:
                    mu = mu_all[idx]
                    sigma = sigma_all[idx]
                    pi = pi_all[idx]

            comp_list = []
            if has_mixture:
                # normalize base gaussians then multiply by pi
                for k in range(n_modes):
                    base = (1.0 / (np.sqrt(2.0 * np.pi) * sigma[k])) * np.exp(-0.5 * ((y_grid - mu[k]) / sigma[k]) ** 2)
                    try:
                        base_int = float(integrate.trapezoid(base, y_grid))
                    except Exception:
                        base_int = 0.0
                    if base_int > 0:
                        base = base / base_int
                    comp = float(pi[k]) * base
                    comp_list.append(comp)
                    l_comp[k].set_data(y_grid, comp)
                    l_comp[k].set_label(f'Comp {k+1} (pi={pi[k]:.2f})')
                dens = np.sum(np.vstack(comp_list), axis=0)
                l_density.set_data(y_grid, dens)
            else:
                # empirical conditional: select near-samples and show histogram-based density
                y_at_x = get_data_y_at_x(x_sel)
                if y_at_x.size > 0:
                    hist, edges = np.histogram(y_at_x, bins=bins, density=True)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    l_density.set_data(centers, hist)
                else:
                    l_density.set_data([], [])

            # predictive density
            if predictive_density is not None:
                try:
                    if n_features == 1:
                        pred = np.asarray(predictive_density[ix])
                    else:
                        pred = np.asarray(predictive_density[idx])
                    pred_int = float(integrate.trapezoid(pred, y_grid))
                except Exception:
                    pred_int = 0.0
                    pred = None
                if pred is not None and pred_int > 0:
                    pred = pred / pred_int
                    if l_pred is None:
                        pass
                    else:
                        l_pred.set_data(y_grid, pred)

            # sampled y markers (nearest exact matches)
            y_at_x = get_data_y_at_x(x_sel)
            if y_at_x.size > 0:
                l_sampled_y.set_offsets(np.c_[y_at_x, np.zeros_like(y_at_x)])
            else:
                l_sampled_y.set_offsets(np.empty((0, 2)))

            # y-limits: include mixture/components/pred
            y_candidates = []
            try:
                y_candidates.append(float(np.nanmax(l_density.get_ydata())))
            except Exception:
                pass
            for cd in comp_list:
                try:
                    y_candidates.append(float(np.max(cd)))
                except Exception:
                    pass
            if predictive_density is not None and 'pred' in locals() and pred is not None:
                try:
                    y_candidates.append(float(np.max(pred)))
                except Exception:
                    pass
            y_top = max(y_candidates) if y_candidates else 1.0
            if y_top <= 0:
                y_top = 1.0
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(0.0, float(y_top * 1.1))
            try:
                ax.legend()
            except Exception:
                pass
            fig.canvas.draw_idle()

        # initial draw
        update()
        if n_features == 1:
            slider.on_changed(update)
        else:
            slider1.on_changed(update)
            slider2.on_changed(update)
        plt.show()