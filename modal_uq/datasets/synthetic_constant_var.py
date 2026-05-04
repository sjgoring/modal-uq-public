"""
Synthetic dataset generator taken from basic-test-constant-var-diff-conc.py

Provides a simple 1D conditional mixture dataset where the mixture parameters
change over x (split into thirds). The class mirrors the style of
`SyntheticMultiModalConditionalDataset` and exposes `sample_x` and `get_data`.
"""
from typing import Optional, Tuple
import numpy as np
import scipy.integrate as integrate
import pandas as pd
from ..registry import register

@register('dataset','synthetic_constant_var')
class SyntheticConstantVarDataset:
    def __init__(
        self,
        n_samples: int = 1000,
        x_min: float = -10.0,
        x_max: float = 10.0,
        y_min: float = -10.0,
        y_max: float = 10.0,
        y_grid_size: int = 1000,
        pi_1: float = 0.6,
        seed: Optional[int] = 42,
        x_sampler: str = 'grid',
    ):
        self.n_samples = n_samples
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.y_grid_size = y_grid_size
        self.pi_1 = pi_1
        self.pi_2 = 1.0 - pi_1
        self.seed = seed
        self.x_sampler = x_sampler
        self.rng = np.random.default_rng(seed)
        self.needs_pseudo_ground_truth = False

    def sample_x(self) -> np.ndarray:
        if self.x_sampler == 'grid':
            xs = np.linspace(self.x_min, self.x_max, self.n_samples)
            return xs.reshape(-1, 1)
        elif self.x_sampler == 'uniform':
            xs = self.rng.uniform(self.x_min, self.x_max, size=(self.n_samples,))
            return xs.reshape(-1, 1)
        else:
            raise ValueError('Unknown x_sampler')

    def get_feature_grid(self, X: np.ndarray, x_values: Optional[Tuple] = None):
        """Return feature grid for slider/plot helpers.

        - If X has 1 feature: returns array of unique sorted x values or provided x_values
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

    def _mixture_params(self, xs: np.ndarray):
        """Return per-x mixture parameters: pi, mu1, mu2, sigma1, sigma2.
        xs must be shape (n_samples,1) or (n_samples,).
        """
        xs1 = xs.flatten()
        # low component sigma region
        sigma_c = 0.1
        sigma_1_l = np.ones_like(xs1) * sigma_c
        sigma_2_l = sigma_1_l
        mu_1_l = np.ones_like(xs1) * 1.0
        mu_2_l = np.ones_like(xs1) * 0.0

        # override global mixture sigma to match original script behaviour
        mu_l = mu_1_l * self.pi_1 + mu_2_l * self.pi_2
        mixture_sigma = np.sqrt(sigma_c ** 2 + self.pi_1 * (mu_1_l - mu_l) ** 2 + self.pi_2 * (mu_2_l - mu_l) ** 2)

        # high component sigma region
        mu_1_h = np.ones_like(xs1) * 0.8 #note if pi_1 is changed, these mus should be changed to maintain the same mixture mean.
        mu_2_h = np.ones_like(xs1) * 0.3
        mu_h = mu_1_h * self.pi_1 + mu_2_h * self.pi_2
        sigma_c_h = np.sqrt(np.maximum(0.0, mixture_sigma ** 2 - self.pi_1 * (mu_1_h - mu_h) ** 2 - self.pi_2 * (mu_2_h - mu_h) ** 2))
        sigma_1_h = np.ones_like(xs1) * sigma_c_h
        sigma_2_h = sigma_1_h

        # split x into thirds: first & last third -> low sigma, middle third -> high sigma
        x_th1 = self.x_min + (self.x_max - self.x_min) / 3.0
        x_th2 = self.x_min + 2.0 * (self.x_max - self.x_min) / 3.0
        low_mask = np.logical_or(xs1 < x_th1, xs1 > x_th2)

        mu_1 = np.where(low_mask, mu_1_l, mu_1_h)
        mu_2 = np.where(low_mask, mu_2_l, mu_2_h)
        sigma_1 = np.where(low_mask, sigma_1_l, sigma_1_h)
        sigma_2 = np.where(low_mask, sigma_2_l, sigma_2_h)

        pi1 = np.ones_like(xs1) * self.pi_1
        pi2 = np.ones_like(xs1) * self.pi_2
        return pi1, pi2, mu_1, mu_2, sigma_1, sigma_2

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate dataset using same logic as the original test script.

        Returns X, y, global_mode (array), mode_ids, and y_densities (list of densities on y_grid)
        """
        X = self.sample_x() # shape (n_samples, 1)
        y_grid = np.linspace(self.y_min, self.y_max, self.y_grid_size) # shape (y_grid_size,)

        pi1, pi2, mu_1, mu_2, sigma_1, sigma_2 = self._mixture_params(X)

        y_samples = []
        y_densities = []
        mode_ids = np.zeros(X.shape[0], dtype=int)

        for idx in range(X.shape[0]):
            # true mixture density on grid
            dens1 = (1.0 / (np.sqrt(2 * np.pi) * sigma_1[idx])) * np.exp(-0.5 * ((y_grid - mu_1[idx]) / sigma_1[idx]) ** 2)
            dens2 = (1.0 / (np.sqrt(2 * np.pi) * sigma_2[idx])) * np.exp(-0.5 * ((y_grid - mu_2[idx]) / sigma_2[idx]) ** 2)
            y_dens_raw = dens1 * pi1[idx] + dens2 * pi2[idx]
            # normalize to form proper density
            y_dens = y_dens_raw / integrate.trapezoid(y_dens_raw, y_grid)
            y_densities.append(y_dens)

            # draw a sample by discrete sampling from grid proportional to density
            probs = y_dens_raw / np.sum(y_dens_raw)
            y_choice = self.rng.choice(y_grid, size=1, p=probs)[0]
            y_samples.append(y_choice)

            # assign a mode id by sampling component according to pi
            comp = self.rng.choice([0, 1], p=[pi1[idx], pi2[idx]])
            mode_ids[idx] = comp

        y = np.array(y_samples).reshape(-1,)
        global_mode = np.array([mu_1.mean(), mu_2.mean()])

        return X, y, global_mode, mode_ids, np.vstack(y_densities)

    def to_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(X.shape[1])])
        df['y'] = y
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

        # feature grid helper
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
            l_comp = [ax.plot([], [], '--', label=f'Comp {k+1}')[0] for k in range(n_modes)]
        else:
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

            # sampled y markers
            y_at_x = get_data_y_at_x(x_sel)
            if y_at_x.size > 0:
                l_sampled_y.set_offsets(np.c_[y_at_x, np.zeros_like(y_at_x)])
            else:
                l_sampled_y.set_offsets(np.empty((0, 2)))

            # y-limits
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

        ## [24/03] Todo: Plotting needs to obtain ground truth distribution of y|x from the dataset. In addtiion the learnt density needs to be plotted correctly.

    def gt(self, X):
        """Return ground-truth conditional mode (component mean with highest weight) for each row in X."""
        n = X.shape[0]
        pi1, pi2, mu_1, mu_2, _, _= self._mixture_params(X)
        pi = np.stack([pi1, pi2], axis=1)
        mu_vals = np.stack([mu_1, mu_2], axis=1)
        # pick mode with largest pi per sample
        mode_idx = np.argmax(pi, axis=1)
        y_mode_true = np.array([mu_vals[i, mode_idx[i]] for i in range(n)])
        return y_mode_true
    
    def gt_dens(self, X, y_grid):
        """Return ground-truth conditional density on y_grid for each row in X."""
        n = X.shape[0]
        pi1, pi2, mu_1, mu_2, sigma_1, sigma_2 = self._mixture_params(X)
        
        # y_grid = np.linspace(self.y_min, self.y_max, self.y_grid_size)
        y_densities = []
        for i in range(n):
            dens1 = (1.0 / (np.sqrt(2 * np.pi) * sigma_1[i])) * np.exp(-0.5 * ((y_grid - mu_1[i]) / sigma_1[i]) ** 2)
            dens2 = (1.0 / (np.sqrt(2 * np.pi) * sigma_2[i])) * np.exp(-0.5 * ((y_grid - mu_2[i]) / sigma_2[i]) ** 2)
            y_dens_raw = dens1 * pi1[i] + dens2 * pi2[i]
            y_dens = y_dens_raw / integrate.trapezoid(y_dens_raw, y_grid)
            y_densities.append(y_dens)
        return np.vstack(y_densities)