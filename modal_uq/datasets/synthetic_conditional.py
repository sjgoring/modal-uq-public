"""
Synthetic Multi-Modal Regression Data Generator (Conditional Sampling Version)

- Allows user to specify sampling method for X (uniform, normal, grid, or custom)
- For each sampled x, y is drawn from a conditional Gaussian mixture: f(y|x)
- Means, sigmas, and weights can be functions of x
- Full reproducibility via independent seeds
- Usable as a dataset module or standalone script (CSV + plots)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union, Callable, Dict
from ..registry import register

@register('dataset','synthetic_conditional')
class SyntheticMultiModalConditionalDataset:
    def get_feature_grid(self, X, x_values=None):
        """
        Generate the grid of feature values for slider operation in plot_conditional_y_given_x.
        Returns:
            For 1D: x_values (np.ndarray)
            For 2D: (x1_values, x2_values, grid)
        """
        n_features = X.shape[1]
        if x_values is None:
            if n_features == 1:
                x_values = np.unique(np.sort(X[:,0]))
                return x_values
            elif n_features == 2:
                x1_values = np.unique(np.sort(X[:,0]))
                x2_values = np.unique(np.sort(X[:,1]))
                grid = np.array(np.meshgrid(x1_values, x2_values)).reshape(2, -1).T
                return x1_values, x2_values, grid
        else:
            if n_features == 1:
                x_values = np.array(x_values)
                return x_values
            elif n_features == 2:
                x1_values = np.array([v[0] for v in x_values])
                x2_values = np.array([v[1] for v in x_values])
                grid = np.array(np.meshgrid(x1_values, x2_values)).reshape(2, -1).T
                return x1_values, x2_values, grid
            raise ValueError("Grid generation only supported for 1D or 2D feature space.")
    def plot_conditional_y_given_x(self, X, y, pi_fn, mu_fn, sigma_fn, x_values=None, mode_ids=None, bins=100, predictive_density=None):
        """
        Interactive plot of the conditional distribution y|x for a selected x value.
        Handles 1D and 2D feature spaces.
        Args:
            X: (n_samples, n_features) array of feature values
            y: (n_samples,) array of targets
            pi_fn, mu_fn, sigma_fn: functions as used in get_data
            x_values: Optional, list/array of x values to allow selection from (defaults to unique X)
            mode_ids: Optional, mode assignments for each sample
            bins: Number of bins for y axis in density plot
            predictive_density: Optional, array of predicted density values matching the feature grid (for overlay)
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import numpy as np
        n_features = X.shape[1]
        if n_features not in [1,2]:
            raise ValueError("This method is only for 1D or 2D feature space.")
        # Use new grid generation method
        if n_features == 1:
            x_values = self.get_feature_grid(X, x_values)
        else:
            x1_values, x2_values, grid = self.get_feature_grid(X, x_values)
        # y grid for density plot
        y_min, y_max = np.min(y), np.max(y)
        y_grid = np.linspace(y_min, y_max, bins)
        # Initial indices
        ix0, ix1 = 0, 0
        def mixture_density(y_grid, mu, sigma, pi):
            dens = np.zeros_like(y_grid)
            for k in range(len(pi)):
                dens += pi[k] * (1/(np.sqrt(2*np.pi)*sigma[k])) * np.exp(-0.5*((y_grid-mu[k])/sigma[k])**2)
            return dens
        def get_data_y_at_x(x_sel, tol=1e-8):
            if n_features == 1:
                return y[np.abs(X[:,0] - x_sel) < tol]
            elif n_features == 2:
                return y[(np.abs(X[:,0] - x_sel[0]) < tol) & (np.abs(X[:,1] - x_sel[1]) < tol)]
        # Set up plot
        fig, ax = plt.subplots(figsize=(8,5))
        plt.subplots_adjust(bottom=0.3 if n_features==2 else 0.2)
        l_density, = ax.plot([], [], label='Mixture Density y|x')
        l_comp = [ax.plot([], [], '--', label=f'Comp {k+1}')[0] for k in range(self.n_modes)]
        l_sampled_y = ax.scatter([], [], marker='v', color='red', s=80, label='Sampled y|x')
        l_pred_density = None
        if predictive_density is not None:
            l_pred_density = ax.plot([], [], color='black', lw=2, label='Predictive Density')[0]
        ax.set_xlabel('y|x')
        ax.set_ylabel('Density')
        ax.set_title('Conditional Distribution y|x')
        ax.legend()
        # Sliders
        axcolor = 'lightgoldenrodyellow'
        if n_features == 1:
            ax_x = plt.axes([0.15, 0.05, 0.7, 0.05], facecolor=axcolor)
            slider = Slider(ax_x, 'x', np.min(x_values), np.max(x_values), valinit=x_values[ix0], valstep=np.unique(np.diff(x_values)).min() if len(x_values) > 1 else 1)
        else:
            ax_x1 = plt.axes([0.15, 0.10, 0.7, 0.05], facecolor=axcolor)
            ax_x2 = plt.axes([0.15, 0.05, 0.7, 0.05], facecolor=axcolor)
            slider1 = Slider(ax_x1, 'x1', np.min(x1_values), np.max(x1_values), valinit=x1_values[ix0], valstep=np.unique(np.diff(x1_values)).min() if len(x1_values) > 1 else 1)
            slider2 = Slider(ax_x2, 'x2', np.min(x2_values), np.max(x2_values), valinit=x2_values[ix1], valstep=np.unique(np.diff(x2_values)).min() if len(x2_values) > 1 else 1)
        # Precompute mixture params for all possible x selections
        if n_features == 1:
            pi_all = pi_fn(x_values.reshape(-1,1), self.n_modes)
            mu_all = mu_fn(x_values.reshape(-1,1), self.mode_locs, np.arange(self.n_modes))
            sigma_all = sigma_fn(x_values.reshape(-1,1), self.mode_locs, np.arange(self.n_modes))
        else:
            pi_all = pi_fn(grid, self.n_modes)
            mu_all = mu_fn(grid, self.mode_locs, np.arange(self.n_modes))
            sigma_all = sigma_fn(grid, self.mode_locs, np.arange(self.n_modes))
        def update(val=None):
            if n_features == 1:
                x_sel = slider.val
                ix = np.argmin(np.abs(x_values - x_sel))
                x_sel = x_values[ix]
                mu = mu_all[ix]
                sigma = sigma_all[ix]
                pi = pi_all[ix]
                # Plot predictive density if provided
                if predictive_density is not None:
                    l_pred_density.set_data(y_grid, predictive_density[ix])
            else:
                x1_sel = slider1.val
                x2_sel = slider2.val
                ix1 = np.argmin(np.abs(x1_values - x1_sel))
                ix2 = np.argmin(np.abs(x2_values - x2_sel))
                x_sel = (x1_values[ix1], x2_values[ix2])
                # Find index in grid
                grid_idx = np.where((grid[:,0]==x_sel[0]) & (grid[:,1]==x_sel[1]))[0]
                if len(grid_idx) == 0:
                    return
                idx = grid_idx[0]
                mu = mu_all[idx]
                sigma = sigma_all[idx]
                pi = pi_all[idx]
                # Plot predictive density if provided
                if predictive_density is not None:
                    l_pred_density.set_data(y_grid, predictive_density[idx])
            dens = mixture_density(y_grid, mu, sigma, pi)
            l_density.set_data(y_grid, dens)
            for k in range(self.n_modes):
                comp_dens = pi[k] * (1/(np.sqrt(2*np.pi)*sigma[k])) * np.exp(-0.5*((y_grid-mu[k])/sigma[k])**2)
                l_comp[k].set_data(y_grid, comp_dens)
            y_at_x = get_data_y_at_x(x_sel, tol=1e-8)
            l_sampled_y.set_offsets(np.c_[y_at_x, np.zeros_like(y_at_x)])
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(0, np.max(dens)*1.1)
            fig.canvas.draw_idle()
        update()
        if n_features == 1:
            slider.on_changed(update)
        else:
            slider1.on_changed(update)
            slider2.on_changed(update)
        plt.show()

    def __init__(
        self,
        n_samples: int = 1000,
        n_modes: int = 3,
        n_features: int = 1,
        mode_locs: Optional[np.ndarray] = None,
        mode_scales: Optional[Union[float, Sequence[float]]] = 1.0,
        mode_weights: Optional[Sequence[float]] = None,
        component_type: str = 'gaussian',
        component_df: Optional[float] = None,
        x_sampler: Union[str, Callable] = 'uniform',
        x_sampler_params: Optional[Dict] = None,
        seed_master: Optional[int] = 42,
        seed_mode_assign: Optional[int] = None,
        seed_sample: Optional[int] = None,
        seed_noise: Optional[int] = None,
    ):
        """
        Parameters:
            n_samples: Number of data points to generate
            n_modes: Number of mixture components (modes)
            n_features: Number of input features (X dims)
            mode_locs: Array of shape (n_modes, n_features) for mode means. If None, randomised.
            mode_scales: Spread (std or scale) for each mode (float or list of floats)
            mode_weights: Mixture weights (length n_modes). If None, uniform.
            component_type: 'gaussian' or 'student-t'
            component_df: Degrees of freedom for t-distribution (if used)
            x_sampler: 'uniform', 'normal', 'grid', or a callable
            x_sampler_params: dict of parameters for the sampler
            seed_master: Master seed (optional, used to derive all other seeds if set)
        """
        self.n_samples = n_samples
        self.n_modes = n_modes
        self.n_features = n_features
        self.component_type = component_type
        self.component_df = component_df
        self.mode_scales = np.array(mode_scales if isinstance(mode_scales, (list, np.ndarray)) else [mode_scales]*n_modes)
        self.mode_weights = np.array(mode_weights) if mode_weights is not None else np.ones(n_modes) / n_modes
        self.seed_master = seed_master
        self.x_sampler = x_sampler
        self.x_sampler_params = x_sampler_params or {}
        # Derive seeds if only master is set
        self.seed_mode_assign = seed_mode_assign if seed_mode_assign is not None else (hash((seed_master, 'assign')) % 2**32 if seed_master is not None else None)
        self.seed_sample = seed_sample if seed_sample is not None else (hash((seed_master, 'sample')) % 2**32 if seed_master is not None else None)
        self.seed_noise = seed_noise if seed_noise is not None else (hash((seed_master, 'noise')) % 2**32 if seed_master is not None else None)
        # Generators
        self.rng_mode_assign = np.random.default_rng(self.seed_mode_assign)
        self.rng_sample = np.random.default_rng(self.seed_sample)
        self.rng_noise = np.random.default_rng(self.seed_noise)
        # Mode locations
        if mode_locs is not None:
            self.mode_locs = np.array(mode_locs)
        else:
            self.mode_locs = np.random.default_rng(self.seed_master).normal(loc=0, scale=5, size=(n_modes, n_features))
        # Ground truth handling for experiments
        self.needs_pseudo_ground_truth = False

    def sample_x(self) -> np.ndarray:
        if isinstance(self.x_sampler, str):
            params = self.x_sampler_params
            if self.x_sampler == 'uniform':
                low = params.get('low', -10)
                high = params.get('high', 10)
                return self.rng_sample.uniform(low, high, size=(self.n_samples, self.n_features))
            elif self.x_sampler == 'normal':
                mean = params.get('mean', 0)
                std = params.get('std', 1)
                return self.rng_sample.normal(mean, std, size=(self.n_samples, self.n_features))
            elif self.x_sampler == 'grid':
                # For 1D: grid points, for 2D: meshgrid flattened
                grid_points = params.get('grid_points', 100)
                low = params.get('low', -10)
                high = params.get('high', 10)
                if self.n_features == 1:
                    x = np.linspace(low, high, grid_points)
                    idx = self.rng_sample.choice(grid_points, size=self.n_samples, replace=True)
                    return x[idx].reshape(-1, 1)
                elif self.n_features == 2:
                    x1 = np.linspace(low, high, int(np.sqrt(grid_points)))
                    x2 = np.linspace(low, high, int(np.sqrt(grid_points)))
                    xx, yy = np.meshgrid(x1, x2)
                    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
                    idx = self.rng_sample.choice(grid.shape[0], size=self.n_samples, replace=True)
                    return grid[idx]
                else:
                    raise NotImplementedError("Grid sampling only implemented for 1D or 2D features.")
            else:
                raise ValueError(f"Unknown x_sampler: {self.x_sampler}")
        elif callable(self.x_sampler):
            return self.x_sampler(self.n_samples, self.n_features, **self.x_sampler_params)
        else:
            raise ValueError("x_sampler must be a string or callable.")

    def get_data(
        self,
        pi_fn: Optional[Callable] = None,
        mu_fn: Optional[Callable] = None,
        sigma_fn: Optional[Callable] = None,
        noise_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            X: (n_samples, n_features)
            y: (n_samples,)
            global_mode: (n_features,)
            mode_ids: (n_samples,)
        Args:
            pi_fn: Function mapping X to mixture weights (n_samples, n_modes)
            mu_fn: Function mapping X to mode means (n_samples, n_modes)
            sigma_fn: Function mapping X to mode stds/covariances (n_samples, n_modes) or (n_samples, n_modes, n_features, n_features)
            noise_fn: Optional function mapping X to noise std (n_samples,)
        """
        X = self.sample_x()
        n_modes = self.n_modes
        pi = pi_fn(X, n_modes) if pi_fn is not None else np.ones((self.n_samples, n_modes)) / n_modes
        mu = mu_fn(X, self.mode_locs, np.arange(n_modes)) if mu_fn is not None else np.tile(self.mode_locs[:, 0], (self.n_samples, 1))
        sigma = sigma_fn(X, self.mode_locs, np.arange(n_modes)) if sigma_fn is not None else np.tile(self.mode_scales, (self.n_samples, 1))
        y = np.zeros(self.n_samples)
        mode_ids = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            mode = self.rng_mode_assign.choice(n_modes, p=pi[i])
            mode_ids[i] = mode
            mu_i = mu[i, mode] if mu.ndim == 3 else mu[i, mode]
            sigma_i = sigma[i, mode] if sigma.ndim == 3 else sigma[i, mode]
            if self.component_type == 'gaussian':
                y[i] = self.rng_sample.normal(loc=mu_i, scale=sigma_i)
            elif self.component_type == 'student-t':
                df = self.component_df if self.component_df is not None else 3
                y[i] = mu_i + sigma_i * self.rng_sample.standard_t(df)
            else:
                raise ValueError(f"Unknown component_type: {self.component_type}")
            if noise_fn is not None:
                y[i] += self.rng_noise.normal(loc=0, scale=noise_fn(X[i:i+1])[0])
        global_mode = self.mode_locs[np.argmax(np.mean(pi, axis=0))]
        return X, y, global_mode, mode_ids

    def gt(self, X, mu_fn, pi_fn, sigma_fn):
        """
        Returns the ground truth conditional modes of y|x for samples X.
        """
        pi = pi_fn(x.reshape(1,-1), self.n_modes) if hasattr(self, 'pi_fn') else np.ones((1, self.n_modes)) / self.n_modes
        mu = mu_fn(x.reshape(1,-1), self.mode_locs, np.arange(self.n_modes)) if hasattr(self, 'mu_fn') else np.tile(self.mode_locs[:, 0], (1, 1))
        sigma = sigma_fn(x.reshape(1,-1), self.mode_locs, np.arange(self.n_modes)) if hasattr(self, 'sigma_fn') else np.tile(self.mode_scales, (1, 1))
        # For simplicity, we take the mode as the mean of the component with highest weight at x
        mode_idx = np.argmax(pi)
        return mu[0, mode_idx] if mu.ndim == 3 else mu[0, mode_idx]


    def to_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.n_features)])
        df['y'] = y
        return df

    def plot(self, X: np.ndarray, y: np.ndarray, mu_fn = None, sigma_fn = None, mode_ids: Optional[np.ndarray] = None, save_path: Optional[str] = None, plot3d: bool = True):
        if mu_fn is None or sigma_fn is None:
            try:
                mu_fn_plot = self.mu_fn
                sigma_fn_plot = self.sigma_fn
            except AttributeError:
                # fallback: use global mu_fn and sigma_fn
                from inspect import currentframe, getouterframes
                outer = getouterframes(currentframe())[1].frame
                mu_fn_plot = outer.f_globals.get('mu_fn')
                sigma_fn_plot = outer.f_globals.get('sigma_fn')
        else:
            mu_fn_plot = mu_fn
            sigma_fn_plot = sigma_fn
                    
        if self.n_features == 1:
            plt.figure(figsize=(8, 5))
            plt.scatter(X[:, 0], y, c=mode_ids if mode_ids is not None else 'b', cmap='tab10', alpha=0.5, label='Samples')
            # Sort X for smooth plotting
            X_sorted = np.sort(X, axis=0)
            # Use the same mu_fn and sigma_fn as used in get_data
            # Note: assumes mu_fn and sigma_fn are available in scope
            # If not, user should pass them as arguments or set as attributes
            # try:
            #     mu_fn_plot = self.mu_fn
            #     sigma_fn_plot = self.sigma_fn
            # except AttributeError:
            #     # fallback: use global mu_fn and sigma_fn
            #     from inspect import currentframe, getouterframes
            #     outer = getouterframes(currentframe())[1].frame
            #     mu_fn_plot = outer.f_globals.get('mu_fn')
            #     sigma_fn_plot = outer.f_globals.get('sigma_fn')
            # Compute means and sigmas for sorted X
            mu_vals = mu_fn_plot(X_sorted, self.mode_locs, np.arange(self.n_modes))
            sigma_vals = sigma_fn_plot(X_sorted, self.mode_locs, np.arange(self.n_modes))
            for k in range(self.n_modes):
                # Plot mean
                plt.plot(X_sorted[:,0], mu_vals[:,k], color=f'C{k}', label=f'Component {k+1} Mean')
                # Plot shaded area for sigma
                plt.fill_between(
                    X_sorted[:,0],
                    mu_vals[:,k] - sigma_vals[:,k],
                    mu_vals[:,k] + sigma_vals[:,k],
                    color=f'C{k}', alpha=0.15, label=f'Component {k+1} ±1σ'
                )
            plt.xlabel('x1')
            plt.ylabel('y')
            plt.title('Synthetic Multi-Modal Data (1D)')
            plt.legend()
            if save_path:
                plt.savefig(save_path)
            plt.show()
        elif self.n_features == 2:
            if plot3d:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(X[:, 0], X[:, 1], y, c=mode_ids if mode_ids is not None else y, cmap='tab10' if mode_ids is not None else 'viridis', alpha=0.6)
                # Plot means and sigma surfaces for each mode
                # from inspect import currentframe, getouterframes
                # outer = getouterframes(currentframe())[1].frame
                # mu_fn_plot = mu_fn
                # sigma_fn_plot = sigma_fn
                # Create a grid for plotting surfaces
                grid_points = 10
                x1 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), grid_points)
                x2 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), grid_points)
                xx, yy = np.meshgrid(x1, x2)
                grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
                mu_vals_grid = mu_fn_plot(grid, self.mode_locs, np.arange(self.n_modes))
                sigma_vals_grid = sigma_fn_plot(grid, self.mode_locs, np.arange(self.n_modes))
                for k in range(self.n_modes):
                    # Plot mean surface
                    ax.plot_surface(
                        xx, yy, mu_vals_grid[:,k].reshape(xx.shape),
                        color=f'C{k}', alpha=0.3, linewidth=0, antialiased=False, label=f'Component {k+1} Mean'
                    )
                    # Plot upper and lower sigma surfaces
                    ax.plot_surface(
                        xx, yy, (mu_vals_grid[:,k] + sigma_vals_grid[:,k]).reshape(xx.shape),
                        color=f'C{k}', alpha=0.15, linewidth=0, antialiased=False
                    )
                    ax.plot_surface(
                        xx, yy, (mu_vals_grid[:,k] - sigma_vals_grid[:,k]).reshape(xx.shape),
                        color=f'C{k}', alpha=0.15, linewidth=0, antialiased=False
                    )
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_zlabel('y')
                ax.set_title('3D Synthetic Multi-Modal Data (2D X)')
                if mode_ids is not None:
                    legend1 = ax.legend(*scatter.legend_elements(), title="Modes")
                    ax.add_artist(legend1)
                if save_path:
                    plt.savefig(save_path)
                plt.show()
            else:
                plt.figure(figsize=(8, 6))
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5)
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.title('Input Space (colored by y)')
                if save_path:
                    plt.savefig(save_path)
                plt.show()
        else:
            print("Plotting only supported for 1D or 2D X.")

    # Example functions for conditional mixture
    def test_sigma_fn(self, X, mode_locs, k):
        # Example: constant sigma for all modes
        # return np.ones((X.shape[0], len(k))) * 0.5

        # Example: 0 sigma for all modes
        # return np.zeros((X.shape[0], len(k))) * 0.5

        # Example: varying sigma for each modes
        out = np.zeros([X.shape[0], len(k)])        
        # out[:, 0] = abs(np.sin(X[:,0]*np.pi)) * 5
        # out[:, 1] = abs(np.sin(X[:,0]*np.pi+np.pi)) * 5


        out[:, 0] = (abs((X[:,0]-10)*(X[:,0])*(X[:,0]+10))) * 0.025
        # mode k=1 sigma varies with the sin of pi x
        out[:, 1] = abs(np.sin(X[:,0]*np.pi/5)) * 10
        print(out[0:10,:])
        # out[:, 1] = 
        # print(out[0:10,:])
        # print(X[0:10,0])
        return out

        # # Example: Sigma depends on x or -x
        # out = np.zeros([X.shape[0], len(k)])
        # out[:, 0] = abs(X[:,0]) * 0.5 + 0.1
        # out[:, 1] = abs(-X[:,0]) * 0.5 + 0.1
        # return out
    
    def test_mu_fn(self, X, mode_locs, k):
        # Example: two modes, nonlinear in x
        out = np.zeros([X.shape[0], len(k)])
        out[:, 0] = X[:, 0]**2 - 25  # mode 1: y ~ x
        out[:, 1] = -X[:, 0]**2 + 25  # mode 2: y ~ -x

        if X.shape[1] > 1:
            out[:, 0] = X[:, 0]**2 - 25 + 5*X[:, 1]  # mode 1: y ~ x
            out[:, 1] = -X[:, 0]**2 + 25 - 5*X[:, 1]  # mode 2: y ~ -x

        return out

    def test_pi_fn(self, X, n_modes):
        # Example: fixed weights
        return np.ones((X.shape[0], n_modes)) * [0.5, 0.5]

    def test_no_fn(self, X):
        return np.zeros(X.shape[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic multi-modal regression data (conditional version).")
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_modes', type=int, default=2)
    parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--mode_locs', type=int, default=None)
    parser.add_argument('--component_type', type=str, default='gaussian', choices=['gaussian', 'student-t'])
    parser.add_argument('--component_df', type=float, default=None)
    parser.add_argument('--mode_scales', type=float, nargs='+', default=[0.5,0.5])
    parser.add_argument('--x_sampler', type=str, default='uniform', choices=['uniform', 'normal', 'grid'])
    parser.add_argument('--x_low', type=float, default=-10)
    parser.add_argument('--x_high', type=float, default=10)
    parser.add_argument('--x_mean', type=float, default=0)
    parser.add_argument('--x_std', type=float, default=1)
    parser.add_argument('--x_grid_points', type=int, default=100)
    parser.add_argument('--seed_master', type=int, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--output_plot', type=str, default=None)
    args = parser.parse_args()

    # Expand mode_scales if needed
    if len(args.mode_scales) == 1:
        mode_scales = [args.mode_scales[0]] * args.n_modes
    else:
        mode_scales = args.mode_scales

    # x_sampler_params
    x_sampler_params = {}
    if args.x_sampler == 'uniform':
        x_sampler_params = {'low': args.x_low, 'high': args.x_high}
    elif args.x_sampler == 'normal':
        x_sampler_params = {'mean': args.x_mean, 'std': args.x_std}
    elif args.x_sampler == 'grid':
        x_sampler_params = {'low': args.x_low, 'high': args.x_high, 'grid_points': args.x_grid_points}

    dataset = SyntheticMultiModalConditionalDataset(
        n_samples=args.n_samples,
        n_modes=args.n_modes,
        n_features=args.n_features,
        mode_locs=None,  # Not used in this example
        mode_scales=mode_scales,
        component_type=args.component_type,
        component_df=args.component_df,
        x_sampler=args.x_sampler,
        x_sampler_params=x_sampler_params,
        seed_master=args.seed_master
    )
    X, y, global_mode, mode_ids = dataset.get_data(pi_fn=dataset.test_pi_fn, mu_fn=dataset.test_mu_fn, sigma_fn=dataset.test_sigma_fn, noise_fn=dataset.test_no_fn)
    df = dataset.to_dataframe(X, y)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved data to {args.output_csv}")
    dataset.plot(X, y, mu_fn = dataset.test_mu_fn, sigma_fn = dataset.test_sigma_fn, mode_ids=mode_ids)
    # print(f"Global mode location: {global_mode}")
    dataset.plot_conditional_y_given_x(X, y, dataset.test_pi_fn, dataset.test_mu_fn, dataset.test_sigma_fn)
