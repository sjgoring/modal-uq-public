"""
Synthetic Multi-Modal Regression Data Generator

- Supports arbitrary number of modes (k), dimensions (d), and sample size (n)
- Allows explicit control of mode locations, dispersions, and weights
- Supports heavy-tailed components (Student-t)
- Full reproducibility via independent seeds for each stochastic process
- Usable as a dataset module or standalone script (CSV + plots)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union

class SyntheticMultiModalDataset:
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
        seed_master: Optional[int] = 42,
        seed_mode_locs: Optional[int] = None,
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
            seed_master: Master seed (optional, used to derive all other seeds if set)
            seed_mode_locs: Seed for mode location randomisation
            seed_mode_assign: Seed for mode assignment
            seed_sample: Seed for sample generation within components
            seed_noise: Seed for noise/outlier generation (future use)
        """
        self.n_samples = n_samples
        self.n_modes = n_modes
        self.n_features = n_features
        self.component_type = component_type
        self.component_df = component_df
        self.mode_scales = np.array(mode_scales if isinstance(mode_scales, (list, np.ndarray)) else [mode_scales]*n_modes)
        self.mode_weights = np.array(mode_weights) if mode_weights is not None else np.ones(n_modes) / n_modes
        self.seed_master = seed_master
        # Derive seeds if only master is set
        self.seed_mode_locs = seed_mode_locs if seed_mode_locs is not None else (hash((seed_master, 'locs')) % 2**32 if seed_master is not None else None)
        self.seed_mode_assign = seed_mode_assign if seed_mode_assign is not None else (hash((seed_master, 'assign')) % 2**32 if seed_master is not None else None)
        self.seed_sample = seed_sample if seed_sample is not None else (hash((seed_master, 'sample')) % 2**32 if seed_master is not None else None)
        self.seed_noise = seed_noise if seed_noise is not None else (hash((seed_master, 'noise')) % 2**32 if seed_master is not None else None)
        # Generators
        self.rng_mode_locs = np.random.default_rng(self.seed_mode_locs)
        self.rng_mode_assign = np.random.default_rng(self.seed_mode_assign)
        self.rng_sample = np.random.default_rng(self.seed_sample)
        self.rng_noise = np.random.default_rng(self.seed_noise)
        # Mode locations
        if mode_locs is not None:
            self.mode_locs = np.array(mode_locs)
        else:
            self.mode_locs = self.rng_mode_locs.normal(loc=0, scale=5, size=(n_modes, n_features))

    def get_data(
        self,
        pi_fn: Optional[callable] = None,
        mu_fn: Optional[callable] = None,
        sigma_fn: Optional[callable] = None,
        noise_fn: Optional[callable] = None
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
        # Sample X from base distribution
        gp = self.rng_sample.uniform(-10, 10, size=(self.n_samples, self.n_features+1))
        X = np.zeros([self.n_samples, self.n_features])
        n_modes = self.n_modes
        pi = pi_fn(gp) if pi_fn is not None else np.ones((self.n_samples, n_modes)) / n_modes
        mu = mu_fn(gp, self.mode_locs, np.arange(n_modes)) if mu_fn is not None else np.tile(self.mode_locs[:, 0], (self.n_samples, 1))
        sigma = sigma_fn(gp, self.mode_locs, np.arange(n_modes)) if sigma_fn is not None else np.tile(self.mode_scales, (self.n_samples, 1))
        y = np.zeros(self.n_samples)
        mode_ids = np.zeros(self.n_samples, dtype=int)
        for i in range(self.n_samples):
            mode = self.rng_mode_assign.choice(n_modes, p=pi[i])
            mode_ids[i] = mode
            # print(mu[i,mode])
            # print(self.rng_sample.multivariate_normal(mean=mu[i, mode], cov=sigma[i, mode]))
            X[i], y[i] = self.rng_sample.multivariate_normal(mean=mu[i, mode], cov=sigma[i, mode])
            # print(X[i], y[i])
            if noise_fn is not None:
                y[i] += self.rng_noise.normal(loc=0, scale=noise_fn(X[i:i+1])[0])
        global_mode = self.mode_locs[np.argmax(np.mean(pi, axis=0))]
        return X, y, global_mode, mode_ids

    def to_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.n_features)])
        df['y'] = y
        return df

    def plot(self, X: np.ndarray, y: np.ndarray, mode_ids: Optional[np.ndarray] = None, save_path: Optional[str] = None, plot3d: bool = True):
        if self.n_features == 1:
            plt.figure(figsize=(8, 5))
            plt.scatter(X[:, 0], y, c=mode_ids if mode_ids is not None else 'b', cmap='tab10', alpha=0.5, label='Samples')
            for k, loc in enumerate(self.mode_locs):
                plt.axhline(loc[0], color=f'C{k}', linestyle='--', label=f'Mode {k+1}')
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic multi-modal regression data.")
    parser.add_argument('--n_samples', type=int, default=1000)
    # parser.add_argument('--n_modes', type=int, default=1)
    parser.add_argument('--n_modes', type=int, default=2)
    parser.add_argument('--n_features', type=int, default=1)
    # parser.add_argument('--mode_locs', type=int, default=[[0],[5]])
    # parser.add_argument('--n_features', type=int, default=2)
    parser.add_argument('--mode_locs', type=int, default=[[0,0],[5,5]])
    # parser.add_argument('--n_features', type=int, default=1)
    # parser.add_argument('--mode_locs', type=int, default=[[0,0]])
    parser.add_argument('--component_type', type=str, default='gaussian', choices=['gaussian', 'student-t'])
    parser.add_argument('--component_df', type=float, default=None)
    parser.add_argument('--mode_scales', type=float, nargs='+', default=[0.5,0.5])
    parser.add_argument('--seed_master', type=int, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--output_plot', type=str, default=None)
    args = parser.parse_args()

    # Expand mode_scales if needed
    if len(args.mode_scales) == 1:
        mode_scales = [args.mode_scales[0]] * args.n_modes
    else:
        mode_scales = args.mode_scales

    # Note these functions are of the joint distribution now, i.e. X, y

    def sigma_fn(X, mode_locs, k):
        if mode_locs.shape[1] == 2:
            # if 2 dimensional (1 feature)
            # a = np.abs(1/-2-np.sum(X**2,axis=1))
            # b = np.sum(X**2,axis=1)
            # b = np.zeros_like(c)
            c = abs(X[:,1])**2*0.1
            # c = np.ones(X.shape[0]) * 0.5
            # c = np.zeros(X.shape[0]) * 0.5
            a = c
            b = np.zeros_like(c)
            sigmas = np.stack([np.stack([a,b],axis=1),np.stack([b,c],axis=1)],axis=2)

        elif mode_locs.shape[1] == 1:
            # If 1 dimensional
            sigmas = np.expand_dims(np.expand_dims(np.abs(1/-2-np.sum(X**3,axis=1)),axis=1),axis=2)

        if mode_locs.shape[0] == 2:
                # If 2 modes, repeat
                sigmas = np.stack([sigmas, sigmas], axis=1)

        # print(sigmas.shape)
        return sigmas

    def mu_fn(X, mode_locs, k):
        # For 1 feature, 1 response only, 2 modes
        # print(mode_locs.shape)
        out = np.zeros([X.shape[0], mode_locs.shape[0], mode_locs.shape[1]])
        # Define y as a function of x
        # print(out.shape)
        out = np.stack([np.stack([X[:,0], -X[:,0]**2 + 25],axis=1), np.stack([X[:,0], X[:,0]**2 - 25],axis=1)], axis=1)
        
        # print(out.shape)
        # + 2 * np.ones(mode_locs.shape[0]) * i for i in range(len(k))
        return out

    def pi_fn(X):
        out = np.ones((X.shape[0], args.n_modes)) * [0.7,0.3]
        # print(out)
        return out

    def no_fn(X):
        return np.zeros(X.shape[0])

    dataset = SyntheticMultiModalDataset(
        n_samples=args.n_samples,
        n_modes=args.n_modes,
        n_features=args.n_features,
        mode_locs=args.mode_locs,
        mode_scales=mode_scales,
        component_type=args.component_type,
        component_df=args.component_df,
        seed_master=args.seed_master
    )
    X, y, global_mode, mode_ids = dataset.get_data(pi_fn=pi_fn, mu_fn=mu_fn, sigma_fn=sigma_fn, noise_fn=no_fn)
    df = dataset.to_dataframe(X, y)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved data to {args.output_csv}")
    dataset.plot(X, y, mode_ids=mode_ids)
    print(f"Global mode location: {global_mode}")

    ## TODO:
    ## Currently this script runs a grid over the joint distribution of X and y.
    # At the time this made more sense to me, but as I think about it, given we specify some relation between x and y (see mu_fn)
    # we could just as well run the grid over x, and then compute y from the specified relation + noise (the marginal dist of y|x at each x). This would be more efficient and more interpretable I think.
    # I will refactor to do this, but for now I will keep the current structure as it is working and I want to move on to the next steps.