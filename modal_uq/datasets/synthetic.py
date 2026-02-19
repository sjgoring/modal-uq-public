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
        n_modes: int = 1,
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

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            X: (n_samples, n_features)
            y: (n_samples,)
            global_mode: (n_features,)
        """
        # Assign each sample to a mode
        mode_ids = self.rng_mode_assign.choice(self.n_modes, size=self.n_samples, p=self.mode_weights)
        # Generate X uniformly in [-10, 10]^d
        X = self.rng_sample.uniform(-10, 10, size=(self.n_samples, self.n_features))
        y = np.zeros(self.n_samples)
        for k in range(self.n_modes):
            idx = np.where(mode_ids == k)[0]
            if len(idx) == 0:
                continue
            if self.component_type == 'gaussian':
                y[idx] = self.rng_sample.normal(
                    loc=np.dot(X[idx], np.ones(self.n_features)) * 0 + self.mode_locs[k, 0],
                    scale=self.mode_scales[k],
                    size=len(idx)
                )
            elif self.component_type == 'student-t':
                y[idx] = self.rng_sample.standard_t(
                    df=self.component_df if self.component_df else 3,
                    size=len(idx)
                ) * self.mode_scales[k] + self.mode_locs[k, 0]
            else:
                raise ValueError(f"Unknown component_type: {self.component_type}")
        # Global mode is the mode with highest weight (break ties by first)
        global_mode = self.mode_locs[np.argmax(self.mode_weights)]
        return X, y, global_mode

    def to_dataframe(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.n_features)])
        df['y'] = y
        return df

    def plot(self, X: np.ndarray, y: np.ndarray, mode_ids: Optional[np.ndarray] = None, save_path: Optional[str] = None):
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
    parser.add_argument('--n_modes', type=int, default=3)
    parser.add_argument('--n_features', type=int, default=1)
    parser.add_argument('--component_type', type=str, default='gaussian', choices=['gaussian', 'student-t'])
    parser.add_argument('--component_df', type=float, default=None)
    parser.add_argument('--mode_scales', type=float, nargs='+', default=[1.0])
    parser.add_argument('--seed_master', type=int, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--output_plot', type=str, default=None)
    args = parser.parse_args()

    # Expand mode_scales if needed
    if len(args.mode_scales) == 1:
        mode_scales = [args.mode_scales[0]] * args.n_modes
    else:
        mode_scales = args.mode_scales

    dataset = SyntheticMultiModalDataset(
        n_samples=args.n_samples,
        n_modes=args.n_modes,
        n_features=args.n_features,
        mode_scales=mode_scales,
        component_type=args.component_type,
        component_df=args.component_df,
        seed_master=args.seed_master
    )
    X, y, global_mode = dataset.get_data()
    df = dataset.to_dataframe(X, y)
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved data to {args.output_csv}")
    dataset.plot(X, y)
    print(f"Global mode location: {global_mode}")
