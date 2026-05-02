"""
Predictive distribution objects.

The deep ensemble produces a predictive distribution that is a Gaussian mixture
over the M ensemble components. This module provides a class that wraps this
mixture and supports density evaluation, sampling, and HDR computation.

For comparison, we also support arbitrary 1D distributions defined via a
density function on a grid (used for true conditional densities under
non-Gaussian noise).
"""

import numpy as np
from scipy import stats


class GaussianMixture1D:
    """A 1D Gaussian mixture distribution.
    
    Represents p(y) = sum_m w_m * N(y | mu_m, sigma_m^2).
    Used for ensemble predictive distributions.
    """
    
    def __init__(self, mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray = None):
        """
        Args:
            mus: array of shape (M,) with component means.
            sigmas: array of shape (M,) with component standard deviations.
            weights: array of shape (M,) with component weights (default: uniform).
        """
        self.mus = np.asarray(mus, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)
        M = len(mus)
        if weights is None:
            self.weights = np.ones(M) / M
        else:
            self.weights = np.asarray(weights, dtype=float)
            self.weights = self.weights / self.weights.sum()
        self.M = M
    
    def density(self, y: np.ndarray) -> np.ndarray:
        """Evaluate density at points y."""
        y = np.atleast_1d(y)
        # Shape: (n_y, M)
        component_densities = stats.norm.pdf(
            y[:, None], loc=self.mus[None, :], scale=self.sigmas[None, :]
        )
        return component_densities @ self.weights
    
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n values from the mixture."""
        components = rng.choice(self.M, size=n, p=self.weights)
        return rng.normal(loc=self.mus[components], scale=self.sigmas[components])
    
    def mean(self) -> float:
        return float(self.weights @ self.mus)
    
    def variance(self) -> float:
        """Total variance via law of total variance."""
        within = float(self.weights @ self.sigmas ** 2)
        between = float(self.weights @ (self.mus - self.mean()) ** 2)
        return within + between
    
    def grid(self, n_grid: int = 2000, n_sigma: float = 6.0) -> np.ndarray:
        """Construct a fine grid covering the bulk of the mixture's mass.
        
        Range is [min(mu) - n_sigma * max(sigma), max(mu) + n_sigma * max(sigma)].
        """
        margin = n_sigma * self.sigmas.max()
        y_min = self.mus.min() - margin
        y_max = self.mus.max() + margin
        return np.linspace(y_min, y_max, n_grid)


class GridDensity1D:
    """A 1D density represented on a fine grid.
    
    Used for arbitrary densities (e.g., t-distribution noise) where we don't
    have a closed-form mixture representation.
    """
    
    def __init__(self, y_grid: np.ndarray, density_values: np.ndarray):
        """
        Args:
            y_grid: array of shape (n,) with grid points (sorted ascending).
            density_values: array of shape (n,) with density values at grid points.
        """
        self.y_grid = np.asarray(y_grid)
        self.density_values = np.asarray(density_values)
        # Renormalize to ensure mass = 1 over grid
        mass = np.trapz(self.density_values, self.y_grid)
        self.density_values = self.density_values / mass
    
    def density(self, y: np.ndarray) -> np.ndarray:
        """Evaluate density at y by interpolation."""
        return np.interp(y, self.y_grid, self.density_values, left=0, right=0)
    
    def grid(self) -> np.ndarray:
        return self.y_grid


def compute_hdr(
    distribution,
    alpha: float,
    n_grid: int = 5000,
) -> tuple[float, np.ndarray, float]:
    """Compute the (1 - alpha) HDR of a 1D distribution.
    
    Uses the standard algorithm: evaluate density on a fine grid, sort by
    density (descending), accumulate mass until threshold, return the
    corresponding level set.
    
    Args:
        distribution: object with `grid()` and `density(y)` methods.
        alpha: tolerance level in (0, 1). HDR captures (1 - alpha) of mass.
        n_grid: number of grid points (used if distribution doesn't specify).
    
    Returns:
        (volume, hdr_mask, threshold):
            volume: Lebesgue measure of the HDR.
            hdr_mask: boolean array of shape (n_grid,) indicating HDR membership.
            threshold: the density threshold t_alpha.
    """
    if hasattr(distribution, 'grid') and callable(distribution.grid):
        try:
            y_grid = distribution.grid(n_grid=n_grid)
        except TypeError:
            y_grid = distribution.grid()
    else:
        raise ValueError("Distribution must have a grid() method.")
    
    densities = distribution.density(y_grid)
    dy = y_grid[1] - y_grid[0]
    
    # Sort indices by density (descending)
    sort_idx = np.argsort(-densities)
    sorted_densities = densities[sort_idx]
    
    # Mass at each sorted grid cell is density * dy
    sorted_mass = sorted_densities * dy
    cumulative_mass = np.cumsum(sorted_mass)
    
    target_mass = 1 - alpha
    # Find smallest k such that cumulative_mass[k] >= target_mass
    k = np.searchsorted(cumulative_mass, target_mass)
    if k >= len(cumulative_mass):
        k = len(cumulative_mass) - 1
    
    threshold = sorted_densities[k]
    hdr_mask = densities >= threshold
    volume = float(hdr_mask.sum() * dy)
    
    return volume, hdr_mask, float(threshold)


if __name__ == "__main__":
    # Sanity check: HDR of a single Gaussian at alpha=0.5 should be ~1.349 sigma
    # (since 50% HDR of N(0, 1) is [-0.6745, 0.6745], width 1.349)
    g = GaussianMixture1D(mus=np.array([0.0]), sigmas=np.array([1.0]))
    vol_50, _, _ = compute_hdr(g, alpha=0.5)
    print(f"50% HDR width of N(0,1): {vol_50:.3f} (expected ~1.349)")
    
    vol_10, _, _ = compute_hdr(g, alpha=0.1)
    print(f"90% HDR width of N(0,1): {vol_10:.3f} (expected ~3.290)")
    
    # Bimodal mixture: 50% HDR may be disconnected
    mix = GaussianMixture1D(mus=np.array([-3.0, 3.0]), sigmas=np.array([0.5, 0.5]))
    vol_50_bi, mask_bi, _ = compute_hdr(mix, alpha=0.5)
    grid_bi = mix.grid()
    # Count connected components in the mask
    transitions = np.diff(mask_bi.astype(int))
    n_components = (transitions == 1).sum()
    print(f"50% HDR of bimodal mixture: volume {vol_50_bi:.3f}, "
          f"{n_components} component(s)")
    print(f"  (Two well-separated peaks should give 2 components.)")
    
    # Check mean and variance of mixture
    print(f"Bimodal mixture: mean = {mix.mean():.3f}, var = {mix.variance():.3f}")
    print(f"  (Expected: mean = 0, var = 0.25 + 9 = 9.25)")
