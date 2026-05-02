"""
Friedman #1 benchmark with configurable noise distributions.

The Friedman #1 function is:
    y = 10 * sin(pi * x_1 * x_2) + 20 * (x_3 - 0.5)^2 + 10 * x_4 + 5 * x_5 + epsilon

with x in [0, 1]^d (d >= 5), where the last d-5 features are noise.
"""

import numpy as np
import torch
from scipy import stats


def friedman_mean(x: np.ndarray) -> np.ndarray:
    """Compute the deterministic part of Friedman #1.
    
    Args:
        x: array of shape (n, d), d >= 5, with x in [0, 1].
    
    Returns:
        array of shape (n,) with the mean function values.
    """
    return (10 * np.sin(np.pi * x[:, 0] * x[:, 1])
            + 20 * (x[:, 2] - 0.5) ** 2
            + 10 * x[:, 3]
            + 5 * x[:, 4])


def noise_scale_function(X: np.ndarray, base_scale: float = 1.0) -> np.ndarray:
    """Per-sample noise scale (heteroscedastic).
    
    Sigma(x) = base_scale * (0.5 + |x_1 - 0.5|).
    Varies between 0.5 * base_scale (at x_1 = 0.5) and 1.0 * base_scale
    (at x_1 = 0 or 1), giving a 2x heteroscedasticity range.
    
    Uses x_1 specifically because it appears in the signal (sin(pi * x_1 * x_2)),
    so the network must learn that noise covaries with an already-attended feature.
    
    Args:
        X: array of shape (n, d).
        base_scale: overall scale multiplier.
    
    Returns:
        array of shape (n,) with per-sample noise scale.
    """
    return base_scale * (0.5 + np.abs(X[:, 0] - 0.5))


def sample_noise(
    X: np.ndarray, dist: str, base_scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample heteroscedastic noise values.
    
    For each sample i, draws epsilon_i from a base distribution (rescaled to
    unit variance where possible) and multiplies by sigma(x_i).
    
    Args:
        X: array of shape (n, d), the input features.
        dist: one of "gaussian", "t5", "t3".
        base_scale: base scale parameter (passed to noise_scale_function).
        rng: numpy random generator.
    
    Returns:
        array of shape (n,) with noise samples.
    """
    n = X.shape[0]
    sigma_x = noise_scale_function(X, base_scale=base_scale)
    
    if dist == "gaussian":
        eps = rng.standard_normal(n)
    elif dist == "t5":
        eps = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    elif dist == "t3":
        eps = rng.standard_t(df=3, size=n) / np.sqrt(3)
    else:
        raise ValueError(f"Unknown noise distribution: {dist}")
    
    return sigma_x * eps


def generate_friedman(
    n: int,
    d: int = 10,
    noise_dist: str = "gaussian",
    noise_scale: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Friedman #1 data with specified noise.
    
    Args:
        n: number of samples.
        d: feature dimension (>= 5; first 5 are signal, rest are noise).
        noise_dist: noise distribution ("gaussian", "t5", "t3").
        noise_scale: base noise scale (sigma(x) varies with x; see noise_scale_function).
        seed: random seed.
    
    Returns:
        (X, y): X of shape (n, d), y of shape (n,).
    """
    if d < 5:
        raise ValueError("d must be >= 5 for Friedman #1.")
    
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, d))
    y = friedman_mean(X) + sample_noise(X, noise_dist, noise_scale, rng)
    return X, y


def true_conditional_density(
    y: np.ndarray,
    x: np.ndarray,
    noise_dist: str,
    noise_scale: float,
) -> np.ndarray:
    """Evaluate the true conditional density p(y | x) at given points.
    
    This is the ground-truth density used for oracle AU and TU computation.
    Uses the heteroscedastic noise scale sigma(x) = noise_scale_function(x, noise_scale).
    
    Args:
        y: array of shape (n_y,) with y values to evaluate density at.
        x: array of shape (d,) with single feature vector (the conditioning point).
        noise_dist: noise distribution ("gaussian", "t5", "t3").
        noise_scale: base noise scale (passed to noise_scale_function).
    
    Returns:
        array of shape (n_y,) with density values.
    """
    mean = friedman_mean(x[None, :])[0]
    sigma_local = float(noise_scale_function(x[None, :], base_scale=noise_scale)[0])
    residuals = y - mean
    
    if noise_dist == "gaussian":
        return stats.norm.pdf(residuals, loc=0, scale=sigma_local)
    elif noise_dist == "t5":
        # Rescaling by sqrt(5/3) when sampling, so density rescales accordingly
        scaled_scale = sigma_local / np.sqrt(5 / 3)
        return stats.t.pdf(residuals / scaled_scale, df=5) / scaled_scale
    elif noise_dist == "t3":
        scaled_scale = sigma_local / np.sqrt(3)
        return stats.t.pdf(residuals / scaled_scale, df=3) / scaled_scale
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")


if __name__ == "__main__":
    # Quick sanity check
    for dist in ["gaussian", "t5", "t3"]:
        X, y = generate_friedman(n=1000, noise_dist=dist, noise_scale=1.0, seed=42)
        residuals = y - friedman_mean(X)
        print(f"{dist}: y range [{y.min():.2f}, {y.max():.2f}], "
              f"residual median |.| = {np.median(np.abs(residuals)):.3f}")
        
        # Verify true density integrates to 1 (Monte Carlo check)
        x_test = X[0]
        y_grid = np.linspace(-50, 50, 10000)
        density = true_conditional_density(y_grid, x_test, dist, 1.0)
        integral = np.trapz(density, y_grid)
        print(f"  density integral at x_test: {integral:.4f} (should be ~1)")
        
        # Verify heteroscedasticity: residual magnitude should covary with sigma(x)
        sigma_x = noise_scale_function(X, base_scale=1.0)
        # Bin by sigma(x), check whether |residual| matches
        order = np.argsort(sigma_x)
        n_bins = 4
        bin_size = len(X) // n_bins
        for b in range(n_bins):
            idx = order[b * bin_size:(b + 1) * bin_size]
            mean_sigma = sigma_x[idx].mean()
            mean_abs_res = np.median(np.abs(residuals[idx]))
            print(f"    bin {b}: sigma(x) ≈ {mean_sigma:.3f}, "
                  f"median |residual| ≈ {mean_abs_res:.3f}")
