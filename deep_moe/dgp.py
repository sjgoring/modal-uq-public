"""
1D heteroskedastic data-generating process.

Three noise types:
- "gaussian": Y | X ~ N(mu(x), sigma(x)^2). Heteroskedastic Gaussian.
- "bimodal": Y | X ~ 0.5 N(mu(x) - delta(x), s(x)^2) + 0.5 N(mu(x) + delta(x), s(x)^2).
  Bimodality strengthens with |x| via delta(x) = 0.5 + |x|.
- "skewed": Y | X ~ skew-normal centered at mu(x) with scale sigma(x) and shape
  alpha(x). The shape parameter is x-dependent — left-skewed for x<0, right-skewed
  for x>0, with magnitude growing with |x|. Mean, variance, and skewness all
  depend on x.

Mean function: mu(x) = sin(pi * x) (used by all noise types).
X uniform on [-2, 2]. The 1D nature means we can plot true and predicted
densities to verify model fits.
"""

import numpy as np
from scipy import stats

from predictive import GridDensity1D


# ---------- DGP parameters ----------

X_MIN, X_MAX = -2.0, 2.0


def mean_function(x: np.ndarray) -> np.ndarray:
    """mu(x) = sin(pi * x). x is array of shape (n,) or (n, 1); returns (n,)."""
    if x.ndim == 2:
        x = x[:, 0]
    return np.sin(np.pi * x)


def inner_sigma(x: np.ndarray) -> np.ndarray:
    """Inner-component sigma for bimodal noise. Constant 0.2 across x."""
    if x.ndim == 2:
        x = x[:, 0]
    return 0.2 * np.ones_like(x)


def delta_function(x: np.ndarray) -> np.ndarray:
    """Mode-offset for bimodal noise. delta(x) = 0.5 + |x|."""
    if x.ndim == 2:
        x = x[:, 0]
    return 0.5 + np.abs(x)


def gaussian_sigma(x: np.ndarray) -> np.ndarray:
    """For gaussian noise: total std calibrated to match bimodal's total spread."""
    s = inner_sigma(x)
    delta = delta_function(x)
    return np.sqrt(s ** 2 + delta ** 2)


def skew_alpha(x: np.ndarray) -> np.ndarray:
    """Skew shape parameter alpha(x) = 5 * sign(x) * sqrt(|x|).
    
    At x=0, alpha=0 (Gaussian). At x=±1, |alpha|=5 (moderate skew).
    At x=±2, |alpha|=~7.07 (strong skew). Sign flips with x: left-skewed for
    negative x, right-skewed for positive x.
    """
    if x.ndim == 2:
        x = x[:, 0]
    return 5.0 * np.sign(x) * np.sqrt(np.abs(x))


def skew_sigma(x: np.ndarray) -> np.ndarray:
    """Heteroskedastic sigma for skewed noise: sigma(x) = 0.3 + 0.2 * |x|.
    
    Grows with |x|, giving a third moment-dependence (variance varies with x).
    Combined with skew_alpha(x), all three moments depend on x.
    """
    if x.ndim == 2:
        x = x[:, 0]
    return 0.3 + 0.2 * np.abs(x)


def _skew_normal_centering(alpha: float) -> tuple[float, float]:
    """For skew-normal with shape alpha, return (mean_factor, std_factor) used
    to standardize. A standard SN(alpha) has:
        mean = sqrt(2/pi) * delta, where delta = alpha / sqrt(1+alpha^2)
        var  = 1 - 2*delta^2/pi
    We use these to center and scale samples to mean 0, variance 1.
    """
    delta = alpha / np.sqrt(1.0 + alpha * alpha)
    sn_mean = np.sqrt(2.0 / np.pi) * delta
    sn_var = 1.0 - 2.0 * delta * delta / np.pi
    sn_std = np.sqrt(sn_var)
    return sn_mean, sn_std


# ---------- Sampling ----------


def generate(n: int, noise_dist: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) of shape (n, 1) and (n,)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(X_MIN, X_MAX, size=(n, 1))
    mu = mean_function(X)
    
    if noise_dist == "gaussian":
        sigma = gaussian_sigma(X)
        y = mu + sigma * rng.standard_normal(n)
    elif noise_dist == "bimodal":
        s = inner_sigma(X)
        delta = delta_function(X)
        signs = rng.choice([-1.0, 1.0], size=n)
        y = mu + signs * delta + s * rng.standard_normal(n)
    elif noise_dist == "skewed":
        alphas = skew_alpha(X)
        sigmas = skew_sigma(X)
        # Sample standard skew-normal per alpha, then standardize
        eps = np.zeros(n)
        for i in range(n):
            a = alphas[i]
            # Sample SN(alpha) via the U + |V| construction:
            # if (U, V) ~ N(0, [[1, delta], [delta, 1]]), then U + |V| not the right
            # construction. Use the standard one: Z = delta * |W| + sqrt(1-delta^2) * V
            # where W, V ~ N(0,1) independent.
            delta_a = a / np.sqrt(1.0 + a * a)
            W = rng.standard_normal()
            V = rng.standard_normal()
            z = delta_a * np.abs(W) + np.sqrt(1.0 - delta_a ** 2) * V
            sn_mean, sn_std = _skew_normal_centering(a)
            eps[i] = (z - sn_mean) / sn_std  # standardized to mean 0, var 1
        y = mu + sigmas * eps
    else:
        raise ValueError(f"Unknown noise_dist: {noise_dist}")
    
    return X, y


# ---------- True conditional density ----------


def true_conditional_density(
    y: np.ndarray, x: np.ndarray, noise_dist: str
) -> np.ndarray:
    """Evaluate p*(y | x) at the given y values for a single input x.
    
    Args:
        y: array of shape (n_y,).
        x: array of shape (1,) or (1, 1).
        noise_dist: "gaussian", "bimodal", or "skewed".
    
    Returns:
        density values of shape (n_y,).
    """
    x_arr = np.atleast_2d(x)  # (1, 1)
    mu = mean_function(x_arr)[0]
    
    if noise_dist == "gaussian":
        sigma = gaussian_sigma(x_arr)[0]
        return stats.norm.pdf(y, loc=mu, scale=sigma)
    elif noise_dist == "bimodal":
        s = inner_sigma(x_arr)[0]
        delta = delta_function(x_arr)[0]
        return 0.5 * (
            stats.norm.pdf(y, loc=mu - delta, scale=s)
            + stats.norm.pdf(y, loc=mu + delta, scale=s)
        )
    elif noise_dist == "skewed":
        alpha = skew_alpha(x_arr)[0]
        sigma = skew_sigma(x_arr)[0]
        sn_mean, sn_std = _skew_normal_centering(alpha)
        # The standardized noise z ~ centered/scaled SN(alpha) with mean 0, var 1.
        # We have y = mu + sigma * z, so z = (y - mu)/sigma, and
        # the std SN value s = sn_std * z + sn_mean.
        # Density transformation: p(y) = (1 / sigma) * (1 / sn_std) *
        #                                 sn_pdf((y - mu)/sigma * sn_std + sn_mean)
        # But scipy.stats.skewnorm.pdf takes a shape param and a location/scale,
        # so we can just apply location/scale directly.
        # Effective loc = mu - sigma * sn_mean / sn_std
        # Effective scale = sigma / sn_std
        effective_loc = mu - sigma * sn_mean / sn_std
        effective_scale = sigma / sn_std
        return stats.skewnorm.pdf(y, alpha, loc=effective_loc, scale=effective_scale)
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")


def make_true_density(
    x: np.ndarray, noise_dist: str, n_grid: int = 5000
) -> GridDensity1D:
    """Build a GridDensity1D for the true conditional density at input x.
    
    Grid range chosen to cover the bulk of mass for each noise type.
    """
    x_arr = np.atleast_2d(x)
    mu = mean_function(x_arr)[0]
    
    if noise_dist == "gaussian":
        outer_sigma = gaussian_sigma(x_arr)[0]
        margin_left = 5 * outer_sigma
        margin_right = 5 * outer_sigma
    elif noise_dist == "bimodal":
        s = inner_sigma(x_arr)[0]
        delta = delta_function(x_arr)[0]
        outer_sigma = np.sqrt(s ** 2 + delta ** 2)
        margin_left = 5 * outer_sigma
        margin_right = 5 * outer_sigma
    elif noise_dist == "skewed":
        # Skew-normal can have a long tail on one side; size the grid asymmetrically.
        alpha = skew_alpha(x_arr)[0]
        sigma = skew_sigma(x_arr)[0]
        if alpha >= 0:
            # Right-skewed: long right tail
            margin_left = 4 * sigma
            margin_right = 8 * sigma
        else:
            # Left-skewed: long left tail
            margin_left = 8 * sigma
            margin_right = 4 * sigma
    else:
        raise ValueError(noise_dist)
    
    y_grid = np.linspace(mu - margin_left, mu + margin_right, n_grid)
    densities = true_conditional_density(y_grid, x, noise_dist)
    return GridDensity1D(y_grid, densities)


# ---------- Sanity check ----------

if __name__ == "__main__":
    print("DGP sanity checks")
    print("=" * 50)
    
    for nd in ["gaussian", "bimodal", "skewed"]:
        X, y = generate(n=1000, noise_dist=nd, seed=42)
        print(f"\n{nd}: X range [{X.min():.2f}, {X.max():.2f}], "
              f"y range [{y.min():.2f}, {y.max():.2f}]")
        print(f"  median |y - mu(x)|: "
              f"{np.median(np.abs(y - mean_function(X))):.3f}")
        
        # Verify density integrates to 1
        for x_test in [np.array([-1.5]), np.array([0.0]), np.array([1.5])]:
            tdens = make_true_density(x_test, nd)
            mass = np.trapz(tdens.density_values, tdens.y_grid)
            print(f"  density mass at x={x_test[0]:+.1f}: {mass:.6f} "
                  f"(should be ~1.0)")
