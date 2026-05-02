"""
Uncertainty measures for regression.

Implements:
- Variance-based: AU, EU, TU (decomposed via law of total variance).
- Entropy-based: differential entropy, expected within-component entropy,
  Jensen gap (mutual information).
- QUEST: V_alpha, integrated volume, truth-relative TU via TVD penalty.

Each measure operates on:
- A predictive distribution (Gaussian mixture from ensemble) for plug-in EU/TU.
- The true conditional density p_theta_star for AU and TU oracle.
"""

import numpy as np
from scipy import integrate

from predictive import GaussianMixture1D, GridDensity1D, compute_hdr


# ==================== Variance-based measures ====================

def variance_au(predictive: GaussianMixture1D) -> float:
    """Aleatoric uncertainty as expected within-component variance."""
    return float(predictive.weights @ predictive.sigmas ** 2)


def variance_eu(predictive: GaussianMixture1D) -> float:
    """Epistemic uncertainty as variance of component means."""
    mean_of_means = predictive.mean()
    return float(predictive.weights @ (predictive.mus - mean_of_means) ** 2)


def variance_tu(predictive: GaussianMixture1D) -> float:
    """Total variance via law of total variance: AU + EU."""
    return variance_au(predictive) + variance_eu(predictive)


# ==================== Entropy-based measures ====================

def gaussian_entropy(sigma: float) -> float:
    """Differential entropy of N(mu, sigma^2) (independent of mu)."""
    return 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)


def entropy_au(predictive: GaussianMixture1D) -> float:
    """Aleatoric uncertainty as expected within-component differential entropy."""
    component_entropies = np.array([gaussian_entropy(s) for s in predictive.sigmas])
    return float(predictive.weights @ component_entropies)


def entropy_tu(predictive: GaussianMixture1D, n_grid: int = 5000) -> float:
    """Total uncertainty as differential entropy of the predictive mixture.
    
    Computed numerically since mixture differential entropy has no closed form.
    """
    y_grid = predictive.grid(n_grid=n_grid)
    densities = predictive.density(y_grid)
    # Avoid log(0) via masking
    mask = densities > 1e-300
    return float(-np.trapz(
        np.where(mask, densities * np.log(np.where(mask, densities, 1)), 0),
        y_grid
    ))


def entropy_eu(predictive: GaussianMixture1D, n_grid: int = 5000) -> float:
    """Epistemic uncertainty as Jensen gap: h(predictive) - E[h(p_theta)]."""
    return entropy_tu(predictive, n_grid=n_grid) - entropy_au(predictive)


# ==================== QUEST measures ====================

def quest_au_local(true_dist, alpha: float) -> float:
    """Local AU: V_alpha of the true conditional density."""
    volume, _, _ = compute_hdr(true_dist, alpha=alpha)
    return volume


def quest_tu_local(
    true_dist,
    predictive: GaussianMixture1D,
    alpha: float,
    n_grid: int = 5000,
) -> float:
    """Local TU: V_alpha(p_theta_star) / (1 - TVD(p_alpha_star, p_alpha_hat)).
    
    p_alpha_star is the true density conditioned on the true HDR.
    p_alpha_hat is the predictive density conditioned on the predictive HDR.
    
    Both conditional densities are extended by zero to a common grid covering
    both supports, so TVD is well-defined even when HDRs are disjoint.
    """
    # Construct a common grid that covers both true and predictive supports.
    # Get tentative grids from each and take the union range.
    pred_grid = predictive.grid(n_grid=n_grid)
    
    # If true_dist provides its own grid, use its range too
    if hasattr(true_dist, 'grid') and callable(true_dist.grid):
        try:
            true_grid = true_dist.grid(n_grid=n_grid)
        except TypeError:
            true_grid = true_dist.grid()
        y_min = min(pred_grid[0], true_grid[0])
        y_max = max(pred_grid[-1], true_grid[-1])
    else:
        y_min, y_max = pred_grid[0], pred_grid[-1]
    
    y_grid = np.linspace(y_min, y_max, n_grid)
    dy = y_grid[1] - y_grid[0]
    
    # Evaluate both densities on the common grid
    true_density = true_dist.density(y_grid)
    pred_density = predictive.density(y_grid)
    
    # Compute HDR masks on the common grid
    v_true, mask_true_grid, _ = compute_hdr_on_grid(true_density, dy, alpha=alpha)
    _, mask_pred_grid, _ = compute_hdr_on_grid(pred_density, dy, alpha=alpha)
    
    # Conditional densities (extended by zero to common grid)
    # Note: we need to renormalize by the actual mass within each HDR on this grid,
    # not just (1 - alpha), to handle grid discretization correctly.
    true_hdr_mass = float(np.trapz(np.where(mask_true_grid, true_density, 0), y_grid))
    pred_hdr_mass = float(np.trapz(np.where(mask_pred_grid, pred_density, 0), y_grid))
    
    if true_hdr_mass < 1e-12 or pred_hdr_mass < 1e-12:
        # Pathological case (numerical underflow)
        return v_true * 1e12  # large but finite
    
    p_alpha_star = np.where(mask_true_grid, true_density / true_hdr_mass, 0.0)
    p_alpha_hat = np.where(mask_pred_grid, pred_density / pred_hdr_mass, 0.0)
    
    # TVD between the two conditional densities
    tvd = 0.5 * float(np.trapz(np.abs(p_alpha_star - p_alpha_hat), y_grid))
    tvd = min(max(tvd, 0.0), 1.0 - 1e-10)
    
    return v_true / (1 - tvd)


def compute_hdr_on_grid(
    densities: np.ndarray, dy: float, alpha: float
) -> tuple[float, np.ndarray, float]:
    """Helper: compute HDR given densities already evaluated on a grid."""
    sort_idx = np.argsort(-densities)
    sorted_densities = densities[sort_idx]
    sorted_mass = sorted_densities * dy
    cumulative_mass = np.cumsum(sorted_mass)
    target_mass = 1 - alpha
    k = np.searchsorted(cumulative_mass, target_mass)
    if k >= len(cumulative_mass):
        k = len(cumulative_mass) - 1
    threshold = sorted_densities[k]
    hdr_mask = densities >= threshold
    volume = float(hdr_mask.sum() * dy)
    return volume, hdr_mask, float(threshold)


def quest_eu_local(
    theta_samples: np.ndarray,
    alpha: float,
    bandwidth: str | float = "scott",
    n_grid: int = 100,
) -> float:
    """Local EU: V_alpha of the second-order distribution q on parameter space.
    
    For an ensemble, theta_samples is the M x 2 array of (mu, log_sigma) pairs
    from the M ensemble members at a given input x. We fit a 2D KDE and compute
    the (1 - alpha)-HDR volume on the parameter space.
    
    Args:
        theta_samples: array of shape (M, 2) with parameter samples.
            Conventionally (mu, log_sigma) for the Gaussian-output ensemble.
        alpha: tolerance level in (0, 1).
        bandwidth: KDE bandwidth ("scott", "silverman", or numeric value).
        n_grid: grid resolution per dimension.
    
    Returns:
        EU as the area (Lebesgue measure on parameter space) of the (1-alpha) HDR.
    """
    from scipy.stats import gaussian_kde
    
    M = theta_samples.shape[0]
    if M < 2:
        # Single component: degenerate q (point mass), EU = 0
        return 0.0
    
    # Fit 2D KDE on the ensemble samples
    # gaussian_kde expects (d, n) shape
    kde = gaussian_kde(theta_samples.T, bw_method=bandwidth)
    
    # Construct a 2D grid covering the bulk of the KDE mass
    margins_per_dim = []
    for j in range(2):
        col = theta_samples[:, j]
        std = max(col.std(), 1e-3)
        # Extend by 4 std beyond the sample range
        lo = col.min() - 4 * std
        hi = col.max() + 4 * std
        margins_per_dim.append((lo, hi))
    
    grid_axes = [np.linspace(lo, hi, n_grid) for (lo, hi) in margins_per_dim]
    g0, g1 = np.meshgrid(grid_axes[0], grid_axes[1], indexing='ij')
    grid_points = np.stack([g0.ravel(), g1.ravel()])  # shape (2, n_grid^2)
    
    densities = kde(grid_points).reshape(n_grid, n_grid)
    
    # Cell area
    da = (grid_axes[0][1] - grid_axes[0][0]) * (grid_axes[1][1] - grid_axes[1][0])
    
    # HDR via density sorting
    flat_d = densities.ravel()
    sort_idx = np.argsort(-flat_d)
    sorted_d = flat_d[sort_idx]
    sorted_mass = sorted_d * da
    cumulative = np.cumsum(sorted_mass)
    target = 1 - alpha
    k = np.searchsorted(cumulative, target)
    if k >= len(cumulative):
        k = len(cumulative) - 1
    threshold = sorted_d[k]
    hdr_mask = densities >= threshold
    return float(hdr_mask.sum() * da)


def quest_eu_global(
    theta_samples: np.ndarray,
    n_alpha: int = 50,
    bandwidth: str | float = "scott",
    n_grid: int = 100,
) -> float:
    """Global EU: integrated volume of the second-order KDE."""
    alphas = np.linspace(0.01, 0.99, n_alpha)
    volumes = np.array([
        quest_eu_local(theta_samples, a, bandwidth=bandwidth, n_grid=n_grid)
        for a in alphas
    ])
    return float(np.trapz(volumes, alphas))


def quest_au_global(true_dist, n_alpha: int = 50) -> float:
    """Global AU: integrated volume of true conditional density."""
    alphas = np.linspace(0.01, 0.99, n_alpha)
    volumes = np.array([quest_au_local(true_dist, a) for a in alphas])
    return float(np.trapz(volumes, alphas))


def quest_tu_global(
    true_dist,
    predictive: GaussianMixture1D,
    n_alpha: int = 50,
) -> float:
    """Global TU: integral of local TU over alpha."""
    alphas = np.linspace(0.01, 0.99, n_alpha)
    tu_values = np.array([quest_tu_local(true_dist, predictive, a) for a in alphas])
    return float(np.trapz(tu_values, alphas))


# ==================== Sanity checks ====================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    # Single Gaussian: variance and entropy should match closed-form values
    g = GaussianMixture1D(mus=np.array([0.0]), sigmas=np.array([2.0]))
    print("Single N(0, 4):")
    print(f"  variance AU = {variance_au(g):.4f} (expected 4.000)")
    print(f"  variance EU = {variance_eu(g):.4f} (expected 0.000)")
    print(f"  variance TU = {variance_tu(g):.4f} (expected 4.000)")
    print(f"  entropy AU = {entropy_au(g):.4f} (expected {gaussian_entropy(2.0):.4f})")
    print(f"  entropy EU = {entropy_eu(g):.4f} (expected ~0)")
    print(f"  entropy TU = {entropy_tu(g):.4f} (expected {gaussian_entropy(2.0):.4f})")
    
    # 90% HDR of N(0, 4) should have width 2 * 1.645 * 2 = 6.58
    print(f"  V_0.1 = {quest_au_local(g, 0.1):.4f} (expected ~6.580)")
    # 50% HDR: 2 * 0.6745 * 2 = 2.698
    print(f"  V_0.5 = {quest_au_local(g, 0.5):.4f} (expected ~2.698)")
    
    # Bimodal mixture
    print("\nBimodal mixture N(-3, 0.25) + N(3, 0.25):")
    mix = GaussianMixture1D(mus=np.array([-3.0, 3.0]), sigmas=np.array([0.5, 0.5]))
    print(f"  variance AU = {variance_au(mix):.4f} (expected 0.250)")
    print(f"  variance EU = {variance_eu(mix):.4f} (expected 9.000)")
    print(f"  variance TU = {variance_tu(mix):.4f} (expected 9.250)")
    print(f"  entropy TU = {entropy_tu(mix):.4f}")
    print(f"  V_0.5 = {quest_au_local(mix, 0.5):.4f} (expected ~1.349 = 2 * 0.6745)")
    
    # TU sanity check: when predictive == truth, TVD = 0, TU = AU
    print("\nTU when predictive matches truth:")
    truth = GaussianMixture1D(mus=np.array([0.0]), sigmas=np.array([1.0]))
    pred_match = GaussianMixture1D(mus=np.array([0.0]), sigmas=np.array([1.0]))
    tu_match = quest_tu_local(truth, pred_match, alpha=0.5)
    au = quest_au_local(truth, alpha=0.5)
    print(f"  TU = {tu_match:.4f}, AU = {au:.4f} (should be approximately equal)")
    
    # TU when predictive is overconfident wrong: should be much larger than AU
    print("\nTU when predictive is overconfident wrong:")
    pred_wrong = GaussianMixture1D(mus=np.array([5.0]), sigmas=np.array([0.1]))
    tu_wrong = quest_tu_local(truth, pred_wrong, alpha=0.5)
    print(f"  TU = {tu_wrong:.4f}, AU = {au:.4f} (TU should be much larger)")
    
    # QUEST EU sanity checks
    print("\nQUEST EU on parameter samples:")
    # Tightly clustered samples: small EU
    tight = rng.normal(0, 0.05, size=(5, 2))
    eu_tight = quest_eu_local(tight, alpha=0.5)
    print(f"  Tight cluster (std=0.05): EU = {eu_tight:.4f}")
    
    # Loose samples: larger EU
    loose = rng.normal(0, 1.0, size=(5, 2))
    eu_loose = quest_eu_local(loose, alpha=0.5)
    print(f"  Loose cluster (std=1.0):  EU = {eu_loose:.4f}")
    print(f"  (Loose should be substantially larger than tight.)")
