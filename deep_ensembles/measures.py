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

def _v_alpha_and_tvd(
    p_dist,
    q_dist,
    alpha: float,
    n_grid: int = 5000,
) -> tuple[float, float]:
    """Compute V_alpha(p) and TVD(p_alpha, q_alpha) on a shared grid.
    
    p_alpha is the conditional density of p restricted to its (1-alpha)-HDR;
    similarly for q_alpha. Both are extended by zero to the shared grid so
    TVD is well-defined even when HDRs are disjoint.
    
    Returns:
        (V_alpha_p, tvd_pq): V_alpha of p, and TVD between conditionals.
    """
    def _get_grid(dist):
        # GaussianMixture1D.grid(n_grid=...) and GridDensity1D.grid() differ
        try:
            return dist.grid(n_grid=n_grid)
        except TypeError:
            return dist.grid()
    
    p_grid = _get_grid(p_dist)
    q_grid = _get_grid(q_dist)
    
    y_min = min(p_grid[0], q_grid[0])
    y_max = max(p_grid[-1], q_grid[-1])
    y_grid = np.linspace(y_min, y_max, n_grid)
    dy = y_grid[1] - y_grid[0]
    
    p_density = p_dist.density(y_grid)
    q_density = q_dist.density(y_grid)
    
    v_p, mask_p, _ = compute_hdr_on_grid(p_density, dy, alpha=alpha)
    _, mask_q, _ = compute_hdr_on_grid(q_density, dy, alpha=alpha)
    
    p_hdr_mass = float(np.trapz(np.where(mask_p, p_density, 0), y_grid))
    q_hdr_mass = float(np.trapz(np.where(mask_q, q_density, 0), y_grid))
    
    if p_hdr_mass < 1e-12 or q_hdr_mass < 1e-12:
        return v_p, 1.0 - 1e-10
    
    p_alpha = np.where(mask_p, p_density / p_hdr_mass, 0.0)
    q_alpha = np.where(mask_q, q_density / q_hdr_mass, 0.0)
    
    tvd = 0.5 * float(np.trapz(np.abs(p_alpha - q_alpha), y_grid))
    tvd = min(max(tvd, 0.0), 1.0 - 1e-10)
    
    return v_p, tvd


def _component_distribution(predictive: GaussianMixture1D, m: int) -> GaussianMixture1D:
    """Extract the m-th component of an ensemble predictive as a single-component dist."""
    return GaussianMixture1D(
        mus=np.array([predictive.mus[m]]),
        sigmas=np.array([predictive.sigmas[m]]),
    )


# ----- Oracle versions (use known true conditional density p_theta_star) -----

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
    """Oracle local TU: V_alpha(p_theta_star) / (1 - TVD(p_alpha_star, p_alpha_hat))."""
    v_true, tvd = _v_alpha_and_tvd(true_dist, predictive, alpha, n_grid=n_grid)
    return v_true / (1 - tvd)


# ----- C2 plug-in: predicting model = w, truth approx = predictive bar_p -----

def quest_au_local_c2(predictive: GaussianMixture1D, alpha: float) -> float:
    """C2 local AU: E_w[V_alpha(p_w)] = mean V_alpha across ensemble components."""
    M = predictive.M
    vols = np.zeros(M)
    for m in range(M):
        comp = _component_distribution(predictive, m)
        v, _, _ = compute_hdr(comp, alpha=alpha)
        vols[m] = v
    return float(predictive.weights @ vols)


def quest_tu_local_c2(
    predictive: GaussianMixture1D,
    alpha: float,
    n_grid: int = 5000,
) -> float:
    """C2 local TU: E_w[V_alpha(p_w) / (1 - TVD(p_w_alpha, predictive_alpha))].
    
    Predicting model = each ensemble component w; truth approx = bar_p (predictive).
    """
    M = predictive.M
    tu_per_w = np.zeros(M)
    for m in range(M):
        comp = _component_distribution(predictive, m)
        v_w, tvd = _v_alpha_and_tvd(comp, predictive, alpha, n_grid=n_grid)
        tu_per_w[m] = v_w / (1 - tvd)
    return float(predictive.weights @ tu_per_w)


def quest_eu_local_c2(predictive: GaussianMixture1D, alpha: float) -> float:
    """C2 local EU as TU - AU (per Schweighofer framework)."""
    return quest_tu_local_c2(predictive, alpha) - quest_au_local_c2(predictive, alpha)


# ----- C3 plug-in: both predicting and truth-approx marginalized over posterior -----

def quest_tu_local_c3(
    predictive: GaussianMixture1D,
    alpha: float,
    n_grid: int = 5000,
) -> float:
    """C3 local TU: E_w[E_w_tilde[V_alpha(p_w) / (1 - TVD(p_w_alpha, p_w_tilde_alpha))]].
    
    Both predicting model and truth approximation are sampled from the posterior
    (i.e., from ensemble components). We exclude the diagonal m == m_tilde where
    TVD = 0 (which would inflate TU spuriously).
    
    Returns:
        Average TU over all M*(M-1) ordered pairs of distinct components.
    """
    M = predictive.M
    if M < 2:
        # Fallback: degenerate ensemble, TVD undefined; return AU
        return quest_au_local_c2(predictive, alpha)
    
    total = 0.0
    n_pairs = 0
    for m in range(M):
        comp_m = _component_distribution(predictive, m)
        for m_tilde in range(M):
            if m == m_tilde:
                continue
            comp_mt = _component_distribution(predictive, m_tilde)
            v_w, tvd = _v_alpha_and_tvd(comp_m, comp_mt, alpha, n_grid=n_grid)
            # Weighted by p(m) p(m_tilde); for uniform weights this is just averaging
            weight = predictive.weights[m] * predictive.weights[m_tilde]
            total += weight * v_w / (1 - tvd)
            n_pairs += 1
    
    # Normalize by sum of weights used (excludes diagonal)
    diag_weight = float(np.sum(predictive.weights ** 2))
    off_diag_weight = 1.0 - diag_weight
    if off_diag_weight < 1e-12:
        return quest_au_local_c2(predictive, alpha)
    return total / off_diag_weight


def quest_eu_local_c3(predictive: GaussianMixture1D, alpha: float) -> float:
    """C3 local EU as TU - AU."""
    return quest_tu_local_c3(predictive, alpha) - quest_au_local_c2(predictive, alpha)


# ----- Global versions of all three modes (integrate over alpha) -----

def quest_au_global_c2(predictive: GaussianMixture1D, n_alpha: int = 30) -> float:
    """C2 global AU: integral of E_w[V_alpha(p_w)] over alpha."""
    alphas = np.linspace(0.01, 0.99, n_alpha)
    vals = np.array([quest_au_local_c2(predictive, a) for a in alphas])
    return float(np.trapz(vals, alphas))


def quest_tu_global_c2(predictive: GaussianMixture1D, n_alpha: int = 30) -> float:
    alphas = np.linspace(0.01, 0.99, n_alpha)
    vals = np.array([quest_tu_local_c2(predictive, a) for a in alphas])
    return float(np.trapz(vals, alphas))


def quest_tu_global_c3(predictive: GaussianMixture1D, n_alpha: int = 30) -> float:
    alphas = np.linspace(0.01, 0.99, n_alpha)
    vals = np.array([quest_tu_local_c3(predictive, a) for a in alphas])
    return float(np.trapz(vals, alphas))


def quest_eu_global_c2(predictive: GaussianMixture1D, n_alpha: int = 30) -> float:
    return quest_tu_global_c2(predictive, n_alpha) - quest_au_global_c2(predictive, n_alpha)


def quest_eu_global_c3(predictive: GaussianMixture1D, n_alpha: int = 30) -> float:
    return quest_tu_global_c3(predictive, n_alpha) - quest_au_global_c2(predictive, n_alpha)


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
    
    # ---------- C2 / C3 plug-in sanity checks ----------
    print("\nC2/C3 plug-in TU sanity checks:")
    
    # Case 1: ensemble of identical components -> TU should equal AU
    # (all components agree -> no spread in beliefs -> EU = 0)
    identical = GaussianMixture1D(
        mus=np.array([0.0, 0.0, 0.0]), sigmas=np.array([1.0, 1.0, 1.0]),
    )
    au_c2 = quest_au_local_c2(identical, alpha=0.5)
    tu_c2 = quest_tu_local_c2(identical, alpha=0.5)
    tu_c3 = quest_tu_local_c3(identical, alpha=0.5)
    print(f"  Identical ensemble (M=3, all N(0,1)):")
    print(f"    AU (C2) = {au_c2:.4f}, TU (C2) = {tu_c2:.4f}, TU (C3) = {tu_c3:.4f}")
    print(f"    (All ~equal, since components agree perfectly.)")
    
    # Case 2: spread-out ensemble with different means -> TU > AU
    spread = GaussianMixture1D(
        mus=np.array([-2.0, 0.0, 2.0]), sigmas=np.array([1.0, 1.0, 1.0]),
    )
    au_c2 = quest_au_local_c2(spread, alpha=0.5)
    tu_c2 = quest_tu_local_c2(spread, alpha=0.5)
    tu_c3 = quest_tu_local_c3(spread, alpha=0.5)
    print(f"  Spread ensemble (mus = -2, 0, 2; all sigma=1):")
    print(f"    AU (C2) = {au_c2:.4f}, TU (C2) = {tu_c2:.4f}, TU (C3) = {tu_c3:.4f}")
    print(f"    (TU should both exceed AU; C3 typically inflates more than C2.)")
    
    # Case 3: more extreme disagreement
    extreme = GaussianMixture1D(
        mus=np.array([-5.0, 0.0, 5.0]), sigmas=np.array([0.5, 0.5, 0.5]),
    )
    au_c2 = quest_au_local_c2(extreme, alpha=0.5)
    tu_c2 = quest_tu_local_c2(extreme, alpha=0.5)
    tu_c3 = quest_tu_local_c3(extreme, alpha=0.5)
    print(f"  Extreme disagreement (mus = -5, 0, 5; sigma=0.5):")
    print(f"    AU (C2) = {au_c2:.4f}, TU (C2) = {tu_c2:.4f}, TU (C3) = {tu_c3:.4f}")
    print(f"    (TU should massively inflate AU.)")
