"""
Minimal test examples for uncertainty measures: QUEST, DifferentialEntropy, and PredictiveVariance.

These tests use simple Gaussian-based dummy models with known analytical properties
to verify correct computation of uncertainty decompositions.
"""

import numpy as np
import pytest
from modal_uq.uncertainty.quest import QUESTUncertainty
from modal_uq.uncertainty.variance import PredictiveVariance
from modal_uq.uncertainty.differential_entropy import DifferentialEntropy
from modal_uq.models.base import ModelBase
from modal_uq.models.ensemble import Ensemble


# ============================================================================
# DUMMY MODELS FOR TESTING
# ============================================================================

class GaussianDummyModel(ModelBase):
    """Deterministic model returning single Gaussian density [N,G]."""
    
    def __init__(self, mean=0.0, std=0.5, grid_points=512):
        super().__init__()
        self.mean = mean
        self.std = std
        self.grid_points = grid_points
        self._y_min = -3.0
        self._y_max = 3.0
    
    def fit(self, X, y, X_val=None, y_val=None):
        pass
    
    def predict_density(self, X, y_grid, context='predict'):
        """Return Gaussian density [N,G]."""
        n = len(X)
        # Standard Gaussian density
        dens = np.exp(-0.5 * ((y_grid[None, :] - self.mean) / self.std) ** 2)
        # Normalize
        dens = dens / (np.sqrt(2 * np.pi * self.std ** 2) + 1e-12)
        return np.repeat(dens, n, axis=0)

    def sample_output(self, X, n_samples, rng=None):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        samples = rng.normal(loc=self.mean, scale=self.std, size=(n_samples, N, 1))
        return samples

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        rng = np.random.default_rng(rng)
        ns = min(n_samples, 10000)
        samples = self.sample_output(X, ns, rng)
        lows = np.quantile(samples, q_low, axis=0)[:, 0]
        highs = np.quantile(samples, q_high, axis=0)[:, 0]
        pad = (highs - lows) * pad_frac
        bounds = np.stack([lows - pad, highs + pad], axis=-1)
        return bounds.reshape((-1, 1, 2))

    def density_function_for_input(self, X):
        def density(points, input_index=0):
            pts = np.asarray(points).reshape(-1)
            return np.exp(-0.5 * ((pts - self.mean) / self.std) ** 2) / (np.sqrt(2 * np.pi) * self.std)
        return density


class TwoGaussianMixtureDummyModel(ModelBase):
    """Stochastic model returning mixture of two Gaussians [S,N,G]."""
    
    def __init__(self, mean1=0.0, std1=0.3, mean2=0.0, std2=0.7, grid_points=512):
        super().__init__()
        self.mean1 = mean1
        self.std1 = std1
        self.mean2 = mean2
        self.std2 = std2
        self.grid_points = grid_points
        self._y_min = -3.0
        self._y_max = 3.0
    
    def fit(self, X, y, X_val=None, y_val=None):
        pass
    
    def predict_density(self, X, y_grid, context='predict'):
        """Return mixture of two Gaussians [S=2,N,G]."""
        n = len(X)
        
        # First Gaussian
        dens1 = np.exp(-0.5 * ((y_grid[None, :] - self.mean1) / self.std1) ** 2)
        dens1 = dens1 / (np.sqrt(2 * np.pi * self.std1 ** 2) + 1e-12)
        dens1 = np.repeat(dens1, n, axis=0)  # [N,G]
        
        # Second Gaussian
        dens2 = np.exp(-0.5 * ((y_grid[None, :] - self.mean2) / self.std2) ** 2)
        dens2 = dens2 / (np.sqrt(2 * np.pi * self.std2 ** 2) + 1e-12)
        dens2 = np.repeat(dens2, n, axis=0)  # [N,G]
        
        # Stack: 0.5 * dens1 + 0.5 * dens2 are each [N,G], stack to [S=2,N,G]
        return np.stack([dens1, dens2], axis=0)

    def sample_output(self, X, n_samples, rng=None):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        # For each draw choose component with p=0.5
        comps = rng.integers(0, 2, size=(n_samples, N))
        samples = np.zeros((n_samples, N, 1))
        for s in range(n_samples):
            # draw for component 0
            mask = comps[s] == 0
            if mask.any():
                samples[s, mask, 0] = rng.normal(loc=self.mean1, scale=self.std1, size=mask.sum())
            # draw for component 1
            mask1 = comps[s] == 1
            if mask1.any():
                samples[s, mask1, 0] = rng.normal(loc=self.mean2, scale=self.std2, size=mask1.sum())
        return samples

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        ns = min(n_samples, 10000)
        samples = self.sample_output(X, ns, rng)
        lows = np.quantile(samples, q_low, axis=0)[:, 0]
        highs = np.quantile(samples, q_high, axis=0)[:, 0]
        pad = (highs - lows) * pad_frac
        return np.stack([lows - pad, highs + pad], axis=-1).reshape((-1, 1, 2))

    def density_function_for_input(self, X):
        def density(points, input_index=0):
            pts = np.asarray(points).reshape(-1)
            p1 = np.exp(-0.5 * ((pts - self.mean1) / self.std1) ** 2) / (np.sqrt(2 * np.pi) * self.std1)
            p2 = np.exp(-0.5 * ((pts - self.mean2) / self.std2) ** 2) / (np.sqrt(2 * np.pi) * self.std2)
            return 0.5 * (p1 + p2)
        return density


class ContextAwareDummyModel(ModelBase):
    """Model with identical densities for both contexts (epistemic uncertainty = 0)."""
    
    def __init__(self, mean=0.0, std=0.5, grid_points=512):
        super().__init__()
        self.mean = mean
        self.std = std
        self.grid_points = grid_points
        self._y_min = -3.0
        self._y_max = 3.0
    
    def fit(self, X, y, X_val=None, y_val=None):
        pass
    
    def predict_density(self, X, y_grid, context='predict'):
        """Return same Gaussian for both contexts [N,G]."""
        n = len(X)
        dens = np.exp(-0.5 * ((y_grid[None, :] - self.mean) / self.std) ** 2)
        dens = dens / (np.sqrt(2 * np.pi * self.std ** 2) + 1e-12)
        return np.repeat(dens, n, axis=0)

    def sample_output(self, X, n_samples, rng=None):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        samples = rng.normal(loc=self.mean, scale=self.std, size=(n_samples, N, 1))
        return samples

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        ns = min(n_samples, 10000)
        samples = self.sample_output(X, ns, rng)
        lows = np.quantile(samples, q_low, axis=0)[:, 0]
        highs = np.quantile(samples, q_high, axis=0)[:, 0]
        pad = (highs - lows) * pad_frac
        return np.stack([lows - pad, highs + pad], axis=-1).reshape((-1, 1, 2))

    def density_function_for_input(self, X):
        def density(points, input_index=0):
            pts = np.asarray(points).reshape(-1)
            return np.exp(-0.5 * ((pts - self.mean) / self.std) ** 2) / (np.sqrt(2 * np.pi) * self.std)
        return density


class Bivariate2DGaussianDummyModel(ModelBase):
    """Deterministic model returning 2D bivariate Gaussian density on a 2D grid."""
    
    def __init__(self, mean=np.array([0.0, 0.0]), cov=None, grid_points=64):
        super().__init__()
        self.mean = np.asarray(mean)
        if cov is None:
            cov = np.eye(2)
        self.cov = np.asarray(cov)
        self.grid_points = grid_points
        self._y_min = -3.0
        self._y_max = 3.0
    
    def fit(self, X, y, X_val=None, y_val=None):
        pass
    
    def predict_density(self, X, y_grid, context='predict'):
        """Return 2D bivariate Gaussian density on y_grid.
        
        Parameters
        ----------
        X : array of shape (N, D)
            Input features (not used; density is fixed)
        y_grid : array of shape (2, G) or tuple of 2 arrays
            2D grid: y_grid[0] = y1 values, y_grid[1] = y2 values
        
        Returns
        -------
        dens : array of shape (N, G, G)
            Density values on the 2D grid for each sample
        """
        n = len(X)
        
        # Handle grid format: convert to meshgrid if needed
        if isinstance(y_grid, (list, tuple)) and len(y_grid) == 2:
            y1, y2 = y_grid
        else:
            # Assume y_grid is [2, G]
            y1, y2 = y_grid[0], y_grid[1]
        
        Y1, Y2 = np.meshgrid(y1, y2, indexing='ij')
        Y_stack = np.stack([Y1, Y2], axis=-1)  # shape (G, G, 2)
        
        # Compute 2D Gaussian density at each grid point
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
        
        dens_2d = np.zeros((Y_stack.shape[0], Y_stack.shape[1]))
        for i in range(Y_stack.shape[0]):
            for j in range(Y_stack.shape[1]):
                diff = Y_stack[i, j] - self.mean
                dens_2d[i, j] = norm_const * np.exp(-0.5 * (diff @ inv_cov @ diff))
        
        # Repeat for each sample: (G, G) -> (N, G, G)
        dens = np.repeat(dens_2d[None, :, :], n, axis=0)
        return dens
    
    def default_y_grid(self, X, grid_points=64, y_pad=1.0):
        """Return 2D grid for bivariate output."""
        lo, hi = -3.0, 3.0
        pad = y_pad * (hi - lo)
        y = np.linspace(lo - pad, hi + pad, grid_points)
        return np.stack([y, y], axis=0)  # shape (2, grid_points)

    def sample_output(self, X, n_samples, rng=None):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        samples = rng.multivariate_normal(mean=self.mean, cov=self.cov, size=(n_samples,))
        # samples shape (n_samples, d) -> expand to (n_samples, N, d)
        return np.repeat(samples[:, None, :], N, axis=1)

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        ns = min(n_samples, 10000)
        samples = self.sample_output(X, ns, rng)  # (ns, N, d)
        lows = np.quantile(samples, q_low, axis=0)  # (N, d)
        highs = np.quantile(samples, q_high, axis=0)
        pad = (highs - lows) * pad_frac
        bounds = np.stack([lows - pad, highs + pad], axis=-1)  # (N, d, 2)
        return bounds

    def density_function_for_input(self, X):
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        norm_const = 1.0 / (np.sqrt((2.0 * np.pi) ** self.mean.size * det_cov))

        def density(points, input_index=0):
            pts = np.atleast_2d(points)
            if pts.shape[1] != self.mean.size:
                pts = pts.reshape(-1, self.mean.size)
            diff = pts - self.mean
            exponent = -0.5 * np.sum((diff @ inv_cov) * diff, axis=1)
            return norm_const * np.exp(exponent)

        return density


class Bivariate2DGaussianMixtureDummyModel(ModelBase):
    """2D bimodal mixture of two Gaussians for testing HDR on disconnected regions."""
    
    def __init__(self, mean1=np.array([-5.0, 0.0]), mean2=np.array([5.0, 0.0]), 
                 cov=None, weight=0.5, grid_points=64):
        """
        Parameters
        ----------
        mean1, mean2 : array of shape (2,)
            Means of the two Gaussian components
        cov : array of shape (2, 2), optional
            Shared covariance matrix (default: identity)
        weight : float
            Mixture weight for first component (1-weight for second)
        grid_points : int
            Grid resolution for density computation
        """
        super().__init__()
        self.mean1 = np.asarray(mean1)
        self.mean2 = np.asarray(mean2)
        if cov is None:
            cov = np.eye(2)
        self.cov = np.asarray(cov)
        self.weight = weight
        self.grid_points = grid_points
        self._y_min = -3.0
        self._y_max = 3.0
    
    def fit(self, X, y, X_val=None, y_val=None):
        pass
    
    def predict_density(self, X, y_grid, context='predict'):
        """Return 2D bimodal Gaussian mixture density on y_grid.
        
        Returns
        -------
        dens : array of shape (N, G, G)
            Density values on the 2D grid for each sample
        """
        n = len(X)
        
        # Handle grid format
        if isinstance(y_grid, (list, tuple)) and len(y_grid) == 2:
            y1, y2 = y_grid
        else:
            y1, y2 = y_grid[0], y_grid[1]
        
        Y1, Y2 = np.meshgrid(y1, y2, indexing='ij')
        Y_stack = np.stack([Y1, Y2], axis=-1)  # shape (G, G, 2)
        
        # Compute mixture of two 2D Gaussians
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
        
        dens_2d = np.zeros((Y_stack.shape[0], Y_stack.shape[1]))
        for i in range(Y_stack.shape[0]):
            for j in range(Y_stack.shape[1]):
                diff1 = Y_stack[i, j] - self.mean1
                diff2 = Y_stack[i, j] - self.mean2
                dens_2d[i, j] = self.weight * norm_const * np.exp(-0.5 * (diff1 @ inv_cov @ diff1)) + \
                                (1 - self.weight) * norm_const * np.exp(-0.5 * (diff2 @ inv_cov @ diff2))
        
        # Repeat for each sample
        dens = np.repeat(dens_2d[None, :, :], n, axis=0)
        return dens
    
    def default_y_grid(self, X, grid_points=64, y_pad=1.0):
        """Return 2D grid for bivariate output."""
        lo, hi = -10.0, 10.0
        pad = y_pad * (hi - lo)
        y = np.linspace(lo - pad, hi + pad, grid_points)
        return np.stack([y, y], axis=0)

    def sample_output(self, X, n_samples, rng=None):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        
        # Sample from mixture: choose component then draw
        component = rng.uniform(size=(n_samples,)) < self.weight
        samples = np.empty((n_samples, N, 2))
        
        # Component 1
        if component.any():
            samples[component] = rng.multivariate_normal(
                mean=self.mean1, cov=self.cov, size=component.sum()
            )[:, None, :].repeat(N, axis=1)
        
        # Component 2
        if (~component).any():
            samples[~component] = rng.multivariate_normal(
                mean=self.mean2, cov=self.cov, size=(~component).sum()
            )[:, None, :].repeat(N, axis=1)
        
        return samples

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        ns = min(n_samples, 10000)
        samples = self.sample_output(X, ns, rng)  # (ns, N, 2)
        lows = np.quantile(samples, q_low, axis=0)  # (N, 2)
        highs = np.quantile(samples, q_high, axis=0)
        pad = (highs - lows) * pad_frac
        bounds = np.stack([lows - pad, highs + pad], axis=-1)  # (N, 2, 2)
        return bounds

    def density_function_for_input(self, X):
        """Return callable that evaluates the mixture density at arbitrary 2D points."""
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        norm_const = 1.0 / (np.sqrt((2.0 * np.pi) ** 2 * det_cov))

        def density(points, input_index=0):
            pts = np.atleast_2d(points)
            if pts.shape[1] != 2:
                pts = pts.reshape(-1, 2)
            diff1 = pts - self.mean1
            diff2 = pts - self.mean2
            exp1 = -0.5 * np.sum((diff1 @ inv_cov) * diff1, axis=1)
            exp2 = -0.5 * np.sum((diff2 @ inv_cov) * diff2, axis=1)
            return self.weight * norm_const * np.exp(exp1) + \
                   (1 - self.weight) * norm_const * np.exp(exp2)

        return density


class MinimalMockEnsemble(Ensemble):
    """Minimal mock Ensemble for testing QUEST epistemic uncertainty."""
    
    def __init__(self, n_members=3, grid_points=512):
        super().__init__()
        self.n_members = n_members
        self.grid_points = grid_points
        self._y_min = -3.0
        self._y_max = 3.0
    
    def fit(self, X, y, X_val=None, y_val=None):
        pass
    
    def predict_density(self, X, y_grid, context='predict'):
        """Return [N,G] for both contexts (simplified)."""
        n = len(X)
        dens = np.exp(-0.5 * ((y_grid[None, :] - 0.0) / 0.5) ** 2)
        dens = dens / (np.sqrt(2 * np.pi * 0.5 ** 2) + 1e-12)
        return np.repeat(dens, n, axis=0)
    
    def get_second_order_distribution(self, X, n_mc_samples=100000, random_state=None):
        """Return KDE + MC samples for each input (new MC-only format).
        
        Parameters
        ----------
        X : array
            Input features
        n_mc_samples : int
            Number of MC samples to draw from each KDE
        random_state : int or Generator
            Random state for reproducibility
            
        Returns
        -------
        kdes_list : list of scipy.stats.gaussian_kde
            One fitted KDE per input
        samples_list : list of arrays
            Pre-sampled arrays from KDE, shape (n_mc_samples, 1) per input
        """
        from scipy.stats import gaussian_kde
        rng = np.random.default_rng(random_state)
        n = len(X)
        
        kdes_list = []
        samples_list = []
        
        for i in range(n):
            # Create a simple Gaussian KDE over a 1D parameter space
            # Sample 5 parameter values representing ensemble member parameters
            member_params = np.linspace(-1.0, 1.0, self.n_members)
            
            try:
                kde = gaussian_kde(member_params)
            except Exception as e:
                raise RuntimeError(f"Error fitting KDE for sample {i}: {e}")
            
            # Pre-sample from KDE
            # Note: gaussian_kde.resample doesn't accept random_state
            mc_samples = kde.resample(size=n_mc_samples)  # (1, n_mc_samples)
            mc_samples = mc_samples.T  # -> (n_mc_samples, 1)
            
            kdes_list.append(kde)
            samples_list.append(mc_samples)
        
        return kdes_list, samples_list

    def sample_output(self, X, n_samples, rng=None):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        samples = rng.normal(loc=0.0, scale=0.5, size=(n_samples, N, 1))
        return samples

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        ns = min(n_samples, 10000)
        samples = self.sample_output(X, ns, rng)
        lows = np.quantile(samples, q_low, axis=0)[:, 0]
        highs = np.quantile(samples, q_high, axis=0)[:, 0]
        pad = (highs - lows) * pad_frac
        return np.stack([lows - pad, highs + pad], axis=-1).reshape((-1, 1, 2))

    def density_function_for_input(self, X):
        def density(points, input_index=0):
            pts = np.asarray(points).reshape(-1)
            return np.exp(-0.5 * (pts / 0.5) ** 2) / (np.sqrt(2 * np.pi) * 0.5)
        return density


# ============================================================================
# PREDICTIVE VARIANCE TESTS
# ============================================================================

def test_variance_shape_and_finiteness():
    """Test that variance returns correct shape and finite values."""
    model = GaussianDummyModel()
    X = np.array([[0.0]])  # Single sample
    
    variance = PredictiveVariance(decomposition='total', grid_points=512)
    scores = variance.score(model, X)
    
    assert scores.shape == (1,), f"Expected shape (1,), got {scores.shape}"
    assert np.isfinite(scores).all(), "Expected all finite values"
    assert scores[0] > 0, "Variance should be positive"


def test_variance_batch_processing():
    """Test variance computation on batch of multiple samples."""
    model = GaussianDummyModel()
    X = np.random.randn(5, 1)  # 5 samples
    
    variance = PredictiveVariance(decomposition='total', grid_points=512)
    scores = variance.score(model, X)
    
    assert scores.shape == (5,), f"Expected shape (5,), got {scores.shape}"
    assert np.isfinite(scores).all(), "Expected all finite values"
    assert (scores > 0).all(), "All variances should be positive"


def test_variance_stochastic_aggregation():
    """Test that stochastic model (S>1) is properly aggregated over S dimension."""
    model = TwoGaussianMixtureDummyModel()
    X = np.array([[0.0], [1.0]])  # 2 samples
    
    variance = PredictiveVariance(decomposition='total', grid_points=256)
    scores = variance.score(model, X)
    
    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert np.isfinite(scores).all(), "Expected all finite values"
    # Mixture should have higher variance than single Gaussian
    assert scores[0] > 0.1, "Mixture variance should be substantial"


def test_variance_decomposition_consistency():
    """Test law of total variance when predict and approximate are identical.
    
    When predict context = approximate context (same distributions),
    we expect: total ≈ aleatoric, epistemic ≈ 0
    """
    model = ContextAwareDummyModel(std=0.4)
    X = np.array([[0.0], [0.5], [-0.5]])  # 3 samples
    
    var_total = PredictiveVariance(decomposition='total', grid_points=256)
    var_aleatoric = PredictiveVariance(decomposition='aleatoric', grid_points=256)
    
    total = var_total.score(model, X)
    aleatoric = var_aleatoric.score(model, X)
    
    # Should be very close (numerical integration tolerance 1e-6)
    np.testing.assert_allclose(total, aleatoric, rtol=1e-6, atol=1e-6,
                                err_msg="total ≈ aleatoric when contexts are identical")


def test_variance_positive_definite():
    """Test that all variance values are non-negative."""
    model = GaussianDummyModel(mean=0.5, std=0.3)
    X = np.random.randn(10, 1)
    
    variance = PredictiveVariance(decomposition='total', grid_points=512)
    scores = variance.score(model, X)
    
    assert (scores >= 0).all(), "Variance must be non-negative"


def test_variance_computation_correctness():
    """Verify variance matches theoretical value for standard normal."""
    # N(0, 1) has variance exactly 1
    model = GaussianDummyModel(mean=0.0, std=1.0)
    X = np.array([[0.0]])
    
    variance = PredictiveVariance(decomposition='total', grid_points=1024)
    scores = variance.score(model, X)
    
    # Should be close to 1.0 (numerical integration error)
    np.testing.assert_allclose(scores[0], 1.0, rtol=1e-2, atol=1e-3,
                                err_msg="Variance of N(0,1) should be ~1.0")


# ============================================================================
# DIFFERENTIAL ENTROPY TESTS
# ============================================================================

def test_entropy_shape_and_finiteness():
    """Test that entropy returns correct shape and finite values."""
    model = GaussianDummyModel()
    X = np.array([[0.0]])
    
    entropy = DifferentialEntropy(base=np.e, decomposition='total', grid_points=512)
    scores = entropy.score(model, X)
    
    assert scores.shape == (1,), f"Expected shape (1,), got {scores.shape}"
    assert np.isfinite(scores).all(), "Expected all finite values"
    assert scores[0] > 0, "Entropy should be positive"


def test_entropy_different_bases():
    """Test entropy computation with different bases and verify conversion.
    
    H_base2(p) = H_e(p) / ln(2)
    """
    model = GaussianDummyModel()
    X = np.array([[0.0], [1.0]])
    
    entropy_e = DifferentialEntropy(base=np.e, decomposition='total', grid_points=512)
    entropy_2 = DifferentialEntropy(base=2.0, decomposition='total', grid_points=512)
    
    scores_e = entropy_e.score(model, X)
    scores_2 = entropy_2.score(model, X)
    
    # Verify conversion formula
    np.testing.assert_allclose(scores_2, scores_e / np.log(2.0), rtol=1e-6, atol=1e-6,
                                err_msg="H_base2 = H_e / ln(2)")


def test_entropy_decomposition_consistency():
    """Test decomposition when contexts are identical: total ≈ aleatoric, epistemic ≈ 0."""
    model = ContextAwareDummyModel(std=0.6)
    X = np.array([[0.0], [0.2]])
    
    entropy_total = DifferentialEntropy(base=np.e, decomposition='total', grid_points=256)
    entropy_aleatoric = DifferentialEntropy(base=np.e, decomposition='aleatoric', grid_points=256)
    entropy_epistemic = DifferentialEntropy(base=np.e, decomposition='epistemic', grid_points=256)
    
    total = entropy_total.score(model, X)
    aleatoric = entropy_aleatoric.score(model, X)
    epistemic = entropy_epistemic.score(model, X)
    
    # total ≈ aleatoric
    np.testing.assert_allclose(total, aleatoric, rtol=1e-6, atol=1e-6,
                                err_msg="total ≈ aleatoric when contexts are identical")
    
    # epistemic ≈ 0
    np.testing.assert_allclose(epistemic, np.zeros_like(epistemic), rtol=1e-6, atol=1e-6,
                                err_msg="epistemic ≈ 0 when contexts are identical")


def test_entropy_kl_divergence_properties():
    """Test KL divergence properties via epistemic entropy calculation.
    
    Properties:
    - KL(p||q) >= 0
    - KL(p||p) = 0
    - KL is asymmetric: KL(p||q) != KL(q||p) in general
    """
    entropy = DifferentialEntropy(base=np.e)
    y_grid = np.linspace(-5, 5, 512)
    
    # Create two different Gaussian densities
    p = np.exp(-0.5 * ((y_grid - 0.0) / 1.0) ** 2)
    p = p / np.trapz(p, y_grid)
    
    q = np.exp(-0.5 * ((y_grid - 0.0) / 2.0) ** 2)
    q = q / np.trapz(q, y_grid)
    
    # Test KL(p||p) ≈ 0
    kl_self = entropy._kl_divergence(p, p, y_grid)
    assert kl_self >= 0 and kl_self < 1e-6, f"KL(p||p) should be ~0, got {kl_self}"
    
    # Test KL(p||q) > 0
    kl_pq = entropy._kl_divergence(p, q, y_grid)
    assert kl_pq > 0, "KL(p||q) should be positive when distributions differ"
    
    # Test KL(q||p) > 0
    kl_qp = entropy._kl_divergence(q, p, y_grid)
    assert kl_qp > 0, "KL(q||p) should be positive when distributions differ"
    
    # Test asymmetry: KL(p||q) != KL(q||p)
    assert not np.isclose(kl_pq, kl_qp, rtol=1e-3), "KL should be asymmetric"


def test_entropy_correctness_gaussian():
    """Verify entropy of Gaussian matches theoretical value.
    
    For N(μ, σ²): H = 0.5 * ln(2πeσ²)
    """
    sigma = 0.8
    model = GaussianDummyModel(mean=0.0, std=sigma)
    X = np.array([[0.0]])
    
    entropy = DifferentialEntropy(base=np.e, decomposition='total', grid_points=1024)
    computed = entropy.score(model, X)[0]
    
    # Theoretical entropy
    theoretical = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
    
    # Should match within numerical integration tolerance
    np.testing.assert_allclose(computed, theoretical, rtol=1e-2, atol=1e-3,
                                err_msg=f"Entropy should match theory: computed={computed}, theory={theoretical}")


# ============================================================================
# QUEST TESTS
# ============================================================================

def test_quest_alpha_volume_shape():
    """Test that alpha volume returns correct shape and non-negative values."""
    model = GaussianDummyModel()
    X = np.array([[0.0], [0.5]])  # 2 samples
    
    quest = QUESTUncertainty(alpha=0.1, decomposition='aleatoric', scope='local', grid_points=512)
    scores = quest.score(model, X)
    
    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert np.isfinite(scores).all(), "Expected all finite values"
    assert (scores >= 0).all(), "Alpha volume should be non-negative"


def test_quest_alpha_volume_monotonicity():
    """Test that alpha volume decreases as alpha increases.
    
    Higher alpha means larger tail probability excluded, so HDR gets smaller.
    """
    model = GaussianDummyModel(mean=0.0, std=1.0)
    X = np.array([[0.0]])
    
    alphas = [0.05, 0.1, 0.2]
    volumes = []
    
    for alpha in alphas:
        quest = QUESTUncertainty(alpha=alpha, decomposition='aleatoric', scope='local', grid_points=512)
        volume = quest.score(model, X)[0]
        volumes.append(volume)
    
    # Verify monotonic decrease
    for i in range(len(volumes) - 1):
        assert volumes[i] >= volumes[i + 1], \
            f"Alpha volume should decrease with alpha: {volumes}"

def test_hdr_2d_bimodal_separated_gaussians():
    """
    Test HDR behaviour for a 2D bimodal distribution (well-separated Gaussians).

    This verifies:
    1. Correct HDR mass (≈ 1 - alpha)
    2. Proper handling of disconnected regions (multimodality)
    3. Reasonable Lebesgue volume estimation matching theoretical bounds

    Geometry:
    - Two 2D Gaussians at (-d, 0) and (+d, 0) with identity covariance
    - Equal mixture weights (0.5 each)
    - Strong separation (d=5.0) ensures bimodality

    Expected behaviour:
    - HDR splits into two disjoint regions
    - Each region resembles a single Gaussian HDR
    - Total volume ≈ 2 × single Gaussian HDR volume (within tolerance)
    """
    from scipy.stats import chi2
    
    # Configuration
    alpha = 0.1
    n_mc_samples = 100000  # MC samples for HDR computation
    d_sep = 5.0  # Strong separation to ensure bimodal structure
    
    # Create 2D bimodal mixture model
    model = Bivariate2DGaussianMixtureDummyModel(
        mean1=np.array([-d_sep, 0.0]),
        mean2=np.array([d_sep, 0.0]),
        cov=np.eye(2),
        weight=0.5,
        grid_points=64
    )
    
    # Single test input
    X = np.array([[0.0, 0.0]])
    
    # --- Step 1: Compute HDR volume using QUEST API ---
    quest = QUESTUncertainty(
        alpha=alpha,
        decomposition='aleatoric',
        scope='local',
        mc_n_samples=n_mc_samples,
        mc_random_state=0
    )
    volume_est = quest.score(model, X)[0]
    
    # Sanity checks
    assert np.isfinite(volume_est), "Volume is not finite"
    assert volume_est > 0, "Volume must be positive"
    
    # --- Step 2: HDR mass verification via internal method (test infrastructure) ---
    # Manually compute HDR to verify mass correctness and cluster separation
    rng = np.random.default_rng(0)
    
    # Sample from the model
    X_mc = model.sample_output(X, n_mc_samples, rng)  # shape (n_mc_samples, 1, 2)
    X_mc_flat = X_mc[:, 0, :]  # shape (n_mc_samples, 2)
    
    # Evaluate density at samples
    density_func = model.density_function_for_input(X)
    densities = density_func(X_mc_flat, input_index=0)  # shape (n_mc_samples,)
    
    # Find HDR threshold
    c_alpha = np.quantile(densities, alpha)
    mask = densities >= c_alpha
    
    # Mass check (critical correctness condition)
    mass_est = mask.mean()
    assert abs(mass_est - (1 - alpha)) < 0.02, \
        f"HDR mass incorrect: expected {1-alpha:.3f}, got {mass_est:.3f}"
    
    # --- Step 3: Multimodality verification (CRITICAL) ---
    # Verify that HDR samples cluster into two distinct groups
    hdr_points = X_mc_flat[mask]  # points in HDR
    
    # Split by x-axis (modes are at x = ±d_sep)
    left_cluster = hdr_points[:, 0] < 0
    right_cluster = hdr_points[:, 0] >= 0
    
    left_frac = left_cluster.mean()
    right_frac = right_cluster.mean()
    
    assert left_frac > 0.3 and right_frac > 0.3, \
        f"HDR not clearly bimodal: left={left_frac:.3f}, right={right_frac:.3f} (expected both > 0.3)"
    
    # --- Step 4: Theoretical volume validation ---
    # For 2D Gaussian, HDR volume at confidence level (1-alpha) is:
    # V(alpha) = π * χ²_quantile(1-alpha, df=2)
    # For bimodal mixture with well-separated modes, total volume ≈ 2 × V_single
    
    chi2_quantile = chi2.ppf(1 - alpha, df=2)
    expected_single_mode = np.pi * chi2_quantile
    expected_bimodal = 2.0 * expected_single_mode
    
    # Allow 15% tolerance for MC sampling and numerical integration errors
    tolerance_frac = 0.15
    lower_bound = expected_bimodal * (1 - tolerance_frac)
    upper_bound = expected_bimodal * (1 + tolerance_frac)
    
    assert lower_bound < volume_est < upper_bound, \
        f"Volume out of expected range: expected {expected_bimodal:.3f} ± {tolerance_frac*100:.0f}%, got {volume_est:.3f}"


def test_quest_integrated_volume():
    """Test integrated volume (global scope) computation."""
    model = GaussianDummyModel(mean=0.0, std=0.5)
    X = np.array([[0.0], [0.3]])  # 2 samples
    
    quest = QUESTUncertainty(decomposition='aleatoric', scope='global', grid_points=256)
    scores = quest.score(model, X)
    
    assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
    assert np.isfinite(scores).all(), "Expected all finite values"
    assert (scores >= 0).all(), "Integrated volume should be non-negative"


def test_quest_integrated_volume_gaussian_closed_form():
    """Test QUEST integrated volume against the closed-form standard Gaussian value.

    For a standard normal density, the HDR length at tail probability alpha is
    2 * Phi^{-1}(1 - alpha / 2), and integrating this over alpha in [0, 1]
    gives 4 / sqrt(2*pi).
    """
    model = GaussianDummyModel(mean=0.0, std=1.0)
    X = np.array([[0.0]])

    quest = QUESTUncertainty(decomposition='aleatoric', scope='global', grid_points=4096)
    score = quest.score(model, X)[0]

    expected = 4.0 / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(score, expected, rtol=3e-2, atol=3e-2,
                                err_msg=f"Gaussian QUEST IV should be close to {expected}, got {score}")


def test_quest_integrated_volume_2d_bivariate_gaussian():
    """Test QUEST integrated volume for 2D bivariate standard normal.
    
    For a 2D standard normal, the HDR at tail probability alpha is an ellipse
    with area proportional to chi2_quantile(1-alpha, df=2). Integrating over
    alpha gives approximately π (the limiting case as alpha → 1).
    """
    model = Bivariate2DGaussianDummyModel(
        mean=np.array([0.0, 0.0]),
        cov=np.eye(2),
        grid_points=64
    )
    X = np.array([[0.0, 0.0]])  # Single 2D sample
    
    quest = QUESTUncertainty(decomposition='aleatoric', scope='global', grid_points=64)
    score = quest.score(model, X)[0]
    
    # For 2D standard normal, the IV should be approximately 2π
    # (numerical integration; exact value depends on grid resolution)
    expected = 2.0 * np.pi
    np.testing.assert_allclose(score, expected, rtol=0.1, atol=0.1,
                                err_msg=f"2D Gaussian QUEST IV should be close to 2π~{expected}, got {score}")


def test_quest_hdr_on_standard_gaussian():
    """Test HDR computation on standard normal N(0,1).
    
    With alpha=0.05 (95% mass), HDR should be approximately [-1.96, 1.96].
    Lebesgue measure (total length) should be roughly 3.92.
    """
    model = GaussianDummyModel(mean=0.0, std=1.0)
    X = np.array([[0.0]])
    
    alpha = 0.05
    quest = QUESTUncertainty(alpha=alpha, decomposition='aleatoric', scope='local', grid_points=1024)
    volume = quest.score(model, X)[0]
    
    # For 95% HDR of N(0,1), expect length ~3.92 (from -1.96 to 1.96)
    # Allow some numerical integration tolerance
    expected = 3.92
    np.testing.assert_allclose(volume, expected, rtol=0.15, atol=0.2,
                                err_msg=f"HDR of N(0,1) should be ~3.92, got {volume}")


def test_quest_epistemic_requires_ensemble():
    """Test that epistemic QUEST computation works with mock Ensemble."""
    ensemble = MinimalMockEnsemble(n_members=3)
    X = np.array([[0.0]])
    
    quest = QUESTUncertainty(alpha=0.1, decomposition='epistemic', scope='local', grid_points=256)
    scores = quest.score(ensemble, X)
    
    assert scores.shape == (1,), f"Expected shape (1,), got {scores.shape}"
    assert np.isfinite(scores).all(), "Expected all finite values"
    assert (scores >= 0).all(), "Epistemic uncertainty should be non-negative"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_cross_measure_consistency():
    """Test that all three uncertainty measures work on same model without errors."""
    model = GaussianDummyModel(mean=0.0, std=0.7)
    X = np.random.randn(3, 1)
    
    # Variance
    variance = PredictiveVariance(decomposition='total', grid_points=256)
    var_scores = variance.score(model, X)
    
    # Entropy
    entropy = DifferentialEntropy(base=np.e, decomposition='total', grid_points=256)
    ent_scores = entropy.score(model, X)
    
    # QUEST
    quest = QUESTUncertainty(alpha=0.1, decomposition='aleatoric', scope='local', grid_points=256)
    quest_scores = quest.score(model, X)
    
    # All should return same shape
    assert var_scores.shape == (3,)
    assert ent_scores.shape == (3,)
    assert quest_scores.shape == (3,)
    
    # All should be finite
    assert np.isfinite(var_scores).all()
    assert np.isfinite(ent_scores).all()
    assert np.isfinite(quest_scores).all()


def test_batch_heterogeneity():
    """Test that measures capture variation across samples with different uncertainty levels.
    
    Create a scenario where different inputs have naturally different uncertainty,
    and verify that all three measures capture this variation.
    """
    # Use custom model that varies uncertainty by input
    class HeterogeneousDummyModel(ModelBase):
        def fit(self, X, y, X_val=None, y_val=None):
            pass
        
        def predict_density(self, X, y_grid, context='predict'):
            n = len(X)
            dens = np.zeros((n, len(y_grid)))
            for i in range(n):
                # Vary std by input: x[0] controls uncertainty
                x_val = X[i, 0]
                std = 0.2 + 0.5 * np.abs(x_val)  # Higher x → higher uncertainty
                d = np.exp(-0.5 * ((y_grid - 0.0) / std) ** 2)
                d = d / (np.sqrt(2 * np.pi * std ** 2) + 1e-12)
                dens[i] = d
            return dens

        def sample_output(self, X, n_samples, rng=None):
            rng = np.random.default_rng(rng)
            N = X.shape[0]
            samples = np.zeros((n_samples, N, 1))
            for i in range(N):
                std = 0.2 + 0.5 * np.abs(X[i, 0])
                samples[:, i, 0] = rng.normal(loc=0.0, scale=std, size=n_samples)
            return samples

        def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
            rng = np.random.default_rng(rng)
            samples = self.sample_output(X, min(n_samples, 10000), rng)
            lows = np.quantile(samples, q_low, axis=0)[:, 0]
            highs = np.quantile(samples, q_high, axis=0)[:, 0]
            pad = (highs - lows) * pad_frac
            return np.stack([lows - pad, highs + pad], axis=-1).reshape((-1, 1, 2))

        def density_function_for_input(self, X):
            stds = np.array([0.2 + 0.5 * np.abs(x[0]) for x in X])

            def density(points, input_index=0):
                pts = np.asarray(points).reshape(-1)
                std = stds[input_index]
                return np.exp(-0.5 * ((pts - 0.0) / std) ** 2) / (np.sqrt(2 * np.pi) * std)

            return density
    
    model = HeterogeneousDummyModel()
    X = np.array([[-1.0], [0.0], [1.0]])  # Varying inputs
    
    # Compute all measures
    variance = PredictiveVariance(decomposition='total', grid_points=256).score(model, X)
    entropy = DifferentialEntropy(base=np.e, decomposition='total', grid_points=256).score(model, X)
    quest = QUESTUncertainty(alpha=0.1, decomposition='aleatoric', scope='local', grid_points=256).score(model, X)
    
    # All should show variation (not all equal)
    var_unique = len(np.unique(np.round(variance, 4)))
    ent_unique = len(np.unique(np.round(entropy, 4)))
    quest_unique = len(np.unique(np.round(quest, 4)))
    
    assert var_unique > 1, "Variance should vary across samples"
    assert ent_unique > 1, "Entropy should vary across samples"
    assert quest_unique > 1, "QUEST should vary across samples"
    
    # Check monotonicity: higher |x| → higher uncertainty for all measures
    assert variance[0] <= variance[2], "Variance should increase with |x|"
    assert entropy[0] <= entropy[2], "Entropy should increase with |x|"
    assert quest[1] <= quest[0] + 5e-3, "QUEST should be higher away from the center"
    assert quest[1] <= quest[2] + 5e-3, "QUEST should be higher away from the center"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
