import numpy as np
from scipy.stats import norm, gaussian_kde

from modal_uq.uncertainty.quest import QUESTUncertainty
from modal_uq.models.ensemble import Ensemble


class StandardGaussianMCModel:
    def sample_output(self, X, n_samples, rng):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        return rng.normal(loc=0.0, scale=1.0, size=(n_samples, N, 1))

    def output_bounds(self, X, q_low=1e-3, q_high=1 - 1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        samples = self.sample_output(X, n_samples=min(n_samples, 10000), rng=rng)
        lows = np.quantile(samples, q_low, axis=0)[:, 0]
        highs = np.quantile(samples, q_high, axis=0)[:, 0]
        pad = (highs - lows) * pad_frac
        return np.stack([lows - pad, highs + pad], axis=-1).reshape((-1, 1, 2))

    def density_function_for_input(self, X):
        def density(points, input_index=0):
            pts = np.asarray(points).reshape(-1)
            return norm.pdf(pts)

        return density


class EpidemicTestEnsembleKDE(Ensemble):
    """Mock ensemble for testing epistemic QUEST with KDE + MC samples."""
    
    def __init__(self, n_members=5):
        super().__init__()
        self.n_members = n_members
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
        """Return KDE + MC samples for epistemic uncertainty (new MC-only format)."""
        rng = np.random.default_rng(random_state)
        n = len(X)
        
        kdes_list = []
        samples_list = []
        
        for i in range(n):
            # Create a simple 1D Gaussian KDE from mock member parameters
            member_params = np.linspace(-1.5, 1.5, self.n_members)
            
            try:
                kde = gaussian_kde(member_params)
            except Exception as e:
                raise RuntimeError(f"Error fitting KDE for sample {i}: {e}")
            
            # Pre-sample from KDE
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


def test_hdr_fraction_matches_target_mass():
    model = StandardGaussianMCModel()
    X = np.array([[0.0]])
    quest = QUESTUncertainty(alpha=0.1, decomposition='aleatoric', scope='local')
    quest.mc_n_samples = 20000

    density_func = model.density_function_for_input(X)
    sampler = lambda X_, n_samples, rng: model.sample_output(X_, n_samples, rng)
    c_alpha, hdr_indicator, samples, sample_mask = quest._hdr_from_density_function(
        density_func, sampler, X, alpha=0.1, n_samples=20000, random_state=0
    )

    observed = sample_mask.mean(axis=0)[0]
    assert np.isfinite(c_alpha[0])
    assert np.isclose(observed, 0.9, atol=0.02)


def test_mc_lebesgue_measure_is_reasonable_for_gaussian_hdr():
    model = StandardGaussianMCModel()
    X = np.array([[0.0]])
    quest = QUESTUncertainty(alpha=0.1, decomposition='aleatoric', scope='local')
    quest.mc_n_samples = 20000

    density_func = model.density_function_for_input(X)
    sampler = lambda X_, n_samples, rng: model.sample_output(X_, n_samples, rng)
    bounds = model.output_bounds(X, n_samples=20000, rng=0)

    c_alpha, _, _, _ = quest._hdr_from_density_function(
        density_func, sampler, X, alpha=0.1, n_samples=20000, random_state=0
    )
    volumes, fractions = quest._lebesgue_measure_hdr_mc(
        density_func, c_alpha, bounds, n_samples=20000, random_state=0
    )

    expected_length = 2.0 * norm.ppf(0.95)
    assert np.isfinite(volumes[0])
    assert np.isfinite(fractions[0])
    assert np.isclose(volumes[0], expected_length, rtol=0.15, atol=0.15)
    assert 0.0 <= fractions[0] <= 1.0


def test_quest_epistemic_kde_mc_integration():
    """Test epistemic QUEST with KDE + MC samples (parameter-space HDR)."""
    ensemble = EpidemicTestEnsembleKDE(n_members=5)
    X = np.array([[0.0], [1.0]])  # Two inputs
    
    quest = QUESTUncertainty(alpha=0.1, decomposition='epistemic', scope='local')
    quest.mc_n_samples = 10000
    
    # Compute epistemic scores using KDE + MC samples
    scores = quest.score(ensemble, X)
    
    # Verify output shape and finiteness
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    
    # Epistemic scores should be positive (HDR volume is positive)
    assert np.all(scores > 0)


def test_quest_epistemic_reproducibility():
    """Test that epistemic QUEST produces reproducible results with fixed seed."""
    ensemble = EpidemicTestEnsembleKDE(n_members=5)
    X = np.array([[0.5]])
    
    quest = QUESTUncertainty(alpha=0.1, decomposition='epistemic', scope='local')
    quest.mc_n_samples = 5000
    quest.mc_random_state = 42  # Fix random state
    
    # Run twice and compare
    scores1 = quest.score(ensemble, X)
    scores2 = quest.score(ensemble, X)
    
    # Results should be close (though not exactly equal due to KDE resampling)
    assert np.allclose(scores1, scores2, rtol=0.05, atol=0.05)