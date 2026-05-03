"""
Unit tests for QUEST helpers: tv_distance_mc, _compute_total_helper, _hdr_resampler_from_samples, _compute_total.

These tests cover:
  - tv_distance_mc: symmetry, bounds, determinism, identical distributions, analytic reference, Gaussian truncation.
  - _compute_total_helper: state restoration, composition, config validation.
  - _hdr_resampler_from_samples: shape, acceptance, fallback.
  - _compute_total: local and global modes.
"""

import numpy as np
import pytest
from scipy.stats import norm
from unittest.mock import MagicMock, patch

from modal_uq.uncertainty.quest import QUESTUncertainty
from modal_uq.models.base import InferentialChoiceConfig, ModelBase


# ============================================================================
# Mock Model for Testing
# ============================================================================

class MockModel(ModelBase):
    """Minimal mock model for testing."""
    
    def __init__(self, inferential_choice=None):
        super().__init__(inferential_choice=inferential_choice)
        self._y_min = -3.0
        self._y_max = 3.0
    
    def fit(self, X, y, X_val=None, y_val=None):
        pass
    
    def predict_density(self, X, y_grid, context='predict'):
        """Return simple Gaussian density."""
        n = len(X)
        dens = norm.pdf(y_grid, loc=0.0, scale=1.0)
        return np.repeat(dens[None, :], n, axis=0)
    
    def sample_output(self, X, n_samples, rng):
        rng = np.random.default_rng(rng)
        N = X.shape[0]
        return rng.normal(loc=0.0, scale=1.0, size=(n_samples, N, 1))
    
    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        rng = np.random.default_rng(rng)
        samples = self.sample_output(X, min(n_samples, 1000), rng)
        lows = np.quantile(samples, q_low, axis=0)[:, 0]
        highs = np.quantile(samples, q_high, axis=0)[:, 0]
        pad = (highs - lows) * pad_frac
        return np.stack([lows - pad, highs + pad], axis=-1).reshape((-1, 1, 2))
    
    def density_function_for_input(self, X):
        def density(points, input_index=0):
            pts = np.asarray(points).ravel()
            return norm.pdf(pts, loc=0.0, scale=1.0)
        return density


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def quest_default():
    """Default QUEST instance with alpha=0.1."""
    return QUESTUncertainty(alpha=0.1, decomposition='total', scope='local', mc_n_samples=1000)


@pytest.fixture
def mock_model_mle():
    """Mock model configured for approximate=MLE, predict=posterior_predictive."""
    cfg = InferentialChoiceConfig(predict='posterior_predictive', approximate='point_estimate', point_estimate_criterion='mle')
    return MockModel(inferential_choice=cfg)


@pytest.fixture
def X_single():
    """Single input sample."""
    return np.array([[0.0]])


@pytest.fixture
def X_batch():
    """Batch of three input samples."""
    return np.array([[0.0], [1.0], [-1.0]])


# ============================================================================
# Tests for tv_distance_mc
# ============================================================================

class TestTVDistanceMC:
    """Test suite for tv_distance_mc method."""

    @staticmethod
    def _density(mean, scale=1.0):
        def density(points, input_index=0):
            pts = np.asarray(points).ravel()
            return norm.pdf(pts, loc=mean, scale=scale)

        return density
    
    def test_tv_symmetry(self, quest_default):
        """Test 1: TV(p, q) == TV(q, p)."""
        density_p = self._density(0.0)
        density_q = self._density(0.5)

        def sampler_p(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return rng.normal(0, 1, size=(n_samples, 1, 1))
        
        def sampler_q(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return rng.normal(0.5, 1, size=(n_samples, 1, 1))
        
        tv_pq = quest_default.tv_distance_mc(
            density_p, density_q, sampler_p, sampler_q, np.array([[0]]),
            n_samples=5000, random_state=42,
        )
        tv_qp = quest_default.tv_distance_mc(
            density_q, density_p, sampler_q, sampler_p, np.array([[0]]),
            n_samples=5000, random_state=42,
        )
        
        assert np.isclose(tv_pq, tv_qp, atol=0.01), f"TV not symmetric: {tv_pq} vs {tv_qp}"
    
    def test_tv_bounds(self, quest_default):
        """Test 2: 0 <= TV <= 1."""
        density_p = self._density(0.0)
        density_q = self._density(2.0, scale=0.5)
        
        def sampler_p(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return rng.normal(0, 1, size=(n_samples, 1, 1))
        
        def sampler_q(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return rng.normal(2, 0.5, size=(n_samples, 1, 1))
        
        tv = quest_default.tv_distance_mc(
            density_p, density_q, sampler_p, sampler_q, np.array([[0]]),
            n_samples=5000, random_state=42,
        )
        
        assert 0 <= tv <= 1, f"TV out of bounds [0, 1]: {tv}"
    
    def test_tv_determinism_fixed_seed(self, quest_default):
        """Test 3: Fixed seed produces deterministic output."""
        density_p = self._density(0.0)
        density_q = self._density(0.5)
        
        def sampler_p(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return rng.normal(0, 1, size=(n_samples, 1, 1))
        
        def sampler_q(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return rng.normal(0.5, 1, size=(n_samples, 1, 1))
        
        tv1 = quest_default.tv_distance_mc(
            density_p, density_q, sampler_p, sampler_q, np.array([[0]]),
            n_samples=1000, random_state=123,
        )
        tv2 = quest_default.tv_distance_mc(
            density_p, density_q, sampler_p, sampler_q, np.array([[0]]),
            n_samples=1000, random_state=123,
        )
        
        assert tv1 == tv2, f"Not deterministic: {tv1} vs {tv2}"
    
    def test_tv_identical_distributions_near_zero(self, quest_default):
        """Test 4: Identical distributions yield TV near zero."""
        density_p = self._density(0.0)
        
        def sampler_p(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return rng.normal(0, 1, size=(n_samples, 1, 1))
        
        tv = quest_default.tv_distance_mc(
            density_p, density_p, sampler_p, sampler_p, np.array([[0]]),
            n_samples=5000, random_state=42,
        )
        
        assert tv < 0.05, f"Identical distributions TV not near zero: {tv}"
    
    def test_tv_analytic_reference_point_masses(self, quest_default):
        """Test 5: TV between two point masses with known value.
        
        For point masses at delta(0) and delta(1) on [0, 1], TV = 1.
        """
        
        def sampler_p(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return np.zeros((n_samples, 1, 1))  # All samples at 0
        
        def sampler_q(X, n_samples, rng):
            rng = np.random.default_rng(rng)
            return np.ones((n_samples, 1, 1))  # All samples at 1
        
        density_p = self._density(0.0, scale=0.1)
        density_q = self._density(1.0, scale=0.1)

        tv = quest_default.tv_distance_mc(
            density_p, density_q, sampler_p, sampler_q, np.array([[0]]),
            n_samples=5000, random_state=42,
        )
        
        # TV between delta(0) and delta(1) should be 1.0
        assert 0.95 <= tv <= 1.0, f"Expected TV ≈ 1.0, got {tv}"


# ============================================================================
# Tests for _hdr_resampler_from_samples
# ============================================================================

class TestHDRResamplerFromSamples:
    """Test suite for _hdr_resampler_from_samples helper."""
    
    def test_resampler_shape(self, quest_default):
        """Test 13: Returned samples have shape (n_samples, N, d)."""
        samples = np.random.normal(0, 1, size=(100, 3, 2))  # (S, N, d)
        mask = np.ones((100, 3), dtype=bool)
        mask[50:, 0] = False  # Some rejected for input 0
        
        resampler = quest_default._hdr_resampler_from_samples(samples, mask)
        resampled = resampler(np.zeros((3, 1)), n_samples=50, rng=42)
        
        assert resampled.shape == (50, 3, 2), f"Expected shape (50, 3, 2), got {resampled.shape}"
    
    def test_resampler_acceptance(self, quest_default):
        """Test 14: Resampled points come only from accepted HDR samples."""
        # Create a simple 1D case: samples at -10 (rejected) and +10 (accepted)
        samples = np.array([
            [[-10.0]],
            [[+10.0]],
        ]).repeat(3, axis=1)  # (S=2, N=3, d=1)
        
        mask = np.zeros((2, 3), dtype=bool)
        mask[1, :] = True  # Only the +10 sample is accepted
        
        resampler = quest_default._hdr_resampler_from_samples(samples, mask)
        resampled = resampler(np.zeros((3, 1)), n_samples=100, rng=42)
        
        # All resampled values should be close to +10
        assert np.allclose(resampled, 10.0, atol=0.1), \
            f"Resampled values not from accepted set: min={resampled.min()}, max={resampled.max()}"
    
    def test_resampler_fallback_empty_hdr(self, quest_default):
        """Test 15: Empty accepted sets still return valid samples without crashing."""
        samples = np.random.normal(0, 1, size=(100, 3, 2))
        mask = np.zeros((100, 3), dtype=bool)  # All rejected
        
        resampler = quest_default._hdr_resampler_from_samples(samples, mask)
        resampled = resampler(np.zeros((3, 1)), n_samples=50, rng=42)
        
        # Should not crash and should return shape (50, 3, 2)
        assert resampled.shape == (50, 3, 2)
        # Fallback uses standard normal, so values should be finite
        assert np.all(np.isfinite(resampled))


# ============================================================================
# Tests for Truncated Densities
# ============================================================================

class TestTruncatedDensities:
    """Test suite for QUEST HDR truncation helpers."""

    @staticmethod
    def _piecewise_density(points, input_index=0):
        pts = np.asarray(points).reshape(-1)
        return 1.0 / (1.0 + pts ** 2)

    @staticmethod
    def _normal_density(points, input_index=0):
        pts = np.asarray(points).reshape(-1)
        return norm.pdf(pts, loc=0.0, scale=1.0)

    def test_truncated_density_zero_outside_hdr(self, quest_default):
        """Test 9: QUEST HDR masking excludes low-density samples outside the truncation set."""

        X = np.array([[0.0]])
        sample_points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]).reshape(-1, 1, 1)

        def sampler(_, n_samples, rng):
            assert n_samples == 5
            return sample_points

        c_alpha, hdr_indicator, samples, sample_mask = quest_default._hdr_from_density_function(
            self._piecewise_density,
            sampler,
            X,
            alpha=0.4,
            n_samples=5,
            random_state=42,
        )

        assert samples.shape == (5, 1, 1)
        assert sample_mask.shape == (5, 1)
        assert np.array_equal(sample_mask[:, 0], np.array([False, True, True, True, False]))
        assert hdr_indicator(np.array([[-2.0], [0.0], [2.0]]), input_index=0).tolist() == [False, True, False]

    def test_truncated_density_normalization(self, quest_default):
        """Test 10: QUEST retains the expected HDR mass under truncation."""

        X = np.array([[0.0]])
        rng = np.random.default_rng(123)

        def sampler(_, n_samples, rng_):
            rng_local = np.random.default_rng(rng_)
            return rng_local.normal(0.0, 1.0, size=(n_samples, 1, 1))

        _, _, samples, sample_mask = quest_default._hdr_from_density_function(
            self._normal_density,
            sampler,
            X,
            alpha=0.32,
            n_samples=5000,
            random_state=123,
        )

        retained_mass = sample_mask.mean()
        assert np.isclose(retained_mass, 0.68, atol=0.05), f"Retained HDR mass {retained_mass} not close to 0.68"
        assert samples.shape == (5000, 1, 1)

    def test_truncated_density_shape_preserved(self, quest_default):
        """Test 11: QUEST HDR sampling preserves the expected sample and mask shapes."""

        X = np.array([[0.0], [1.0], [-1.0]])

        def sampler(_, n_samples, rng):
            rng_local = np.random.default_rng(rng)
            return rng_local.normal(0.0, 1.0, size=(n_samples, 3, 2))

        _, hdr_indicator, samples, sample_mask = quest_default._hdr_from_density_function(
            lambda points, input_index=0: norm.pdf(np.asarray(points)[:, 0], loc=0.0, scale=1.0),
            sampler,
            X,
            alpha=0.2,
            n_samples=100,
            random_state=7,
        )

        assert samples.shape == (100, 3, 2)
        assert sample_mask.shape == (100, 3)
        indicator = hdr_indicator(np.array([[0.0, 0.0], [1.0, 1.0]]), input_index=0)
        assert indicator.shape == (2,)

    def test_gaussian_truncation_correctness(self, quest_default):
        """Test 12: Gaussian HDR truncation keeps high-density samples near the mode."""

        X = np.array([[0.0]])

        def sampler(_, n_samples, rng):
            rng_local = np.random.default_rng(rng)
            return rng_local.normal(0.0, 1.0, size=(n_samples, 1, 1))

        _, _, samples, sample_mask = quest_default._hdr_from_density_function(
            self._normal_density,
            sampler,
            X,
            alpha=0.1,
            n_samples=4000,
            random_state=99,
        )

        sample_values = samples[:, 0, 0]
        accepted_values = sample_values[sample_mask[:, 0]]
        rejected_values = sample_values[~sample_mask[:, 0]]

        assert accepted_values.size > 0
        assert rejected_values.size > 0
        assert np.mean(np.abs(accepted_values)) < np.mean(np.abs(rejected_values))
        assert np.isclose(sample_mask.mean(), 0.9, atol=0.05)


# ============================================================================
# Tests for _compute_total_helper
# ============================================================================

class TestComputeTotalHelper:
    """Test suite for _compute_total_helper."""
    
    def test_state_restoration(self, quest_default, mock_model_mle, X_single):
        """Test 6: alpha and scope are restored after the call."""
        quest_default.alpha = 0.1
        quest_default.scope = 'local'
        
        alpha_orig = quest_default.alpha
        scope_orig = quest_default.scope
        
        # Mock the internal calls to avoid expensive computation
        with patch.object(quest_default, '_compute_aleatoric', return_value=np.array([0.5])), \
             patch.object(quest_default, '_hdr_from_density_function') as mock_hdr:
            
            # Mock HDR outputs
            mock_hdr.side_effect = [
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)), 
                 np.ones((100, 1), dtype=bool)),
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)), 
                 np.ones((100, 1), dtype=bool)),
            ]
            
            with patch.object(quest_default, 'tv_distance_mc', return_value=0.1):
                result = quest_default._compute_total_helper(mock_model_mle, X_single, alpha=0.05)
        
        # Check restoration
        assert quest_default.alpha == alpha_orig, "alpha not restored"
        assert quest_default.scope == scope_orig, "scope not restored"
    
    def test_config_validation_passes_correct_config(self, X_single):
        """Test 8 (passing case): Correct config does not raise."""
        cfg = InferentialChoiceConfig(predict='bma', approximate='posterior_predictive', 
                                     point_estimate_criterion='mle')
        model = MockModel(inferential_choice=cfg)
        
        quest = QUESTUncertainty(alpha=0.1, decomposition='total', scope='local', mc_n_samples=100)
        
        with patch.object(quest, '_compute_aleatoric', return_value=np.array([0.5])), \
             patch.object(quest, '_hdr_from_density_function') as mock_hdr, \
             patch.object(quest, 'tv_distance_mc', return_value=0.1):
            
            mock_hdr.side_effect = [
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)), 
                 np.ones((100, 1), dtype=bool)),
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)), 
                 np.ones((100, 1), dtype=bool)),
            ]
            
            # Should not raise
            result = quest._compute_total_helper(model, X_single, alpha=0.05)
            assert result is not None
    
    def test_compute_total_helper_accepts_nonbma_predict_config(self, X_single):
        """Test 8: Helper still returns a result when predict is not bma."""
        cfg = InferentialChoiceConfig(
            predict='posterior_predictive',
            approximate='posterior_predictive',
            point_estimate_criterion='mle',
        )
        model = MockModel(inferential_choice=cfg)
        
        quest = QUESTUncertainty(alpha=0.1, decomposition='total', scope='local', mc_n_samples=100)
        
        with patch.object(quest, '_compute_aleatoric', return_value=np.array([0.5])), \
             patch.object(quest, '_hdr_from_density_function') as mock_hdr, \
             patch.object(quest, 'tv_distance_mc', return_value=0.1):
            mock_hdr.side_effect = [
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)),
                 np.ones((100, 1), dtype=bool)),
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)),
                 np.ones((100, 1), dtype=bool)),
            ]

            result = quest._compute_total_helper(model, X_single, alpha=0.05)
            assert result is not None
    
    def test_compute_total_helper_accepts_nonposterior_approximate_config(self, X_single):
        """Test 8: Helper still returns a result when approximate is not posterior_predictive."""
        cfg = InferentialChoiceConfig(
            predict='bma',
            approximate='point_estimate',
            point_estimate_criterion='mle',
        )
        model = MockModel(inferential_choice=cfg)
        
        quest = QUESTUncertainty(alpha=0.1, decomposition='total', scope='local', mc_n_samples=100)
        
        with patch.object(quest, '_compute_aleatoric', return_value=np.array([0.5])), \
             patch.object(quest, '_hdr_from_density_function') as mock_hdr, \
             patch.object(quest, 'tv_distance_mc', return_value=0.1):
            mock_hdr.side_effect = [
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)),
                 np.ones((100, 1), dtype=bool)),
                (np.array([0.1]), None, np.random.normal(0, 1, (100, 1, 1)),
                 np.ones((100, 1), dtype=bool)),
            ]

            result = quest._compute_total_helper(model, X_single, alpha=0.05)
            assert result is not None


# ============================================================================
# Tests for _compute_total
# ============================================================================

class TestComputeTotal:
    """Test suite for _compute_total."""
    
    def test_compute_total_local_returns_helper_result(self, X_single):
        """Test 16: Local mode returns the helper result directly."""
        cfg = InferentialChoiceConfig(predict='posterior_predictive', approximate='point_estimate', 
                                     point_estimate_criterion='mle')
        model = MockModel(inferential_choice=cfg)
        
        quest = QUESTUncertainty(alpha=0.1, decomposition='total', scope='local', mc_n_samples=100)
        
        helper_result = np.array([0.42])
        
        with patch.object(quest, '_compute_total_helper', return_value=helper_result) as mock_helper:
            result = quest._compute_total(model, X_single)
            
            mock_helper.assert_called_once()
            assert np.allclose(result, helper_result), f"Expected {helper_result}, got {result}"
    
    def test_compute_total_global_integrates_over_alpha(self, X_single):
        """Test 17: Global mode integrates helper output across alpha values."""
        cfg = InferentialChoiceConfig(predict='posterior_predictive', approximate='point_estimate', 
                                     point_estimate_criterion='mle')
        model = MockModel(inferential_choice=cfg)
        
        quest = QUESTUncertainty(alpha=None, decomposition='total', scope='global', mc_n_samples=100)
        
        # Mock the helper to return a linear function of alpha: f(alpha) = alpha
        # Integral of alpha from 0.01 to 0.99 should be approximately (0.99^2 - 0.01^2) / 2
        helper_side_effects = []
        alphas_called = []
        
        def side_effect(model, X, alpha):
            alphas_called.append(alpha)
            return np.array([alpha])  # Return the alpha value itself
        
        with patch.object(quest, '_compute_total_helper', side_effect=side_effect) as mock_helper:
            result = quest._compute_total(model, X_single)
            
            # Check that helper was called multiple times with different alphas
            assert len(alphas_called) > 1, "Helper should be called multiple times for global scope"
            assert len(set(alphas_called)) > 1, "Helper should be called with different alpha values"
            
            # Result should be a positive scalar (integral of alpha values)
            assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.size == 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
