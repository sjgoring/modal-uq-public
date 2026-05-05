"""Test that uncertainty measures use canonical y_grid when provided."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch


def test_differential_entropy_compute_methods_accept_y_grid():
    """Verify DifferentialEntropy._compute_* methods accept y_grid parameter."""
    from modal_uq.uncertainty.differential_entropy import DifferentialEntropy
    
    de = DifferentialEntropy(decomposition='total', grid_points=512)
    
    # Check that methods have y_grid parameter
    assert 'y_grid' in de._compute_total.__code__.co_varnames, "_compute_total should accept y_grid"
    assert 'y_grid' in de._compute_aleatoric.__code__.co_varnames, "_compute_aleatoric should accept y_grid"
    assert 'y_grid' in de._compute_epistemic.__code__.co_varnames, "_compute_epistemic should accept y_grid"
    
    # Check that score methods have y_grid parameter
    assert 'y_grid' in de.score.__code__.co_varnames, "score should accept y_grid"
    assert 'y_grid' in de.score_total.__code__.co_varnames, "score_total should accept y_grid"
    assert 'y_grid' in de.score_aleatoric.__code__.co_varnames, "score_aleatoric should accept y_grid"
    assert 'y_grid' in de.score_epistemic.__code__.co_varnames, "score_epistemic should accept y_grid"


def test_variance_compute_methods_accept_y_grid():
    """Verify PredictiveVariance._compute_* methods accept y_grid parameter."""
    from modal_uq.uncertainty.variance import PredictiveVariance
    
    pv = PredictiveVariance(decomposition='total', grid_points=512)
    
    # Check that methods have y_grid parameter
    assert 'y_grid' in pv._compute_total.__code__.co_varnames, "_compute_total should accept y_grid"
    assert 'y_grid' in pv._compute_aleatoric.__code__.co_varnames, "_compute_aleatoric should accept y_grid"
    assert 'y_grid' in pv._compute_epistemic.__code__.co_varnames, "_compute_epistemic should accept y_grid"
    
    # Check that score methods have y_grid parameter
    assert 'y_grid' in pv.score.__code__.co_varnames, "score should accept y_grid"
    assert 'y_grid' in pv.score_total.__code__.co_varnames, "score_total should accept y_grid"
    assert 'y_grid' in pv.score_aleatoric.__code__.co_varnames, "score_aleatoric should accept y_grid"
    assert 'y_grid' in pv.score_epistemic.__code__.co_varnames, "score_epistemic should accept y_grid"


def test_compute_uncertainty_scores_accepts_y_grid():
    """Verify compute_uncertainty_scores accepts and passes y_grid parameter."""
    from modal_uq.analysis.correlation import compute_uncertainty_scores
    
    # Check function signature
    assert 'y_grid' in compute_uncertainty_scores.__code__.co_varnames, "compute_uncertainty_scores should accept y_grid"


def test_differential_entropy_uses_provided_y_grid_over_model_default():
    """Test that y_grid parameter takes precedence over model.default_y_grid()."""
    from modal_uq.uncertainty.differential_entropy import DifferentialEntropy
    
    de = DifferentialEntropy(decomposition='total', grid_points=512)
    
    # Mock the _predict_density_collection and entropy computation
    with patch.object(de, '_predict_density_collection') as mock_predict, \
         patch.object(de, '_cross_entropy_from_density') as mock_ce:
        
        # Setup return values
        mock_model = Mock()
        mock_model.get_inferential_choice_config.return_value = Mock(predict='bma', approximate='posterior_predictive')
        mock_model.default_y_grid = Mock(return_value=np.linspace(-5, 5, 512))
        
        mock_predict.return_value = np.random.rand(2, 3, 100)  # [S, N, G]
        mock_ce.return_value = np.random.rand(2)
        
        X = np.array([[1.0], [2.0], [3.0]])
        custom_y_grid = np.linspace(-10, 10, 256)
        
        # Call _compute_total with custom y_grid
        de._compute_total(mock_model, X, y_grid=custom_y_grid)
        
        # Verify model.default_y_grid was NOT called
        mock_model.default_y_grid.assert_not_called()



def test_variance_uses_provided_y_grid_over_model_default():
    """Test that y_grid parameter takes precedence over model.default_y_grid()."""
    from modal_uq.uncertainty.variance import PredictiveVariance
    
    pv = PredictiveVariance(decomposition='total', grid_points=512)
    
    # Mock the _predict_density_collection and moments computation
    with patch.object(pv, '_predict_density_collection') as mock_predict, \
         patch.object(pv, '_posterior_predictive_density') as mock_ppd, \
         patch.object(pv, '_moments_from_density') as mock_moments:
        
        # Setup return values
        mock_model = Mock()
        mock_model.get_inferential_choice_config.return_value = Mock(predict='bma', approximate='posterior_predictive')
        mock_model.default_y_grid.return_value = np.linspace(-5, 5, 512)
        
        mock_predict.return_value = np.random.rand(2, 3, 100)
        mock_ppd.return_value = np.random.rand(3, 100)
        mock_moments.return_value = (np.random.rand(3), np.random.rand(3))
        
        X = np.array([[1.0], [2.0], [3.0]])
        custom_y_grid = np.linspace(-10, 10, 256)
        
        # Call _compute_total with custom y_grid
        pv._compute_total(mock_model, X, y_grid=custom_y_grid)
        
        # Verify model.default_y_grid was NOT called
        mock_model.default_y_grid.assert_not_called()


def test_quest_uses_injected_y_grid_on_1d_grid_path():
    """Verify QUEST accepts y_grid and uses the injected grid on the 1D HDR path."""
    from modal_uq.uncertainty.quest import QUESTUncertainty

    class DummyQuestModel:
        n_jobs = None

        def get_inferential_choice_config(self):
            return Mock(predict='bma', approximate='posterior_predictive')

        def default_y_grid(self, *args, **kwargs):
            raise AssertionError("default_y_grid should not be called when y_grid is provided")

        def predict_density(self, X, y_grid, context='predict'):
            X = np.asarray(X)
            y_grid = np.asarray(y_grid)
            return np.ones((X.shape[0], y_grid.shape[0]))

    quest = QUESTUncertainty(alpha=0.1, decomposition='aleatoric', scope='local', grid_points=64)
    model = DummyQuestModel()
    X = np.array([[0.0], [1.0]])
    custom_y_grid = np.linspace(-10.0, 10.0, 64)

    with patch.object(quest, '_hdr_from_density_grid_1d', return_value=np.array([0.25, 0.5])) as mock_hdr:
        result = quest.score(model, X, y_grid=custom_y_grid)

    assert result.shape == (2,)
    mock_hdr.assert_called_once()
    assert np.array_equal(mock_hdr.call_args.kwargs['grid'], custom_y_grid)


def test_compute_uncertainty_scores_supports_quest_grid_measures():
    """Verify compute_uncertainty_scores can drive QUEST grid-based measures with y_grid."""
    from modal_uq.analysis.correlation import compute_uncertainty_scores

    class DummyQuestModel:
        n_jobs = None

        def get_inferential_choice_config(self):
            return Mock(predict='bma', approximate='posterior_predictive')

        def default_y_grid(self, *args, **kwargs):
            raise AssertionError("default_y_grid should not be called when y_grid is provided")

        def predict_density(self, X, y_grid, context='predict'):
            X = np.asarray(X)
            y_grid = np.asarray(y_grid)
            return np.ones((X.shape[0], y_grid.shape[0]))

    model = DummyQuestModel()
    X = np.array([[0.0], [1.0]])
    y = np.array([0.0, 1.0])
    custom_y_grid = np.linspace(-10.0, 10.0, 64)
    measure_specs = [
        {"name": "alpha_volume", "params": {"decomposition": "aleatoric", "scope": "local", "alpha": 0.1, "label": "alpha_volume"}},
        {"name": "integrated_volume", "params": {"decomposition": "aleatoric", "scope": "global", "label": "integrated_volume", "n_alpha": 3}},
    ]

    with patch('modal_uq.uncertainty.quest.QUESTUncertainty._hdr_from_density_grid_1d', return_value=np.array([0.25, 0.5])):
        df_scores = compute_uncertainty_scores(measure_specs, model, X, y, y_grid=custom_y_grid)

    assert list(df_scores.columns) == ['alpha_volume', 'integrated_volume']
    assert df_scores.shape == (2, 2)

