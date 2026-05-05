"""Test that SyntheticConstantVarDataset initializes train/test/val splits for active_learning compatibility."""
import pytest
import numpy as np
from modal_uq.datasets.synthetic_constant_var import SyntheticConstantVarDataset


def test_synthetic_constant_var_initializes_train_test_splits():
    """Verify that SyntheticConstantVarDataset creates X_train, y_train, X_test, y_test in __init__."""
    ds = SyntheticConstantVarDataset(
        n_samples=100,
        x_min=-10.0,
        x_max=10.0,
        y_min=-5.0,
        y_max=5.0,
        seed=42,
        split_seed=42
    )
    
    # Check that all required attributes are set
    assert hasattr(ds, 'X_train'), "Missing X_train"
    assert hasattr(ds, 'y_train'), "Missing y_train"
    assert hasattr(ds, 'X_val'), "Missing X_val"
    assert hasattr(ds, 'y_val'), "Missing y_val"
    assert hasattr(ds, 'X_test'), "Missing X_test"
    assert hasattr(ds, 'y_test'), "Missing y_test"
    
    # Check shapes and types
    assert isinstance(ds.X_train, np.ndarray), "X_train should be ndarray"
    assert isinstance(ds.y_train, np.ndarray), "y_train should be ndarray"
    assert ds.X_train.shape == (60, 1), f"Expected X_train shape (60, 1), got {ds.X_train.shape}"
    assert ds.y_train.shape == (60,), f"Expected y_train shape (60,), got {ds.y_train.shape}"
    assert ds.X_test.shape == (20, 1), f"Expected X_test shape (20, 1), got {ds.X_test.shape}"
    assert ds.y_test.shape == (20,), f"Expected y_test shape (20,), got {ds.y_test.shape}"
    
    # Check that splits are disjoint
    assert not np.any(np.isin(ds.X_train, ds.X_test)), "Train and test sets should be disjoint"
    
    # Check total samples
    total_samples = len(ds.X_train) + len(ds.X_val) + len(ds.X_test)
    assert total_samples == 100, f"Expected 100 total samples, got {total_samples}"


def test_synthetic_constant_var_caches_y_grid():
    """Verify that SyntheticConstantVarDataset caches y_grid in __init__."""
    ds = SyntheticConstantVarDataset(
        n_samples=50,
        y_grid_size=500,
        y_min=-5.0,
        y_max=5.0,
        y_pad=1.0,
        seed=42,
        split_seed=42
    )
    
    # Check y_grid is set
    assert hasattr(ds, 'y_grid'), "Missing y_grid"
    assert isinstance(ds.y_grid, np.ndarray), "y_grid should be ndarray"
    assert len(ds.y_grid) == 500, f"Expected y_grid size 500, got {len(ds.y_grid)}"
    
    # Check that y_grid is padded
    assert ds.y_grid[0] < -5.0, "y_grid should be padded below y_min"
    assert ds.y_grid[-1] > 5.0, "y_grid should be padded above y_max"


def test_active_learning_caches_y_grid():
    """Verify that ActiveLearning experiment caches canonical y_grid from dataset."""
    from modal_uq.experiments.active_learning import ActiveLearning
    from modal_uq.models.ensemble import Ensemble
    from modal_uq.pgt.conditional_kde import ConditionalKDE
    
    ds = SyntheticConstantVarDataset(
        n_samples=50,
        y_grid_size=200,
        seed=42,
        split_seed=42
    )
    
    # Create minimal experiment objects
    model = Ensemble(base_model='condgmm', n_members=2, base_params={'n_components': 2})
    pgt = ConditionalKDE()
    cfg = {'experiment': {}, 'uncertainty': []}
    
    # Create ActiveLearning experiment
    al = ActiveLearning(ds, pgt, model, {}, cfg, n_jobs=1)
    
    # Check that y_grid is cached
    assert hasattr(al, 'y_grid'), "ActiveLearning should cache y_grid"
    assert al.y_grid is ds.y_grid, "ActiveLearning should cache the dataset's y_grid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
