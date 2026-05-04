"""Test top-level seeds block support in resolve_runtime_seeds."""
import pytest
from modal_uq.utils.seed import resolve_runtime_seeds


def test_resolve_runtime_seeds_reads_top_level_seeds_block():
    """Verify top-level 'seeds' block is consumed by resolver."""
    cfg = {
        "seeds": {
            "experiment": 100,
            "dataset": 200,
            "model": 300,
            "pgt": 400,
            "uncertainty": 500,
        },
        "experiment": {},
        "dataset": {"name": "synthetic_conditional", "params": {}},
        "model": {"name": "ensemble", "params": {}},
        "pseudo_ground_truth": {"name": "gmm", "params": {}},
        "uncertainty": {
            "measures": [
                {"name": "alpha_volume", "params": {}},
            ]
        },
    }
    
    resolved = resolve_runtime_seeds(cfg)
    
    # Verify that top-level seeds were used as parent seeds
    assert resolved["experiment"]["seed"] == 100
    # Dataset should derive from dataset seed
    assert resolved["dataset"]["params"]["seed_master"] is not None
    # Model should derive from model seed
    assert resolved["model"]["params"]["seed"] is not None
    # PGT should derive from pgt seed
    assert resolved["pseudo_ground_truth"]["params"]["random_state"] is not None
    # Uncertainty should derive from uncertainty seed
    assert resolved["uncertainty"]["measures"][0]["params"]["mc_random_state"] is not None


def test_resolve_runtime_seeds_backward_compatible_without_block():
    """Verify resolver still works without top-level seeds block."""
    cfg = {
        "experiment": {"seed": 42},
        "dataset": {"name": "synthetic_conditional", "params": {}},
        "model": {"name": "ensemble", "params": {}},
        "pseudo_ground_truth": {"name": "gmm", "params": {}},
        "uncertainty": {
            "measures": [
                {"name": "alpha_volume", "params": {}},
            ]
        },
    }
    
    resolved = resolve_runtime_seeds(cfg)
    
    # Explicit experiment seed should be preserved
    assert resolved["experiment"]["seed"] == 42
    # All other seeds should be derived from experiment seed
    assert resolved["dataset"]["params"]["seed_master"] is not None
    assert resolved["model"]["params"]["seed"] is not None


def test_resolve_runtime_seeds_explicit_overrides_top_level():
    """Verify explicit field seeds take precedence over top-level block."""
    cfg = {
        "seeds": {
            "experiment": 100,
            "model": 300,
        },
        "experiment": {"seed": 50},  # explicit, should be preserved
        "dataset": {"name": "synthetic_conditional", "params": {}},
        "model": {"name": "ensemble", "params": {"seed": 350}},  # explicit, should be preserved
        "pseudo_ground_truth": {"name": "gmm", "params": {}},
        "uncertainty": {
            "measures": [
                {"name": "alpha_volume", "params": {}},
            ]
        },
    }
    
    resolved = resolve_runtime_seeds(cfg)
    
    # Explicit values should be preserved, not overridden by top-level
    assert resolved["experiment"]["seed"] == 50
    assert resolved["model"]["params"]["seed"] == 350
