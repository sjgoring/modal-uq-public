"""Test that missing seed instructions generate concrete seed values."""
import pytest
from modal_uq.utils.seed import resolve_runtime_seeds


def test_resolve_runtime_seeds_no_seed_instruction_generates_concrete_values():
    """Verify that omitting all seed specifications results in concrete integer seeds."""
    cfg = {
        # No top-level seeds block
        "experiment": {},  # No explicit seed
        "dataset": {"name": "synthetic_conditional", "params": {}},  # No seed specs
        "model": {"name": "ensemble", "params": {}},  # No seed specs
        "pseudo_ground_truth": {"name": "gmm", "params": {}},  # No seed specs
        "uncertainty": {
            "measures": [
                {"name": "alpha_volume", "params": {}},  # No seed specs
            ]
        },
    }
    
    resolved = resolve_runtime_seeds(cfg)
    
    # Every seed should be resolved to a concrete positive integer
    assert isinstance(resolved["experiment"]["seed"], int)
    assert resolved["experiment"]["seed"] > 0
    
    assert isinstance(resolved["dataset"]["params"]["seed_master"], int)
    assert resolved["dataset"]["params"]["seed_master"] > 0
    
    assert isinstance(resolved["model"]["params"]["seed"], int)
    assert resolved["model"]["params"]["seed"] > 0
    
    assert isinstance(resolved["pseudo_ground_truth"]["params"]["random_state"], int)
    assert resolved["pseudo_ground_truth"]["params"]["random_state"] > 0
    
    assert isinstance(resolved["uncertainty"]["measures"][0]["params"]["mc_random_state"], int)
    assert resolved["uncertainty"]["measures"][0]["params"]["mc_random_state"] > 0


def test_resolve_runtime_seeds_no_seed_produces_valid_children():
    """Verify that baseline (no-seed) config produces valid child seeds in synthetic datasets."""
    cfg = {
        "experiment": {},
        "dataset": {
            "name": "synthetic_conditional",
            "params": {},  # No explicit seed_master or children
        },
        "model": {"name": "ensemble", "params": {}},
        "pseudo_ground_truth": {"name": "gmm", "params": {}},
        "uncertainty": {
            "measures": [
                {"name": "alpha_volume", "params": {}},
            ]
        },
    }
    
    resolved = resolve_runtime_seeds(cfg)
    
    # All synthetic child seeds should be concrete and derivable
    dataset_params = resolved["dataset"]["params"]
    assert isinstance(dataset_params["seed_master"], int)
    assert isinstance(dataset_params["seed_mode_assign"], int)
    assert isinstance(dataset_params["seed_sample"], int)
    assert isinstance(dataset_params["seed_noise"], int)
    
    # They should all be positive
    assert dataset_params["seed_master"] > 0
    assert dataset_params["seed_mode_assign"] > 0
    assert dataset_params["seed_sample"] > 0
    assert dataset_params["seed_noise"] > 0
