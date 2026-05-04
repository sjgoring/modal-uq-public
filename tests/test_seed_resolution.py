import copy

from modal_uq.utils.seed import resolve_runtime_seeds


def _sample_config():
    return {
        "experiment": {
            "type": "selective",
            "output_dir": "runs/test",
        },
        "dataset": {
            "name": "synthetic_conditional",
            "params": {
                "n_samples": 100,
            },
        },
        "pseudo_ground_truth": {
            "name": "conditional_kde",
            "params": {
                "bandwidth": 0.5,
            },
        },
        "model": {
            "name": "ensemble",
            "params": {
                "base_model": "condgmm",
                "n_members": 3,
                "base_params": {
                    "n_components": 2,
                },
            },
        },
        "uncertainty": {
            "measures": [
                {"name": "alpha_volume", "params": {"decomposition": "epistemic"}},
                {"name": "integrated_volume", "params": {"decomposition": "epistemic"}},
            ]
        },
    }


def test_resolve_runtime_seeds_fills_missing_values():
    resolved = resolve_runtime_seeds(copy.deepcopy(_sample_config()))

    assert isinstance(resolved["experiment"]["seed"], int)
    # split_seed is only for datasets that use _apply_split (faithful, forestfires)
    # synthetic datasets don't need split_seed, they manage splits internally
    assert isinstance(resolved["dataset"]["params"]["seed_master"], int)
    assert isinstance(resolved["dataset"]["params"]["seed_mode_assign"], int)
    assert isinstance(resolved["dataset"]["params"]["seed_sample"], int)
    assert isinstance(resolved["dataset"]["params"]["seed_noise"], int)
    assert isinstance(resolved["model"]["params"]["seed"], int)

    for measure in resolved["uncertainty"]["measures"]:
        assert isinstance(measure["params"]["mc_random_state"], int)


def test_resolve_runtime_seeds_preserves_explicit_values():
    cfg = _sample_config()
    cfg["experiment"]["seed"] = 123
    cfg["dataset"]["params"]["seed_master"] = 456
    cfg["model"]["params"]["seed"] = 789
    cfg["uncertainty"]["measures"][0]["params"]["mc_random_state"] = 321

    resolved_a = resolve_runtime_seeds(copy.deepcopy(cfg))
    resolved_b = resolve_runtime_seeds(copy.deepcopy(cfg))

    assert resolved_a == resolved_b
    assert resolved_a["experiment"]["seed"] == 123
    assert resolved_a["dataset"]["params"]["seed_master"] == 456


def test_resolve_runtime_seeds_adds_split_seed_for_faithful():
    """Verify split_seed is added for datasets that use _apply_split (faithful, forestfires)."""
    cfg = {
        "experiment": {},
        "dataset": {
            "name": "faithful",
            "params": {},
        },
        "model": {"name": "ensemble", "params": {}},
        "uncertainty": {"measures": []},
    }
    
    resolved = resolve_runtime_seeds(copy.deepcopy(cfg))
    
    # faithful uses _apply_split, so split_seed should be filled
    assert isinstance(resolved["dataset"]["params"]["split_seed"], int)
    assert isinstance(resolved["model"]["params"]["seed"], int)
    assert resolved["model"]["params"]["seed"] > 0