"""Test that QUEST per-input seeding logic produces deterministic results."""
import pytest
import numpy as np
from modal_uq.utils.seed import derive_seed, resolve_seed


def test_quest_per_input_seed_derivation_deterministic():
    """Verify QUEST per-input seed derivation is deterministic across calls."""
    parent_seed = 100
    N = 10  # number of inputs
    
    # Derive seeds once
    seeds_1 = [derive_seed(parent_seed, "volume", i) for i in range(N)]
    
    # Derive seeds again - should be identical
    seeds_2 = [derive_seed(parent_seed, "volume", i) for i in range(N)]
    
    # All seeds should match
    assert seeds_1 == seeds_2, "Per-input seed derivation should be deterministic"
    
    # All seeds should be unique (different per input)
    assert len(set(seeds_1)) == N, "Each input should get a unique seed"


def test_quest_per_input_seed_derivation_independent_of_order():
    """Verify that derived seeds don't depend on iteration order (threading-safe)."""
    parent_seed = 200
    N = 20
    
    # Derive in forward order
    forward_seeds = [derive_seed(parent_seed, "volume", i) for i in range(N)]
    
    # Derive in random order
    indices = [5, 2, 15, 0, 19, 10, 3, 8, 12, 1, 7, 4, 18, 6, 17, 9, 11, 13, 14, 16]
    random_seeds = {i: derive_seed(parent_seed, "volume", i) for i in indices}
    
    # Seeds should match regardless of order
    for i in range(N):
        assert forward_seeds[i] == random_seeds[i], \
            f"Per-input seed for index {i} should be independent of derivation order"


def test_quest_per_input_seed_rng_produces_different_samples():
    """Verify RNGs from per-input seeds produce different samples (confirming uniqueness)."""
    parent_seed = 300
    N = 5
    n_samples_per_input = 100
    
    # Create RNGs using derived seeds and draw samples
    all_samples = []
    for i in range(N):
        input_seed = derive_seed(parent_seed, "volume", i)
        rng = np.random.default_rng(input_seed)
        samples = rng.uniform(0, 1, size=n_samples_per_input)
        all_samples.append(samples)
    
    # Each RNG should produce different sequences
    for i in range(N - 1):
        for j in range(i + 1, N):
            # Samples from different inputs should not be identical
            assert not np.allclose(all_samples[i], all_samples[j]), \
                f"RNG for input {i} and {j} should produce different sequences"


def test_quest_resolved_seed_to_per_input_derivation():
    """Verify the full pipeline: resolve_seed -> per-input derivation."""
    # Simulate what QUEST does: resolve a random_state, then derive per-input seeds
    cfg = {
        "experiment": {},
        "uncertainty": {
            "measures": [
                {"name": "alpha_volume", "params": {}},
            ]
        },
    }
    
    # In real execution, this would be resolved by resolve_runtime_seeds
    mc_random_state = 400
    resolved_seed = resolve_seed(mc_random_state)
    
    assert isinstance(resolved_seed, int)
    assert resolved_seed > 0
    
    # Now derive per-input seeds
    N = 10
    input_seeds = [derive_seed(resolved_seed, "volume", i) for i in range(N)]
    
    # Should all be concrete positive integers
    for seed in input_seeds:
        assert isinstance(seed, int)
        assert seed > 0


def test_quest_per_input_seeds_stable_across_runs():
    """Verify per-input seeds remain stable when derived multiple times."""
    parent_seed = 500
    
    # Derive input seeds on first run
    run1_seeds = []
    for i in range(15):
        seed = derive_seed(parent_seed, "volume", i)
        run1_seeds.append(seed)
    
    # Derive again (simulating a second execution)
    run2_seeds = []
    for i in range(15):
        seed = derive_seed(parent_seed, "volume", i)
        run2_seeds.append(seed)
    
    # Must match exactly
    assert run1_seeds == run2_seeds, \
        "Per-input seeds must be stable across multiple derivations"
