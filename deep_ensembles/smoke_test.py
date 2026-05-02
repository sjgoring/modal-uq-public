"""
End-to-end smoke test: tiny settings, exercises multi-seed parallel execution.

Should complete in 1-2 minutes on a modern laptop.
"""

import shutil
import time
from pathlib import Path

from selective_prediction import run_experiment


def main():
    print("=" * 60)
    print("SMOKE TEST: tiny settings, multi-seed, parallel")
    print("=" * 60)
    
    smoke_dir = Path("smoke_results")
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    
    t0 = time.time()
    
    # Run with tiny settings on Gaussian noise only, 3 seeds, parallel.
    results = run_experiment(
        noise_dist="gaussian",
        noise_scale=1.0,
        n_train=200,
        n_test=30,
        M=3,
        n_epochs=50,
        hidden_dim=20,
        n_hidden=2,
        batch_size=32,
        lr=1e-3,
        n_seeds=3,
        base_seed=42,
        n_jobs=-1,           # use all CPUs
        n_coverage_points=20,
        output_dir="smoke_results",
    )
    
    elapsed = time.time() - t0
    print(f"\nSmoke test completed in {elapsed:.1f}s.")
    
    aurcs = results["aurc_mean"]
    print("\nSanity checks:")
    
    # All AURCs should be finite
    bad = [k for k, v in aurcs.items() if not (-10 < v < 1e10)]
    if bad:
        print(f"  ⚠ Some AURC means have suspect values: {bad}")
    else:
        print(f"  ✓ All {len(aurcs)} AURC means are finite and reasonable.")
    
    # SE values should be present (since K > 1) and non-negative
    ses = results["aurc_se"]
    bad_se = [k for k, v in ses.items() if v < 0 or not (v < 1e10)]
    if bad_se:
        print(f"  ⚠ Some AURC SEs have suspect values: {bad_se}")
    else:
        print(f"  ✓ All AURC SEs are non-negative and finite.")
    
    # The aggregated full-coverage MSE (won't be great with this tiny setup,
    # but should be finite and positive)
    print(f"\n  Full-coverage MSE (mean across seeds): "
          f"{results['test_mse_mean']:.3f} ± {results['test_mse_se']:.3f}")
    
    print("\nIf no warnings above, the multi-seed pipeline is working.")
    print("You can now run the full experiment:")
    print("  python selective_prediction.py --noise all --n-seeds 20")


if __name__ == "__main__":
    main()
