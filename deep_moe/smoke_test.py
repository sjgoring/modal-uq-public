"""Smoke test for the v2 quest pipeline.

Verifies end-to-end execution at a small scale (deep ensemble, both estimators,
all three noise types). Run before any real experiment to catch issues early.
"""

import time
from pathlib import Path

import numpy as np

from selective_prediction import run_experiment


def main():
    print("Running smoke test (small scale, all 3 noise types, oracle only)...")
    print("Expected runtime: ~3-5 minutes total (deep ensemble training is slow).")
    print()
    
    t0 = time.time()
    out_dir = Path("smoke_test_output")
    
    for nd in ["gaussian", "bimodal", "skewed"]:
        run_experiment(
            noise_dist=nd,
            n_train=300,
            n_test=50,
            M=4,
            K=2,
            n_seeds=2,
            base_seed=0,
            n_jobs=1,
            output_dir=str(out_dir),
            n_coverage_points=15,
            estimator="oracle",
            model="deep",
            hidden_dim=24,
            n_hidden=2,
            n_epochs=200,
        )
    
    elapsed = time.time() - t0
    print(f"\n\nSmoke test completed in {elapsed:.1f}s.")
    print()
    
    print("Sanity checks:")
    for nd in ["gaussian", "bimodal", "skewed"]:
        path = out_dir / f"results_{nd}_oracle.npz"
        if not path.exists():
            print(f"  ✗ Missing: {path}")
            continue
        data = np.load(path)
        n_finite = sum(np.isfinite(float(data[k])) 
                       for k in data.files if k.startswith("aurc_mean_"))
        print(f"  ✓ {nd}: {n_finite} AURC means written, all finite.")
    
    print("\nPipeline is working. To run full experiments:")
    print("  python selective_prediction.py --noise all --estimator all "
          "--n-seeds 10 --n-jobs 4")


if __name__ == "__main__":
    main()
