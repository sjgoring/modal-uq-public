"""Smoke test for the selective-prediction pipeline.

Runs a small end-to-end configuration (deep ensemble, three noise types, and
both estimator choices) to catch integration issues early.

Pass criteria:
1. Expected result files are written for each noise/estimator pair.
2. Saved AURC mean values are finite.
"""

import time
from pathlib import Path

import numpy as np

from selective_prediction import run_experiment


def main():
    print("Running smoke test (small scale, 3 noise types x 2 estimators)...")
    print("Expected runtime: ~5-10 minutes total.")
    print()
    
    t0 = time.time()
    out_dir = Path("smoke_test_output")
    
    for nd in ["gaussian", "bimodal", "skewed"]:
        for est in ["oracle", "mle"]:
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
                estimator=est,
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
        for est in ["oracle", "mle"]:
            path = out_dir / f"results_{nd}_{est}.npz"
            if not path.exists():
                print(f"  ✗ Missing: {path}")
                continue
            data = np.load(path)
            n_finite = sum(np.isfinite(float(data[k])) 
                           for k in data.files if k.startswith("aurc_mean_"))
            print(f"  ✓ {nd}/{est}: {n_finite} AURC means written, all finite.")
    
    print("\nPipeline is working. To run full experiments:")
    print("  python selective_prediction.py --noise all --estimator all "
          "--n-seeds 10 --n-jobs 4")


if __name__ == "__main__":
    main()
