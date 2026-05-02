"""
Selective prediction experiment for QUEST.

Key features:
- MSE as the primary metric.
- Multi-seed support with parallel CPU execution via joblib.
- Mean +/- SE aggregation for risk-coverage curves and AURC.

Pipeline (per seed):
1. Generate Friedman #1 train/test data with specified noise distribution.
2. Train a deep ensemble on the training set.
3. For each test point, compute all uncertainty measures (variance, entropy, QUEST).
4. Rank test points by each UM, compute selective MSE curves.
5. Compute AURC = area under MSE-coverage curve, so lower = better.

For oracle measures (QUEST AU and TU), we use the known true conditional density
under the synthetic noise model.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from friedman import generate_friedman, true_conditional_density, friedman_mean
from ensemble import DeepEnsemble
from predictive import GridDensity1D
from measures import (
    variance_au, variance_eu, variance_tu,
    entropy_au, entropy_eu, entropy_tu,
    quest_au_local, quest_au_global,
    quest_eu_local, quest_eu_global,
    quest_tu_local, quest_tu_global,
)


# Standard set of UMs to evaluate (in display order)
UM_KEYS = [
    "var_au", "var_eu", "var_tu",
    "ent_au", "ent_eu", "ent_tu",
    "quest_au_01", "quest_eu_01", "quest_tu_01",
    "quest_au_05", "quest_eu_05", "quest_tu_05",
    "quest_au_g", "quest_eu_g", "quest_tu_g",
]


@dataclass
class TestPointMeasures:
    """All UMs computed for a single test point."""
    var_au: float = 0.0
    var_eu: float = 0.0
    var_tu: float = 0.0
    ent_au: float = 0.0
    ent_eu: float = 0.0
    ent_tu: float = 0.0
    quest_au_01: float = 0.0
    quest_au_05: float = 0.0
    quest_eu_01: float = 0.0
    quest_eu_05: float = 0.0
    quest_tu_01: float = 0.0
    quest_tu_05: float = 0.0
    quest_au_g: float = 0.0
    quest_eu_g: float = 0.0
    quest_tu_g: float = 0.0


def make_true_density(
    x: np.ndarray, noise_dist: str, noise_scale: float, n_grid: int = 5000
) -> GridDensity1D:
    """Construct a GridDensity1D for the true conditional density at input x.
    
    Grid range is sized using the local sigma(x), so the grid scales correctly
    in heteroscedastic regions.
    """
    from friedman import noise_scale_function
    
    mean = friedman_mean(x[None, :])[0]
    sigma_local = float(noise_scale_function(x[None, :], base_scale=noise_scale)[0])
    
    if noise_dist == "gaussian":
        margin = 5 * sigma_local
    elif noise_dist == "t5":
        margin = 10 * sigma_local
    elif noise_dist == "t3":
        margin = 20 * sigma_local
    else:
        raise ValueError(noise_dist)
    
    y_grid = np.linspace(mean - margin, mean + margin, n_grid)
    densities = true_conditional_density(y_grid, x, noise_dist, noise_scale)
    return GridDensity1D(y_grid, densities)


def compute_all_measures(
    ensemble: DeepEnsemble,
    X_test: np.ndarray,
    noise_dist: str,
    noise_scale: float,
    n_alpha_global: int = 30,
    verbose: bool = False,
) -> list[TestPointMeasures]:
    """Compute all UMs for every test point."""
    results = []
    n_test = X_test.shape[0]
    
    for i in range(n_test):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Processing test point {i+1}/{n_test}")
        
        x = X_test[i]
        pred = ensemble.predictive_distribution(x)
        true_dist = make_true_density(x, noise_dist, noise_scale)
        theta_samples = ensemble.parameter_samples(x)
        
        m = TestPointMeasures()
        
        m.var_au = variance_au(pred)
        m.var_eu = variance_eu(pred)
        m.var_tu = variance_tu(pred)
        
        m.ent_au = entropy_au(pred)
        m.ent_tu = entropy_tu(pred)
        m.ent_eu = m.ent_tu - m.ent_au
        
        m.quest_au_01 = quest_au_local(true_dist, alpha=0.1)
        m.quest_eu_01 = quest_eu_local(theta_samples, alpha=0.1)
        m.quest_tu_01 = quest_tu_local(true_dist, pred, alpha=0.1)
        
        m.quest_au_05 = quest_au_local(true_dist, alpha=0.5)
        m.quest_eu_05 = quest_eu_local(theta_samples, alpha=0.5)
        m.quest_tu_05 = quest_tu_local(true_dist, pred, alpha=0.5)
        
        m.quest_au_g = quest_au_global(true_dist, n_alpha=n_alpha_global)
        m.quest_eu_g = quest_eu_global(theta_samples, n_alpha=n_alpha_global)
        m.quest_tu_g = quest_tu_global(true_dist, pred, n_alpha=n_alpha_global)
        
        results.append(m)
    
    return results


def selective_mse_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    coverages: np.ndarray,
) -> np.ndarray:
    """Compute selective MSE vs coverage curve.
    
    For each coverage level c, retain the c-fraction of points with the lowest
    uncertainty and compute MSE on this subset.
    
    Args:
        y_true: array of shape (n,) with true target values.
        y_pred: array of shape (n,) with predicted values.
        uncertainties: array of shape (n,) (lower = more confident).
        coverages: array of coverage levels in (0, 1].
    
    Returns:
        array of shape (len(coverages),) with MSE values.
    """
    n = len(y_true)
    sort_idx = np.argsort(uncertainties)
    sorted_y_true = y_true[sort_idx]
    sorted_y_pred = y_pred[sort_idx]
    
    mse_values = np.empty(len(coverages))
    for i, c in enumerate(coverages):
        k = max(2, int(round(c * n)))
        yt = sorted_y_true[:k]
        yp = sorted_y_pred[:k]
        mse_values[i] = float(np.mean((yt - yp) ** 2))
    
    return mse_values


def aurc(coverages: np.ndarray, mse_values: np.ndarray) -> float:
    """Area under the selective MSE-coverage curve.
    
    Lower AURC = better (a UM that retains low-error points keeps MSE low at
    small coverage, integrating to a smaller area).
    """
    return float(np.trapz(mse_values, coverages))


def all_um_arrays(measures: list[TestPointMeasures]) -> dict[str, np.ndarray]:
    return {k: np.array([getattr(m, k) for m in measures]) for k in UM_KEYS}


def run_single_seed(
    seed: int,
    noise_dist: str,
    noise_scale: float,
    n_train: int,
    n_test: int,
    M: int,
    n_epochs: int,
    hidden_dim: int,
    n_hidden: int,
    batch_size: int,
    lr: float,
    coverages: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Run the experiment for a single seed."""
    
    X_train, y_train = generate_friedman(
        n=n_train, noise_dist=noise_dist,
        noise_scale=noise_scale, seed=seed,
    )
    X_test, y_test = generate_friedman(
        n=n_test, noise_dist=noise_dist,
        noise_scale=noise_scale, seed=seed + 100000,
    )
    
    ensemble = DeepEnsemble(
        input_dim=X_train.shape[1],
        M=M, hidden_dim=hidden_dim, n_hidden=n_hidden,
    )
    ensemble.fit(
        X_train, y_train,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        device=device, base_seed=seed * M, verbose=False,
    )
    
    mus_test, _ = ensemble.predict(X_test, device=device)
    test_pred_means = mus_test.mean(axis=1)
    test_mse = float(((test_pred_means - y_test) ** 2).mean())
    
    measures = compute_all_measures(
        ensemble, X_test, noise_dist, noise_scale, verbose=False,
    )
    ums = all_um_arrays(measures)
    
    mse_curves = {}
    aurcs = {}
    for name, vals in ums.items():
        mse_curve = selective_mse_curve(y_test, test_pred_means, vals, coverages)
        mse_curves[name] = mse_curve
        aurcs[name] = aurc(coverages, mse_curve)
    
    rng = np.random.default_rng(seed + 999999)
    random_um = rng.uniform(size=n_test)
    mse_random = selective_mse_curve(y_test, test_pred_means, random_um, coverages)
    mse_curves["random"] = mse_random
    aurcs["random"] = aurc(coverages, mse_random)
    
    return {
        "seed": seed,
        "test_mse": test_mse,
        "mse_curves": mse_curves,
        "aurcs": aurcs,
    }


def aggregate_seeds(per_seed_results: list[dict]) -> dict:
    """Aggregate across seeds: compute mean and SE for curves and AURCs."""
    K = len(per_seed_results)
    if K == 0:
        return {}
    
    um_names = list(per_seed_results[0]["mse_curves"].keys())
    
    mse_mean, mse_se = {}, {}
    aurc_mean, aurc_se = {}, {}
    
    for name in um_names:
        curves = np.stack([r["mse_curves"][name] for r in per_seed_results], axis=0)
        mse_mean[name] = curves.mean(axis=0)
        mse_se[name] = (curves.std(axis=0, ddof=1) / np.sqrt(K)
                        if K > 1 else np.zeros_like(curves[0]))
        
        aurc_vals = np.array([r["aurcs"][name] for r in per_seed_results])
        aurc_mean[name] = float(aurc_vals.mean())
        aurc_se[name] = (float(aurc_vals.std(ddof=1) / np.sqrt(K))
                         if K > 1 else 0.0)
    
    test_mses = np.array([r["test_mse"] for r in per_seed_results])
    
    return {
        "mse_mean": mse_mean,
        "mse_se": mse_se,
        "aurc_mean": aurc_mean,
        "aurc_se": aurc_se,
        "test_mse_mean": float(test_mses.mean()),
        "test_mse_se": (float(test_mses.std(ddof=1) / np.sqrt(K))
                        if K > 1 else 0.0),
        "n_seeds": K,
    }


def run_experiment(
    noise_dist: str,
    noise_scale: float,
    n_train: int,
    n_test: int,
    M: int,
    n_epochs: int,
    hidden_dim: int,
    n_hidden: int,
    batch_size: int,
    lr: float,
    n_seeds: int,
    base_seed: int,
    n_jobs: int,
    output_dir: str,
    n_coverage_points: int,
    device: str = "cpu",
) -> dict:
    """Run the full multi-seed experiment for a given noise setting."""
    
    print("=" * 60)
    print(f"Experiment: Friedman #1 with {noise_dist} noise (scale={noise_scale})")
    print(f"  n_train={n_train}, n_test={n_test}, M={M}, epochs={n_epochs}")
    print(f"  n_seeds={n_seeds}, n_jobs={n_jobs}")
    print("=" * 60)
    
    coverages = np.linspace(0.05, 1.0, n_coverage_points)
    
    t0 = time.time()
    seeds = [base_seed + i for i in range(n_seeds)]
    
    if n_jobs == 1:
        per_seed_results = []
        for s in seeds:
            print(f"  Running seed {s}...")
            t_seed = time.time()
            res = run_single_seed(
                s, noise_dist, noise_scale, n_train, n_test, M,
                n_epochs, hidden_dim, n_hidden, batch_size, lr,
                coverages, device,
            )
            per_seed_results.append(res)
            print(f"    Seed {s} took {time.time() - t_seed:.1f}s "
                  f"(test MSE = {res['test_mse']:.3f})")
    else:
        print(f"  Running {n_seeds} seeds in parallel (n_jobs={n_jobs})...")
        per_seed_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_seed)(
                s, noise_dist, noise_scale, n_train, n_test, M,
                n_epochs, hidden_dim, n_hidden, batch_size, lr,
                coverages, device,
            ) for s in seeds
        )
    
    elapsed = time.time() - t0
    print(f"  All seeds done in {elapsed:.1f}s "
          f"({elapsed / n_seeds:.1f}s/seed avg).")
    
    agg = aggregate_seeds(per_seed_results)
    agg["coverages"] = coverages
    agg["noise_dist"] = noise_dist
    agg["noise_scale"] = noise_scale
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "coverages": coverages,
        "noise_dist": noise_dist,
        "noise_scale": noise_scale,
        "n_seeds": agg["n_seeds"],
        "test_mse_mean": agg["test_mse_mean"],
        "test_mse_se": agg["test_mse_se"],
    }
    for name in agg["mse_mean"]:
        save_dict[f"mse_mean_{name}"] = agg["mse_mean"][name]
        save_dict[f"mse_se_{name}"] = agg["mse_se"][name]
        save_dict[f"aurc_mean_{name}"] = agg["aurc_mean"][name]
        save_dict[f"aurc_se_{name}"] = agg["aurc_se"][name]
    
    out_file = output_path / f"results_{noise_dist}.npz"
    np.savez(out_file, **save_dict)
    print(f"  Aggregated results saved to {out_file}")
    
    print("\n" + "=" * 60)
    print(f"AURC summary (lower = better; n_seeds={agg['n_seeds']})")
    print("=" * 60)
    print(f"  Full-coverage test MSE: {agg['test_mse_mean']:.4f} "
          f"(+/- {agg['test_mse_se']:.4f})")
    print()
    print(f"{'UM':<22} {'AURC mean':<12} {'AURC SE':<12}")
    print("-" * 46)
    display = [
        ("random", "Random baseline"),
        ("var_tu", "Variance TU"),
        ("ent_tu", "Entropy TU"),
        ("quest_tu_01", "QUEST TU (a=0.1)"),
        ("quest_tu_05", "QUEST TU (a=0.5)"),
        ("quest_tu_g", "QUEST TU (global)"),
    ]
    for key, label in display:
        print(f"{label:<22} {agg['aurc_mean'][key]:<12.4f} "
              f"{agg['aurc_se'][key]:<12.4f}")
    
    return agg


def main():
    parser = argparse.ArgumentParser(description="QUEST selective prediction experiment")
    parser.add_argument("--noise", type=str, default="all",
                        choices=["gaussian", "t5", "t3", "all"])
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs (-1 = all cores; 1 = serial)")
    parser.add_argument("--n-coverage-points", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    if args.noise == "all":
        noise_dists = ["gaussian", "t5", "t3"]
    else:
        noise_dists = [args.noise]
    
    for nd in noise_dists:
        run_experiment(
            noise_dist=nd,
            noise_scale=args.noise_scale,
            n_train=args.n_train,
            n_test=args.n_test,
            M=args.M,
            n_epochs=args.n_epochs,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            batch_size=args.batch_size,
            lr=args.lr,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
            n_jobs=args.n_jobs,
            n_coverage_points=args.n_coverage_points,
            output_dir=args.output_dir,
            device=args.device,
        )
    
    print("\n" + "=" * 60)
    print("All experiments complete. Results saved to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
