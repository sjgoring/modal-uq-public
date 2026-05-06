"""
Selective prediction experiment for QUEST framework with MoE ensemble base.

Pipeline (per seed):
1. Generate (X, y) train/test data from the 1D heteroskedastic DGP.
2. Train an M-member MoE ensemble.
3. For each test point, compute all UMs (variance, entropy, QUEST oracle, QUEST C2).
4. Compute per-test-point log-density loss.
5. Rank test points by each UM, compute selective loss curves.
6. Compute AURC.

Loss function:
    loss(x) = log p*(y* | x) - log p*(y_hat | x)
where y* is the mode of p*(y | x) and y_hat is the mode of bar_p(y | x), both
found by 1D grid search on a shared grid.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from dgp import generate, make_true_density, true_conditional_density
from moe_ensemble import MoEEnsemble
from deep_ensemble import DeepEnsemble
from predictive import GridDensity1D
from measures import (
    variance_au, variance_eu, variance_tu,
    entropy_au, entropy_eu, entropy_tu,
    quest_au_local, quest_au_global,
    quest_eu_local, quest_eu_global,
    quest_tu_local, quest_tu_global,
    quest_au_local_c2, quest_au_global_c2,
    quest_tu_local_c2, quest_tu_global_c2,
    quest_eu_local_c2, quest_eu_global_c2,
)


UM_KEYS = [
    "var_au", "var_eu", "var_tu",
    "ent_au", "ent_eu", "ent_tu",
    "quest_au_01", "quest_eu_01", "quest_tu_01",
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
    quest_eu_01: float = 0.0
    quest_tu_01: float = 0.0
    quest_au_g: float = 0.0
    quest_eu_g: float = 0.0
    quest_tu_g: float = 0.0


def compute_all_measures(
    ensemble: MoEEnsemble,
    X_test: np.ndarray,
    noise_dist: str,
    estimator: str = "oracle",
    n_alpha_global: int = 30,
    verbose: bool = False,
) -> list[TestPointMeasures]:
    """Compute all UMs for every test point.
    
    Args:
        estimator: "oracle" (uses true p_theta_star) or "c2" (uses ensemble bar_p).
    """
    if estimator not in {"oracle", "c2"}:
        raise ValueError(f"Unknown estimator: {estimator!r}")
    
    n_test = X_test.shape[0]
    measures = []
    
    for i in range(n_test):
        x = X_test[i]
        m = TestPointMeasures()
        pred = ensemble.predictive_distribution(x)
        
        # Variance and entropy (estimator-independent: standard C-row decomposition)
        m.var_au = variance_au(pred)
        m.var_eu = variance_eu(pred)
        m.var_tu = variance_tu(pred)
        m.ent_au = entropy_au(pred)
        m.ent_eu = entropy_eu(pred)
        m.ent_tu = entropy_tu(pred)
        
        # QUEST: dispatch on estimator
        if estimator == "oracle":
            true_dist = make_true_density(x, noise_dist)
            theta_samples = ensemble.parameter_samples(x)
            m.quest_au_01 = quest_au_local(true_dist, alpha=0.1)
            m.quest_eu_01 = quest_eu_local(theta_samples, alpha=0.1)
            m.quest_tu_01 = quest_tu_local(true_dist, pred, alpha=0.1)
            m.quest_au_g = quest_au_global(true_dist, n_alpha=n_alpha_global)
            m.quest_eu_g = quest_eu_global(theta_samples, n_alpha=n_alpha_global)
            m.quest_tu_g = quest_tu_global(true_dist, pred, n_alpha=n_alpha_global)
        elif estimator == "c2":
            m.quest_au_01 = quest_au_local_c2(pred, alpha=0.1)
            m.quest_tu_01 = quest_tu_local_c2(pred, alpha=0.1)
            m.quest_eu_01 = m.quest_tu_01 - m.quest_au_01
            m.quest_au_g = quest_au_global_c2(pred, n_alpha=n_alpha_global)
            m.quest_tu_g = quest_tu_global_c2(pred, n_alpha=n_alpha_global)
            m.quest_eu_g = m.quest_tu_g - m.quest_au_g
        
        measures.append(m)
        if verbose and (i + 1) % 100 == 0:
            print(f"    Computed measures for {i+1}/{n_test}")
    
    return measures


def all_um_arrays(measures: list[TestPointMeasures]) -> dict[str, np.ndarray]:
    return {k: np.array([getattr(m, k) for m in measures]) for k in UM_KEYS}


def compute_log_density_loss(
    X_test: np.ndarray,
    ensemble: MoEEnsemble,
    noise_dist: str,
) -> np.ndarray:
    """Per-test-point loss: log p*(y* | x) - log p*(y_hat | x).
    
    y* and y_hat are modes of p*(.|x) and bar_p(.|x), found by argmax on the
    same grid (the one from make_true_density).
    """
    n_test = X_test.shape[0]
    losses = np.zeros(n_test)
    
    for i in range(n_test):
        x = X_test[i]
        true_dist = make_true_density(x, noise_dist)
        y_grid = true_dist.y_grid
        true_density_on_grid = true_dist.density_values
        
        idx_true = int(np.argmax(true_density_on_grid))
        y_star = float(y_grid[idx_true])
        log_p_star_at_true = float(np.log(max(true_density_on_grid[idx_true], 1e-300)))
        
        pred = ensemble.predictive_distribution(x)
        pred_density_on_grid = pred.density(y_grid)
        idx_pred = int(np.argmax(pred_density_on_grid))
        y_hat = float(y_grid[idx_pred])
        
        p_star_at_pred = float(true_conditional_density(
            np.array([y_hat]), x, noise_dist,
        )[0])
        log_p_star_at_pred = float(np.log(max(p_star_at_pred, 1e-300)))
        
        losses[i] = log_p_star_at_true - log_p_star_at_pred
    
    return losses


def selective_loss_curve(
    losses: np.ndarray,
    uncertainties: np.ndarray,
    coverages: np.ndarray,
) -> np.ndarray:
    """Mean loss on the c-fraction of lowest-uncertainty points, for each c."""
    n = len(losses)
    sort_idx = np.argsort(uncertainties)
    sorted_losses = losses[sort_idx]
    out = np.empty(len(coverages))
    for i, c in enumerate(coverages):
        k = max(2, int(round(c * n)))
        out[i] = float(np.mean(sorted_losses[:k]))
    return out


def aurc(coverages: np.ndarray, loss_values: np.ndarray) -> float:
    return float(np.trapz(loss_values, coverages))


def build_and_fit_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    seed: int,
    M: int,
    K: int,
    # Deep-ensemble-only:
    hidden_dim: int = 32,
    n_hidden: int = 2,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    entropy_bonus: float = 0.0,
    # MoE-only:
    bootstrap: bool = True,
):
    """Construct and train an ensemble of the chosen type.
    
    Returns an object exposing predictive_distribution() and parameter_samples().
    """
    if model == "deep":
        ens = DeepEnsemble(
            input_dim=X.shape[1], M=M, hidden_dim=hidden_dim, n_hidden=n_hidden, K=K,
        )
        ens.fit(
            X, y,
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            base_seed=seed, verbose=False,
            entropy_bonus=entropy_bonus,
        )
        return ens
    elif model == "moe":
        ens = MoEEnsemble(M=M, n_experts=K, bootstrap=bootstrap)
        ens.fit(X, y, base_seed=seed)
        return ens
    else:
        raise ValueError(f"Unknown model: {model!r}")


def run_single_seed(
    seed: int,
    noise_dist: str,
    n_train: int,
    n_test: int,
    M: int,
    K: int,
    coverages: np.ndarray,
    estimator: str = "oracle",
    model: str = "deep",
    # Deep ensemble hyperparameters:
    hidden_dim: int = 32,
    n_hidden: int = 2,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    entropy_bonus: float = 0.0,
    # MoE hyperparameters:
    bootstrap: bool = True,
) -> dict:
    X_train, y_train = generate(n=n_train, noise_dist=noise_dist, seed=seed)
    X_test, _ = generate(n=n_test, noise_dist=noise_dist, seed=seed + 100000)
    
    ensemble = build_and_fit_model(
        X_train, y_train,
        model=model, seed=seed, M=M, K=K,
        hidden_dim=hidden_dim, n_hidden=n_hidden,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        entropy_bonus=entropy_bonus,
        bootstrap=bootstrap,
    )
    
    losses = compute_log_density_loss(X_test, ensemble, noise_dist)
    test_loss_mean = float(losses.mean())
    
    measures = compute_all_measures(
        ensemble, X_test, noise_dist, estimator=estimator, verbose=False,
    )
    ums = all_um_arrays(measures)
    
    loss_curves = {}
    aurcs = {}
    for name, vals in ums.items():
        c = selective_loss_curve(losses, vals, coverages)
        loss_curves[name] = c
        aurcs[name] = aurc(coverages, c)
    
    rng = np.random.default_rng(seed + 999999)
    random_um = rng.uniform(size=n_test)
    rand_curve = selective_loss_curve(losses, random_um, coverages)
    loss_curves["random"] = rand_curve
    aurcs["random"] = aurc(coverages, rand_curve)
    
    return {
        "seed": seed,
        "test_loss_mean": test_loss_mean,
        "loss_curves": loss_curves,
        "aurcs": aurcs,
    }


def aggregate_seeds(per_seed_results: list[dict]) -> dict:
    K = len(per_seed_results)
    if K == 0:
        return {}
    
    um_names = list(per_seed_results[0]["loss_curves"].keys())
    loss_mean, loss_se = {}, {}
    aurc_mean, aurc_se = {}, {}
    
    for name in um_names:
        curves = np.stack([r["loss_curves"][name] for r in per_seed_results], axis=0)
        loss_mean[name] = curves.mean(axis=0)
        loss_se[name] = (curves.std(axis=0, ddof=1) / np.sqrt(K)
                         if K > 1 else np.zeros_like(curves[0]))
        
        aurc_vals = np.array([r["aurcs"][name] for r in per_seed_results])
        aurc_mean[name] = float(aurc_vals.mean())
        aurc_se[name] = (float(aurc_vals.std(ddof=1) / np.sqrt(K)) if K > 1 else 0.0)
    
    test_losses = np.array([r["test_loss_mean"] for r in per_seed_results])
    return {
        "loss_mean": loss_mean,
        "loss_se": loss_se,
        "aurc_mean": aurc_mean,
        "aurc_se": aurc_se,
        "test_loss_mean": float(test_losses.mean()),
        "test_loss_se": (float(test_losses.std(ddof=1) / np.sqrt(K))
                         if K > 1 else 0.0),
        "n_seeds": K,
    }


def run_experiment(
    noise_dist: str,
    n_train: int,
    n_test: int,
    M: int,
    K: int,
    n_seeds: int,
    base_seed: int,
    n_jobs: int,
    output_dir: str,
    n_coverage_points: int,
    estimator: str = "oracle",
    model: str = "deep",
    # Deep ensemble hyperparameters:
    hidden_dim: int = 32,
    n_hidden: int = 2,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    entropy_bonus: float = 0.0,
    # MoE hyperparameters:
    bootstrap: bool = True,
) -> dict:
    print("=" * 60)
    print(f"Experiment: 1D DGP, {noise_dist} noise")
    print(f"  model={model}, M={M}, K={K}")
    if model == "deep":
        print(f"  hidden_dim={hidden_dim}, n_hidden={n_hidden}, n_epochs={n_epochs}, "
              f"entropy_bonus={entropy_bonus}")
    print(f"  n_train={n_train}, n_test={n_test}, n_seeds={n_seeds}, "
          f"n_jobs={n_jobs}, estimator={estimator}")
    print("=" * 60)
    
    coverages = np.linspace(0.05, 1.0, n_coverage_points)
    t0 = time.time()
    seeds = [base_seed + i for i in range(n_seeds)]
    
    common_kwargs = dict(
        noise_dist=noise_dist, n_train=n_train, n_test=n_test,
        M=M, K=K, coverages=coverages, estimator=estimator, model=model,
        hidden_dim=hidden_dim, n_hidden=n_hidden, n_epochs=n_epochs,
        batch_size=batch_size, lr=lr, entropy_bonus=entropy_bonus,
        bootstrap=bootstrap,
    )
    
    if n_jobs == 1:
        per_seed_results = []
        for s in seeds:
            t_seed = time.time()
            res = run_single_seed(seed=s, **common_kwargs)
            per_seed_results.append(res)
            print(f"    Seed {s} took {time.time() - t_seed:.1f}s "
                  f"(test loss = {res['test_loss_mean']:.3f})")
    else:
        per_seed_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_seed)(seed=s, **common_kwargs) for s in seeds
        )
    
    elapsed = time.time() - t0
    print(f"  All seeds done in {elapsed:.1f}s ({elapsed / n_seeds:.1f}s/seed avg).")
    
    agg = aggregate_seeds(per_seed_results)
    agg["coverages"] = coverages
    agg["noise_dist"] = noise_dist
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "coverages": coverages,
        "noise_dist": noise_dist,
        "estimator": estimator,
        "n_seeds": agg["n_seeds"],
        "test_loss_mean": agg["test_loss_mean"],
        "test_loss_se": agg["test_loss_se"],
    }
    for name in agg["loss_mean"]:
        save_dict[f"loss_mean_{name}"] = agg["loss_mean"][name]
        save_dict[f"loss_se_{name}"] = agg["loss_se"][name]
        save_dict[f"aurc_mean_{name}"] = agg["aurc_mean"][name]
        save_dict[f"aurc_se_{name}"] = agg["aurc_se"][name]
    
    out_file = output_path / f"results_{noise_dist}_{estimator}.npz"
    np.savez(out_file, **save_dict)
    print(f"  Aggregated results saved to {out_file}")
    
    print(f"\n  Full-coverage test loss: {agg['test_loss_mean']:.4f} "
          f"(+/- {agg['test_loss_se']:.4f})")
    print(f"\n{'UM':<22} {'AURC mean':<12} {'AURC SE':<12}")
    print("-" * 46)
    display = [
        ("random", "Random baseline"),
        ("var_tu", "Variance TU"),
        ("ent_tu", "Entropy TU"),
        ("quest_tu_01", "QUEST TU (a=0.1)"),
        ("quest_tu_g", "QUEST TU (global)"),
    ]
    for key, label in display:
        print(f"{label:<22} {agg['aurc_mean'][key]:<12.4f} "
              f"{agg['aurc_se'][key]:<12.4f}")
    
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", choices=["gaussian", "bimodal", "skewed", "all"],
                        default="all")
    parser.add_argument("--estimator", choices=["oracle", "c2", "all"], default="all")
    parser.add_argument("--model", choices=["deep", "moe"], default="deep",
                        help="Model class for the ensemble. 'deep' uses a deep "
                             "ensemble of K-component Gaussian-mixture-output nets. "
                             "'moe' uses cgmm Mixture-of-Experts regressors.")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--K", type=int, default=3,
                        help="Components per ensemble member (deep: mixture comps; "
                             "moe: experts).")
    # Deep ensemble flags
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    # MoE flags
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="(MoE only) disable bootstrap resampling.")
    # General
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-coverage-points", type=int, default=20)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    
    if args.noise == "all":
        noise_list = ["gaussian", "bimodal", "skewed"]
    else:
        noise_list = [args.noise]
    estimator_list = ["oracle", "c2"] if args.estimator == "all" else [args.estimator]
    
    for nd in noise_list:
        for est in estimator_list:
            run_experiment(
                noise_dist=nd,
                n_train=args.n_train,
                n_test=args.n_test,
                M=args.M,
                K=args.K,
                n_seeds=args.n_seeds,
                base_seed=args.base_seed,
                n_jobs=args.n_jobs,
                output_dir=args.output_dir,
                n_coverage_points=args.n_coverage_points,
                estimator=est,
                model=args.model,
                hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden,
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                entropy_bonus=args.entropy_bonus,
                bootstrap=not args.no_bootstrap,
            )
    
    print("\n" + "=" * 60)
    print(f"All experiments complete. Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
