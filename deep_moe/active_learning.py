"""
Active learning experiment for QUEST framework with MoE ensemble base.

Pipeline (per seed):
1. Generate a pool set and a held-out evaluation set from the 1D DGP.
2. Sample an initial labeled subset D_init from the pool.
3. Train an M-member ensemble on the labeled subset.
4. Score unlabeled points with the existing B1 uncertainty measures.
5. Acquire labels by ranking the pool with each measure.
6. Retrain after each acquisition round and evaluate mode absolute error.

The uncertainty calculations below are intentionally unchanged: they already
implement the B1 inferential choices with the posterior predictive as the
predictive distribution and the MLE member as the truth approximation.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from joblib import Parallel, delayed

from dgp import generate, make_true_density, true_conditional_density
from moe_ensemble import MoEEnsemble
from deep_ensemble import DeepEnsemble
from predictive import GaussianMixture1D, GridDensity1D
from mpe_dataset import load_mpe_dataset
from measures import (
    variance_au, variance_eu, variance_tu,
    entropy_au, entropy_eu, entropy_tu,
    quest_au_local, quest_au_global,
    quest_eu_local, quest_eu_global,
    quest_tu_local, quest_tu_global,
    # C2 plug-ins are commented out in measures.py.
    # quest_au_local_c2, quest_au_global_c2,
    # quest_tu_local_c2, quest_tu_global_c2,
    # quest_eu_local_c2, quest_eu_global_c2,
)


# UM_KEYS = [
#     "var_au", "var_eu", "var_tu",
#     "ent_au", "ent_eu", "ent_tu",
#     "quest_au_01", "quest_eu_01", "quest_tu_01",
#     "quest_au_g", "quest_eu_g", "quest_tu_g",
# ]

# Testing only
UM_KEYS = ["var_au"]


def setup_logger(output_dir: str, seed: int, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to both console and a seed-specific log file.
    
    Args:
        output_dir: Directory to save log files in.
        seed: Seed number used to create unique log filename.
        level: Logging level (default: INFO).
    
    Returns:
        A logger instance configured with file and stream handlers.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger_name = f"seed_{seed}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler: append mode for robustness
    log_file = output_path / f"log_seed_{seed}.txt"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    
    # Stream handler: write to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    
    # Formatter: include timestamp and seed context
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [seed %(name)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger


def log_print(logger: logging.Logger, msg: str) -> None:
    """Log a message to console and file via the provided logger.
    
    Handles None logger gracefully (falls back to print).
    """
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


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


def select_best_member(
    ensemble, X_train: np.ndarray, y_train: np.ndarray
) -> int:
    """Select the ensemble member with highest mean training log-likelihood.
    
    Returns the index m_hat in {0, ..., M-1}.
    """
    M = ensemble.M
    log_likelihoods = np.zeros(M)
    for m in range(M):
        log_likelihoods[m] = ensemble.member_log_likelihood(X_train, y_train, m)
    return int(np.argmax(log_likelihoods))


def make_truth_approximation(
    x: np.ndarray,
    noise_dist: str,
    estimator: str,
    ensemble=None,
    m_hat: int = -1,
):
    """Build the truth approximation hat_p_star at input x.
    
    For 'oracle': returns a GridDensity1D for the true conditional density.
    For 'mle': returns a GaussianMixture1D for the m_hat-th ensemble member's
               predictive at x.
    """
    if estimator == "oracle":
        return make_true_density(x, noise_dist)
    elif estimator == "mle":
        return ensemble.member_distribution(x, m_hat)
    else:
        raise ValueError(f"Unknown estimator: {estimator!r}")


def compute_all_measures(
    ensemble,
    X_test: np.ndarray,
    noise_dist: str,
    estimator: str = "oracle",
    m_hat: int = -1,
    n_alpha_global: int = 30,
    verbose: bool = False,
) -> list[TestPointMeasures]:
    """Compute all UMs for every test point under the B1 framework.
    
    All UMs (variance, entropy, QUEST) take both:
      - hat_p_star: truth approximation (true density for oracle; one ensemble
                    member's predictive for MLE)
      - bar_p:      ensemble's full posterior predictive (M*K mixture)
    
    Args:
        estimator: "oracle" or "mle".
        m_hat: index of the selected member (used only when estimator="mle").
    """
    if estimator not in {"oracle", "mle"}:
        raise ValueError(f"Unknown estimator: {estimator!r}")
    if estimator == "mle" and m_hat < 0:
        raise ValueError("MLE estimator requires m_hat >= 0.")
    
    n_test = X_test.shape[0]
    measures = []
    
    for i in range(n_test):
        x = X_test[i]
        m = TestPointMeasures()
        bar_p = ensemble.predictive_distribution(x)
        p_hat_star = make_truth_approximation(
            x, noise_dist, estimator, ensemble=ensemble, m_hat=m_hat,
        )
        
        # Variance UMs (B1 framework)
        m.var_au = variance_au(p_hat_star)
        m.var_eu = variance_eu(p_hat_star, bar_p)
        m.var_tu = variance_tu(p_hat_star, bar_p)
        
        # Entropy UMs (B1 framework)
        m.ent_au = entropy_au(p_hat_star)
        m.ent_eu = entropy_eu(p_hat_star, bar_p)
        m.ent_tu = entropy_tu(p_hat_star, bar_p)
        
        # QUEST UMs (oracle formulas, with hat_p_star playing the role of truth)
        theta_samples = ensemble.parameter_samples(x)
        m.quest_au_01 = quest_au_local(p_hat_star, alpha=0.1)
        m.quest_eu_01 = quest_eu_local(theta_samples, alpha=0.1)
        m.quest_tu_01 = quest_tu_local(p_hat_star, bar_p, alpha=0.1)
        m.quest_au_g = quest_au_global(p_hat_star, n_alpha=n_alpha_global)
        m.quest_eu_g = quest_eu_global(theta_samples, n_alpha=n_alpha_global)
        m.quest_tu_g = quest_tu_global(p_hat_star, bar_p, n_alpha=n_alpha_global)
        
        measures.append(m)
        if verbose and (i + 1) % 100 == 0:
            print(f"    Computed measures for {i+1}/{n_test}")
    
    return measures


def all_um_arrays(measures: list[TestPointMeasures]) -> dict[str, np.ndarray]:
    return {k: np.array([getattr(m, k) for m in measures]) for k in UM_KEYS}


def _budget_schedule(n_pool: int, init_size: int, rounds: int) -> list[int]:
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if init_size <= 0:
        raise ValueError("init_size must be positive")
    if init_size >= n_pool:
        raise ValueError("init_size must be smaller than the pool size")

    remaining = n_pool - init_size
    base = remaining // rounds
    remainder = remaining % rounds
    return [base + 1 if idx < remainder else base for idx in range(rounds)]


def _predictive_mode(pred, true_grid: np.ndarray) -> float:
    density_on_grid = pred.density(true_grid)
    return float(true_grid[int(np.argmax(density_on_grid))])


def compute_mode_absolute_error(
    ensemble,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    noise_dist: str,
) -> np.ndarray:
    """Per-point absolute error between predictive mode and held-out y."""
    errors = np.zeros(X_eval.shape[0])
    for i, x in enumerate(X_eval):
        true_dist = make_true_density(x, noise_dist)
        pred = ensemble.predictive_distribution(x)
        y_mode = _predictive_mode(pred, true_dist.y_grid)
        errors[i] = abs(y_mode - float(y_eval[i]))
    return errors


def _predictive_cdf(pred, y: np.ndarray) -> np.ndarray:
    """Evaluate the predictive CDF for a 1D predictive distribution."""
    y = np.atleast_1d(np.asarray(y, dtype=float))

    if isinstance(pred, GaussianMixture1D):
        component_cdfs = stats.norm.cdf(
            y[:, None], loc=pred.mus[None, :], scale=pred.sigmas[None, :]
        )
        return component_cdfs @ pred.weights

    if isinstance(pred, GridDensity1D):
        y_grid = np.asarray(pred.y_grid, dtype=float)
        density = np.asarray(pred.density_values, dtype=float)
        cdf_grid = np.concatenate(
            ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(y_grid)))
        )
        if cdf_grid[-1] > 0:
            cdf_grid = cdf_grid / cdf_grid[-1]
        return np.interp(y, y_grid, cdf_grid, left=0.0, right=1.0)

    if hasattr(pred, "cdf") and callable(pred.cdf):
        return np.asarray(pred.cdf(y), dtype=float)

    if hasattr(pred, "grid") and callable(pred.grid) and hasattr(pred, "density"):
        y_grid = np.asarray(pred.grid(), dtype=float)
        density = np.asarray(pred.density(y_grid), dtype=float)
        cdf_grid = np.concatenate(
            ([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(y_grid)))
        )
        if cdf_grid[-1] > 0:
            cdf_grid = cdf_grid / cdf_grid[-1]
        return np.interp(y, y_grid, cdf_grid, left=0.0, right=1.0)

    raise TypeError(f"Unsupported predictive distribution type: {type(pred)!r}")


def compute_calibration_metric(
    ensemble,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    noise_dist: str,
    n_grid: int = 100,
) -> float:
    """Empirical calibration error based on PIT values.

    For each evaluation pair (x_i, y_i), compute the predictive CDF value
    p_i = P(Y <= y_i | x_i). The calibration curve is then the empirical CDF
    of the p_i values evaluated on a uniform grid p_j in [0, 1]. The returned
    scalar is the integrated squared deviation

        cal = \int_0^1 (F_hat(p) - p)^2 dp,

    where F_hat is the empirical CDF of the PIT values. This matches the
    paper's discrete form cal = sum_j w_j (p_j - \hat p_j)^2 with trapezoidal
    weights on a uniform grid.
    """
    del noise_dist  # The metric depends on the predictive CDF and observed y only.

    y_eval = np.asarray(y_eval, dtype=float).reshape(-1)
    if y_eval.size == 0:
        return float("nan")

    pit_values = np.empty(y_eval.size, dtype=float)
    for i, x in enumerate(X_eval):
        pred = ensemble.predictive_distribution(x)
        pit_values[i] = float(_predictive_cdf(pred, np.array([y_eval[i]]))[0])

    pit_values = np.clip(pit_values, 0.0, 1.0)
    p_grid = np.linspace(0.0, 1.0, n_grid)
    empirical_cdf = np.array([np.mean(pit_values <= p) for p in p_grid], dtype=float)
    return float(np.trapz((empirical_cdf - p_grid) ** 2, p_grid))


def _evaluate_learning_quality(
    ensemble,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    noise_dist: str,
) -> dict[str, np.ndarray]:
    mode_abs_error = compute_mode_absolute_error(ensemble, X_eval, y_eval, noise_dist)
    try:
        calibration = compute_calibration_metric(ensemble, X_eval, y_eval, noise_dist)
    except NotImplementedError:
        calibration = np.nan
    return {
        "mode_abs_error": mode_abs_error,
        "calibration": calibration,
    }


def _safe_nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _safe_nansem(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return 0.0
    return float(finite.std(ddof=1) / np.sqrt(finite.size))


def _select_top_indices(scores: np.ndarray, batch_size: int) -> np.ndarray:
    return np.argsort(-scores)[:batch_size]


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
    d_init: int,
    n_rounds: int,
    M: int,
    K: int,
    model: str = "deep",
    dataset: str = "dgp",
    logger: logging.Logger = None,
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
    if dataset == "mpe":
        ds = load_mpe_dataset(split_seed=seed)
        X_train_full = ds.X_train
        y_train_samples_full = ds.y_train
        X_test_full = ds.X_test
        y_test_samples_full = ds.y_test
        
        # Subsample training pool if n_train is smaller than available
        n_train_effective = min(n_train, X_train_full.shape[0])
        rng_subset_train = np.random.default_rng(seed + 500000)
        train_subset_idx = rng_subset_train.choice(
            X_train_full.shape[0], size=n_train_effective, replace=False
        )
        X_pool = X_train_full[train_subset_idx]
        y_pool_samples = y_train_samples_full[train_subset_idx]
        # Convert per-row sample vectors to scalar targets (modes)
        y_pool = ds.gt(X_pool, y_pool_samples)
        
        # Subsample evaluation set if n_test is smaller than available
        n_test_effective = min(n_test, X_test_full.shape[0])
        rng_subset_test = np.random.default_rng(seed + 500001)
        test_subset_idx = rng_subset_test.choice(
            X_test_full.shape[0], size=n_test_effective, replace=False
        )
        X_eval = X_test_full[test_subset_idx]
        y_eval_samples = y_test_samples_full[test_subset_idx]
        # Convert per-row sample vectors to scalar targets (modes)
        y_eval = ds.gt(X_eval, y_eval_samples)
    else:
        X_pool, y_pool = generate(n=n_train, noise_dist=noise_dist, seed=seed)
        X_eval, y_eval = generate(n=n_test, noise_dist=noise_dist, seed=seed + 100000)

    # Use actual pool size from loaded/generated data, not CLI parameter
    n_pool = X_pool.shape[0]
    budget_schedule = _budget_schedule(n_pool=n_pool, init_size=d_init, rounds=n_rounds)
    rng = np.random.default_rng(seed)
    pool_order = rng.permutation(n_pool)
    init_indices = pool_order[:d_init].tolist()
    pool_indices = pool_order[d_init:].tolist()

    measure_names = UM_KEYS + ["random"]
    per_measure_results = []

    for measure_name in measure_names:
        labeled_indices = init_indices.copy()
        unlabeled_indices = pool_indices.copy()
        curve = []

        for round_idx in range(n_rounds + 1):
            log_print(logger, f" Time {time.strftime('%H:%M:%S')}, Seed {seed}, measure {measure_name}, begin round {round_idx}/{n_rounds}")
            ensemble = build_and_fit_model(
                X_pool[labeled_indices],
                y_pool[labeled_indices],
                model=model,
                seed=seed + round_idx,
                M=M,
                K=K,
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                entropy_bonus=entropy_bonus,
                bootstrap=bootstrap,
            )
            log_print(logger, f" Time {time.strftime('%H:%M:%S')}, Seed {seed}, measure {measure_name}, finished training model for round {round_idx}/{n_rounds}")

            eval_metrics = _evaluate_learning_quality(ensemble, X_eval, y_eval, noise_dist)
            curve.append(
                {
                    "round": round_idx,
                    "labelled_budget": len(labeled_indices),
                    "mode_abs_error": float(np.mean(eval_metrics["mode_abs_error"])),
                        "calibration": float(eval_metrics["calibration"]),
                }
            )

            log_print(logger, f" Time {time.strftime('%H:%M:%S')}, Seed {seed}, measure {measure_name}, evaluated round {round_idx}/{n_rounds} with mode_abs_error={curve[-1]['mode_abs_error']:.4f}")

            if round_idx == n_rounds or len(unlabeled_indices) == 0:
                continue

            batch_size_round = min(budget_schedule[round_idx], len(unlabeled_indices))
            if batch_size_round <= 0:
                continue

            m_hat = select_best_member(ensemble, X_pool[labeled_indices], y_pool[labeled_indices])

            if measure_name == "random":
                scores = rng.uniform(size=len(unlabeled_indices))
            else:
                measures = compute_all_measures(
                    ensemble,
                    X_pool[unlabeled_indices],
                    noise_dist,
                    estimator="mle",
                    m_hat=m_hat,
                    verbose=False,
                )
                scores = all_um_arrays(measures)[measure_name]

            log_print(logger, f" Time {time.strftime('%H:%M:%S')}, Seed {seed}, measure {measure_name}, computed scores for round {round_idx}/{n_rounds}, selecting top {batch_size_round} points to label")

            selected_local = _select_top_indices(scores, batch_size_round)
            selected_indices = [unlabeled_indices[idx] for idx in selected_local]
            labeled_indices.extend(selected_indices)
            selected_set = set(selected_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_set]

            log_print(logger, f" Time {time.strftime('%H:%M:%S')}, Seed {seed}, measure {measure_name}, completed round {round_idx}/{n_rounds} with {len(labeled_indices)} labeled points and {len(unlabeled_indices)} unlabeled points remaining")
            log_print(logger, "-" * 80)

        per_measure_results.append({"measure": measure_name, "curve": curve})

    return {
        "seed": seed,
        "per_measure_results": per_measure_results,
        "budget_schedule": budget_schedule,
        "d_init": d_init,
        "n_rounds": n_rounds,
    }


def aggregate_seeds(per_seed_results: list[dict]) -> dict:
    n_seeds = len(per_seed_results)
    if n_seeds == 0:
        return {}

    measure_names = [r["measure"] for r in per_seed_results[0]["per_measure_results"]]
    metric_names = ["mode_abs_error", "calibration"]

    metric_mean = {metric: {} for metric in metric_names}
    metric_se = {metric: {} for metric in metric_names}
    final_metric_mean = {metric: {} for metric in metric_names}
    final_metric_se = {metric: {} for metric in metric_names}

    for measure_name in measure_names:
        seed_curve_map = [
            next(r for r in seed_result["per_measure_results"] if r["measure"] == measure_name)["curve"]
            for seed_result in per_seed_results
        ]
        for metric_name in metric_names:
            metric_curves = np.array(
                [[row[metric_name] for row in curve] for curve in seed_curve_map],
                dtype=float,
            )
            metric_mean[metric_name][measure_name] = metric_curves.mean(axis=0)
            metric_se[metric_name][measure_name] = (
                metric_curves.std(axis=0, ddof=1) / np.sqrt(n_seeds)
                if n_seeds > 1 else np.zeros_like(metric_curves[0])
            )
            final_vals = metric_curves[:, -1]
            final_metric_mean[metric_name][measure_name] = _safe_nanmean(final_vals)
            final_metric_se[metric_name][measure_name] = _safe_nansem(final_vals)

    return {
        "metric_mean": metric_mean,
        "metric_se": metric_se,
        "final_metric_mean": final_metric_mean,
        "final_metric_se": final_metric_se,
        "measure_names": measure_names,
        "metric_names": metric_names,
        "n_seeds": n_seeds,
    }


def run_experiment(
    noise_dist: str,
    n_train: int,
    n_test: int,
    d_init: int,
    n_rounds: int,
    M: int,
    K: int,
    n_seeds: int,
    base_seed: int,
    n_jobs: int,
    output_dir: str,
    model: str = "deep",
    dataset: str = "dgp",
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
    # Create main logger for experiment orchestration
    main_logger = setup_logger(output_dir, seed=0, level=logging.INFO)
    
    log_print(main_logger, "=" * 60)
    log_print(main_logger, f"Experiment: active learning, 1D DGP, {noise_dist} noise")
    log_print(main_logger, f"  model={model}, M={M}, K={K}")
    if model == "deep":
        log_print(main_logger, f"  hidden_dim={hidden_dim}, n_hidden={n_hidden}, n_epochs={n_epochs}, "
              f"entropy_bonus={entropy_bonus}")
    log_print(main_logger, f"  pool_size={n_train}, eval_size={n_test}, d_init={d_init}, n_rounds={n_rounds}")
    log_print(main_logger, f"  n_seeds={n_seeds}, n_jobs={n_jobs}")
    log_print(main_logger, "=" * 60)
    
    t0 = time.time()
    seeds = [base_seed + i for i in range(n_seeds)]
    
    common_kwargs = dict(
        noise_dist=noise_dist, n_train=n_train, n_test=n_test,
        d_init=d_init, n_rounds=n_rounds, M=M, K=K, model=model,
        hidden_dim=hidden_dim, n_hidden=n_hidden, n_epochs=n_epochs,
        batch_size=batch_size, lr=lr, entropy_bonus=entropy_bonus,
        bootstrap=bootstrap,
    )
    
    if n_jobs == 1:
        per_seed_results = []
        for s in seeds:
            seed_logger = setup_logger(output_dir, seed=s, level=logging.INFO)
            t_seed = time.time()
            res = run_single_seed(seed=s, dataset=dataset, logger=seed_logger, **common_kwargs)
            per_seed_results.append(res)
            log_print(main_logger, f"    Seed {s} took {time.time() - t_seed:.1f}s")
    else:
        per_seed_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_seed)(seed=s, dataset=dataset, logger=setup_logger(output_dir, seed=s), **common_kwargs) for s in seeds
        )
    
    elapsed = time.time() - t0
    log_print(main_logger, f"  All seeds done in {elapsed:.1f}s ({elapsed / n_seeds:.1f}s/seed avg.)")
    
    agg = aggregate_seeds(per_seed_results)
    agg["noise_dist"] = noise_dist
    agg["d_init"] = d_init
    agg["n_rounds"] = n_rounds
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "dataset": dataset,
        "noise_dist": noise_dist,
        "d_init": d_init,
        "n_rounds": n_rounds,
        "n_seeds": agg["n_seeds"],
    }
    for metric_name in agg["metric_names"]:
        for measure_name in agg["measure_names"]:
            save_dict[f"{metric_name}_mean_{measure_name}"] = agg["metric_mean"][metric_name][measure_name]
            save_dict[f"{metric_name}_se_{measure_name}"] = agg["metric_se"][metric_name][measure_name]
            save_dict[f"final_{metric_name}_mean_{measure_name}"] = agg["final_metric_mean"][metric_name][measure_name]
            save_dict[f"final_{metric_name}_se_{measure_name}"] = agg["final_metric_se"][metric_name][measure_name]
    
    out_name = "results_mpe_active_learning.npz" if dataset == "mpe" else f"results_{noise_dist}_active_learning.npz"
    out_file = output_path / out_name
    np.savez(out_file, **save_dict)
    log_print(main_logger, f"  Aggregated results saved to {out_file}")
    
    log_print(main_logger, "\nFinal-round mode absolute error (lower is better)")
    log_print(main_logger, f"{'Measure':<24} {'Mean':<12} {'SE':<12}")
    log_print(main_logger, "-" * 50)
    for measure_name in agg["measure_names"]:
        mean = agg["final_metric_mean"]["mode_abs_error"][measure_name]
        se = agg["final_metric_se"]["mode_abs_error"][measure_name]
        log_print(main_logger, f"{measure_name:<24} {mean:<12.4f} {se:<12.4f}")
    
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", choices=["gaussian", "bimodal", "skewed", "all"],
                        default="all")
    parser.add_argument("--model", choices=["deep", "moe"], default="deep",
                        help="Model class for the ensemble. 'deep' uses a deep "
                             "ensemble of K-component Gaussian-mixture-output nets. "
                             "'moe' uses cgmm Mixture-of-Experts regressors.")
    parser.add_argument("--n-train", type=int, default=1000,
                        help="Total pool size available for active learning. "
                             "For --dataset mpe, limits the available pool size (uses full dataset if larger).")
    parser.add_argument("--n-test", type=int, default=500,
                        help="Held-out evaluation set size. "
                             "For --dataset mpe, limits the evaluation set size (uses full dataset if larger).")
    parser.add_argument("--d-init", type=int, default=100,
                        help="Initial labeled set size D_init.")
    parser.add_argument("--n-rounds", type=int, default=10,
                        help="Number of acquisition rounds.")
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
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--dataset", choices=["dgp", "mpe"], default="dgp",
                        help="Data source: 'dgp' uses synthetic 1D DGP; 'mpe' loads the MPE .npz adapter.")
    args = parser.parse_args()
    
    if args.noise == "all":
        noise_list = ["gaussian", "bimodal", "skewed"]
    else:
        noise_list = [args.noise]
    
    for nd in noise_list:
        run_experiment(
            noise_dist=nd,
            n_train=args.n_train,
            n_test=args.n_test,
            d_init=args.d_init,
            n_rounds=args.n_rounds,
            M=args.M,
            K=args.K,
            n_seeds=args.n_seeds,
            base_seed=args.base_seed,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            model=args.model,
            dataset=args.dataset,
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
