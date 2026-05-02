import os
import json
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modal_uq.datasets.synthetic_constant_var import SyntheticConstantVarDataset
from modal_uq.datasets.moons_synthetic import MoonsSyntheticDataset
from modal_uq.models.condGMM import CondGMM
from modal_uq.models.mdn import MixtureDensityModel
from modal_uq.models.ensemble import Ensemble
from modal_uq.metrics.common import nll_from_density_at_truth
from modal_uq.metrics.mode_errors import modal_absolute_error, modal_squared_error


def make_small_dataset(seed, n_samples=200, test_size=0.3, source="MOONS"):
    ## Synth multi modal
    # ds = SyntheticMultiModalConditionalDataset(
    #     n_samples=int(n_samples/(1-test_size)), n_modes=2, n_features=1, seed_master=seed #here n_samples = n_train + n_test
    # )
    # X, y, _, _ = ds.get_data(
    #     pi_fn=ds.test_pi_fn,
    #     mu_fn=ds.test_mu_fn,
    #     sigma_fn=ds.test_sigma_fn,
    #     noise_fn=ds.test_no_fn,
    # )
    ## Moons
    if source == "MOONS":
        ds = MoonsSyntheticDataset(n_samples=int(n_samples/(1-test_size)), noise=0.1, random_state=seed)
        X, y = ds.sample()
    elif source == "SYNTH_MULTI_MODAL":
        ds = SyntheticConstantVarDataset(
            n_samples=n_samples
        )
        X, y, _, _, _ = ds.get_data(
        )
    else:
        raise ValueError(f"Unknown dataset: {source}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def run_single_trial(seed, train_size, model_cls, model_kwargs, metric="moae", n_samples=200):
    t0 = time.perf_counter()
    X_train, X_test, y_train, y_test = make_small_dataset(seed, n_samples=n_samples)
    # Subsample the training set to the requested size
    if train_size >= X_train.shape[0]:
        X_sub, y_sub = X_train, y_train
    else:
        X_sub, y_sub = X_train[:train_size], y_train[:train_size]

    # Instantiate model inside worker to avoid pickling issues
    if model_cls is MixtureDensityModel:
        model = model_cls(inferential_choice=None, **model_kwargs)
    else:
        # For GP and other SKLearn-like models, pass only relevant kwargs
        model = model_cls(**model_kwargs)

    model.fit(X_sub, y_sub)

    y_grid = model.default_y_grid(X_test, grid_points=128)

    metric = (metric).lower()
    if metric == "nll":
        dens = model.predict_density(X_test, y_grid)
        nlls = nll_from_density_at_truth(dens, y_grid, y_test)
        val = float(np.mean(nlls))
    elif metric in ("moae", "moae"):
        # modal absolute error
        y_mode = model.predict_mode(X_test, y_grid)
        val = float(modal_absolute_error(y_test, y_mode, {}))
    elif metric in ("mose", "msae", "mse"):
        # modal squared error / mean squared error of mode
        y_mode = model.predict_mode(X_test, y_grid)
        val = float(modal_squared_error(y_test, y_mode, {}))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    elapsed = time.perf_counter() - t0
    return val, elapsed

class ProgressEstimator:
        def __init__(self):
            self.sizes = []
            self.times = []

        def add_observation(self, size, elapsed):
            self.sizes.append(float(size))
            self.times.append(float(elapsed))

        def estimate_time_per_job(self, size):
            if len(self.sizes) >= 2:
                coeffs = np.polyfit(self.sizes, self.times, 1)
                est = float(coeffs[0] * size + coeffs[1])
                return max(est, 1e-6)
            if len(self.sizes) == 1:
                # assume linear proportional to size
                rate = self.times[0] / max(self.sizes[0], 1.0)
                return float(rate * size)
            return 0.0

        def estimate_remaining_wall_time(self, remaining_sizes, reps_per_size, n_jobs):
            total_seconds = 0.0
            for s in remaining_sizes:
                est = self.estimate_time_per_job(s)
                total_seconds += est * reps_per_size
            if n_jobs and n_jobs > 0:
                return total_seconds / float(n_jobs)
            return total_seconds

def format_seconds(sec):
    try:
        return str(timedelta(seconds=int(round(sec))))
    except Exception:
        return f"{sec:.1f}s"


def metric_metadata(metric):
    metric_key = (metric or "").lower()
    if metric_key == "nll":
        return metric_key, "Negative log-likelihood (NLL)"
    if metric_key == "moae":
        return metric_key, "Mode absolute error (MOAE)"
    if metric_key in ("mose", "msae", "mse"):
        return metric_key, "Mode squared error (MSE)"
    return metric_key, metric_key.upper()


def refresh_all_status(all_status, estimators, train_sizes, N_REPS, last_printed_lines, min_train):
    lines = []
    for name, sizes_map in all_status.items():
        lines.append(f"[learning_curves] {name}:")
        est = estimators.get(name)
        for s in train_sizes:
            status = sizes_map.get(int(s), "Not Started")
            eta_str = ""
            if status == "In Progress" and int(s) != min_train and est is not None:
                try:
                    if getattr(est, "sizes", None):
                        eta = est.estimate_time_per_job(int(s))
                        eta_str = f" ETA={format_seconds(eta)}"
                except Exception:
                    eta_str = ""
            lines.append(f"  size={int(s)} status={status}{eta_str}")

    if sys.stdout.isatty() and last_printed_lines.get("n", 0) > 0:
        last = last_printed_lines["n"]
        print(f"\x1b[{last}A", end="")
        print("\n".join(lines), flush=True)
    else:
        print("-" * 60)
        print("\n".join(lines), flush=True)

    last_printed_lines["n"] = len(lines)


def test_learning_curves(capsys):
    # CI-friendly defaults
    N_REPS = 3
    K_SIZES = 6
    N_SAMPLES = 100000 # Number of training samples, not total number of samples (i.e. n_train = n_total - n_test)
    MIN_TRAIN = 10
    METRIC = "nll"
    DATA = "SYNTH_MULTI_MODAL"

    # Faster dev defaults
    # N_REPS = 2
    # K_SIZES = 4
    # N_SAMPLES = 2000
    # MIN_TRAIN = 10
    # METRIC = "nll"
    # DATA = "SYNTH_MULTI_MODAL"

    # # Very quick testing values
    # N_REPS = 2
    # K_SIZES = 2
    # N_SAMPLES = 100
    # MIN_TRAIN = 10

    # Model configurations chosen to keep runtime modest for CI
    mdn_kwargs = {"hidden_dim": 32, "n_gaussians": 2, "epochs": 10} 
    # mdn_kwargs = {"hidden_dim": 1, "n_gaussians": 2, "epochs": 2} # for testing
    cond_gmm_kwargs = {"n_components": 2}

    # Evaluate two models: MDN and CondGMM.
    models = [(MixtureDensityModel, mdn_kwargs, "MixtureDensityModel"),
              (CondGMM, cond_gmm_kwargs, "CondGMM"),
              (Ensemble, {"n_members": 3, "base_model":"condgmm"}, "Ensemble")]
    
    # For testing only the MDN.
    # models = [(MixtureDensityModel, mdn_kwargs, "MixtureDensityModel")]

    # Parallel settings
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    # n_jobs = 1 # for testing
    
    # Prepare one dataset to determine maximum available training size
    X_train_all, X_test, y_train_all, y_test = make_small_dataset(seed=0, n_samples=N_SAMPLES, source=DATA)

    max_train = X_train_all.shape[0]
    max_train = max(max_train, MIN_TRAIN)

    # Logarithmically spaced training sizes
    train_sizes = np.unique(
        np.round(np.logspace(np.log10(MIN_TRAIN), np.log10(max_train), K_SIZES)).astype(int)
    )
    metric_key, metric_label = metric_metadata(METRIC)

    means = []
    stds = []

    # Run experiments per model with progress estimation
    results_by_model = {}

    # Status-tracking for live printing
    all_status = {name: {int(s): "Not Started" for s in train_sizes} for (_, _, name) in models}
    last_printed_lines = {"n": 0}


    for model_cls, mk, name in models:
        estimator = ProgressEstimator()
        means = []
        stds = []
        for idx, ts in enumerate(train_sizes):
            seeds = [1000 + idx * 100 + r for r in range(N_REPS)]
            # mark in-progress and refresh status
            all_status[name][int(ts)] = "In Progress"
            refresh_all_status(all_status, {name: estimator}, train_sizes, N_REPS, last_printed_lines, min_train=MIN_TRAIN)

            # run batch of repetitions in parallel
            raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(run_single_trial)(s, ts, model_cls, mk, metric=METRIC, n_samples=N_SAMPLES) for s in seeds
            )
            # raw_results is list of (value, elapsed)
            values = [r[0] for r in raw_results]
            elapseds = [r[1] for r in raw_results]
            mean_elapsed = float(np.mean(elapseds)) if len(elapseds) else 0.0
            estimator.add_observation(ts, mean_elapsed)

            # mark complete and refresh status
            all_status[name][int(ts)] = "Complete"
            refresh_all_status(all_status, {name: estimator}, train_sizes, N_REPS, last_printed_lines, min_train=MIN_TRAIN)

            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))

            # estimate remaining wall time for this model
            remaining_sizes = list(train_sizes[idx + 1 :])
            est_remain = estimator.estimate_remaining_wall_time(remaining_sizes, N_REPS, n_jobs)
            print(
                f"[learning_curves] {name}: completed size={ts} reps={N_REPS} mean_job_s={mean_elapsed:.2f}s est_remaining={format_seconds(est_remain)}",
                flush=True,
            )

        results_by_model[name] = {"means": np.array(means), "stds": np.array(stds)}

    # Basic sanity assertions
    for name, res in results_by_model.items():
        assert res["means"].shape[0] == train_sizes.shape[0]
        assert np.all(np.isfinite(res["means"]))

    # Plot and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join("runs", "_learning_curves", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "metric": METRIC,
        "metric_key": metric_key,
        "metric_label": metric_label,
        "n_reps": N_REPS,
        "k_sizes": K_SIZES,
        "n_samples": N_SAMPLES,
        "min_train": MIN_TRAIN,
        "n_jobs": n_jobs,
        "train_sizes": train_sizes.tolist(),
        "models": [
            {
                "name": name,
                "class": model_cls.__name__,
                "kwargs": mk,
            }
            for model_cls, mk, name in models
        ],
    }

    config_txt_path = os.path.join(out_dir, "config.txt")
    with open(config_txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, sort_keys=True))

    plt.figure()
    for name, res in results_by_model.items():
        plt.plot(train_sizes, res["means"], marker="o", label=name)
        plt.fill_between(train_sizes, res["means"] - res["stds"], res["means"] + res["stds"], alpha=0.2)
    plt.xscale("log")
    plt.xlabel("Training set size")
    plt.ylabel(metric_label)
    plt.title(f"Learning curve: {metric_label} vs training set size")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    out_path = os.path.join(out_dir, f"learning_curves_{metric_key}.png")
    plt.savefig(out_path)
    plt.close()

    # Ensure the output file was created
    assert os.path.exists(out_path)

    assert os.path.exists(config_txt_path)

    # Persist all captured test output next to the generated plot.
    captured = capsys.readouterr()
    output_txt_path = os.path.join(out_dir, "output.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(captured.out)
        if captured.err:
            f.write("\n\n[stderr]\n")
            f.write(captured.err)

    assert os.path.exists(output_txt_path)
