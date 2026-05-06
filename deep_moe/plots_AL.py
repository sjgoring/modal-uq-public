"""Plot active-learning curves and final-round summaries for QUEST experiments.

This script reads the outputs produced by deep_moe/active_learning.py and plots
per-round learning curves plus final-round summary bars for the AL metrics.

It supports both the synthetic DGP runs and the MPE runs, which may produce
an additional results_mpe_active_learning.npz file and can have calibration
values that are entirely NaN.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFERRED_RESULT_ORDER = ["mpe", "gaussian", "bimodal", "skewed"]
METRIC_SPECS = {
    "mode_abs_error": {
        "label": "Mode absolute error",
        "ylabel": "Mean absolute error",
        "title": "Mode absolute error",
    },
    "calibration": {
        "label": "Calibration",
        "ylabel": "Calibration",
        "title": "Calibration",
    },
}


def _available_result_datasets(results_dir: str) -> list[str]:
    base = Path(results_dir)
    found = {}
    for path in base.glob("results_*_active_learning.npz"):
        name = path.name
        prefix = "results_"
        suffix = "_active_learning.npz"
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        dataset = name[len(prefix):-len(suffix)]
        found[dataset] = path

    ordered = [nd for nd in PREFERRED_RESULT_ORDER if nd in found]
    ordered.extend(sorted(nd for nd in found if nd not in PREFERRED_RESULT_ORDER))
    return ordered


def _load_metric_keys(data: np.lib.npyio.NpzFile, metric_name: str) -> list[str]:
    prefix = f"{metric_name}_mean_"
    return sorted(
        key[len(prefix):]
        for key in data.files
        if key.startswith(prefix)
    )


def _is_all_nan(array: np.ndarray) -> bool:
    return not np.isfinite(np.asarray(array, dtype=float)).any()


def plot_metric_curves(
    results_dir: str,
    output_path: str,
    metric_name: str,
):
    """One panel per available result file, in a preferred order."""
    spec = METRIC_SPECS[metric_name]
    datasets = _available_result_datasets(results_dir)
    if not datasets:
        print(f"No results files found in {results_dir}")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.2), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, nd in zip(axes, datasets):
        path = Path(results_dir) / f"results_{nd}_active_learning.npz"
        if not path.exists():
            ax.set_title(f"{nd} (missing)")
            continue

        data = np.load(path, allow_pickle=False)
        n_seeds = int(data["n_seeds"])
        d_init = int(data["d_init"])
        n_rounds = int(data["n_rounds"])
        measure_names = _load_metric_keys(data, metric_name)

        plotted_any = False

        for measure_name in measure_names:
            mean_key = f"{metric_name}_mean_{measure_name}"
            se_key = f"{metric_name}_se_{measure_name}"
            if mean_key not in data.files or se_key not in data.files:
                continue

            mean = np.asarray(data[mean_key], dtype=float)
            se = np.asarray(data[se_key], dtype=float)
            if not np.isfinite(mean).any():
                continue
            x = np.arange(mean.shape[0])
            ax.plot(x, mean, label=measure_name, linewidth=1.5)
            ax.fill_between(x, mean - se, mean + se, alpha=0.18)
            plotted_any = True

        if not plotted_any:
            ax.set_title(f"{nd} (no finite {metric_name} values)")
            ax.axis("off")
            continue

        ax.set_xlabel("Round")
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(f"{nd}  (D_init={d_init}, rounds={n_rounds}, seeds={n_seeds})")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(spec["title"], fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_final_bars(
    results_dir: str,
    output_path: str,
    metric_name: str,
):
    """Final-round bar charts for each available result file."""
    spec = METRIC_SPECS[metric_name]
    datasets = _available_result_datasets(results_dir)
    if not datasets:
        print(f"No results files found in {results_dir}")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4))
    axes = np.atleast_1d(axes)

    for ax, nd in zip(axes, datasets):
        path = Path(results_dir) / f"results_{nd}_active_learning.npz"
        if not path.exists():
            ax.set_title(f"{nd} (missing)")
            continue

        data = np.load(path, allow_pickle=False)
        n_seeds = int(data["n_seeds"])
        measure_names = _load_metric_keys(data, metric_name)

        names, means, ses = [], [], []
        for measure_name in measure_names:
            mean_key = f"final_{metric_name}_mean_{measure_name}"
            se_key = f"final_{metric_name}_se_{measure_name}"
            if mean_key not in data.files or se_key not in data.files:
                continue
            mean = float(data[mean_key])
            se = float(data[se_key])
            if not np.isfinite(mean):
                continue
            names.append(measure_name)
            means.append(mean)
            ses.append(se)

        if not names:
            ax.set_title(f"{nd} (no finite values)")
            ax.axis("off")
            continue

        x_pos = np.arange(len(names))
        ax.bar(x_pos, means, yerr=ses, capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        ax.set_title(f"{nd} (n_seeds={n_seeds})")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Final-round {spec['title']}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def print_final_table(results_dir: str, metric_name: str):
    spec = METRIC_SPECS[metric_name]
    print("\n" + "=" * 90)
    print(f"Final-round table — {spec['title']}")
    print("=" * 90)

    for nd in _available_result_datasets(results_dir):
        path = Path(results_dir) / f"results_{nd}_active_learning.npz"
        if not path.exists():
            print(f"{nd:<12} missing")
            continue

        data = np.load(path, allow_pickle=False)
        measure_names = _load_metric_keys(data, metric_name)
        print(f"\n{nd}")
        for measure_name in measure_names:
            mean_key = f"final_{metric_name}_mean_{measure_name}"
            se_key = f"final_{metric_name}_se_{measure_name}"
            if mean_key not in data.files or se_key not in data.files:
                continue
            mean = float(data[mean_key])
            se = float(data[se_key])
            if not np.isfinite(mean):
                continue
            print(f"  {measure_name:<24} {mean:>10.4f} ± {se:>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--metric", default="all", choices=["mode_abs_error", "calibration", "all"])
    args = parser.parse_args()

    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_list = list(METRIC_SPECS) if args.metric == "all" else [args.metric]

    for metric_name in metric_list:
        plot_metric_curves(
            args.results_dir,
            str(output_dir / f"{metric_name}_curves.pdf"),
            metric_name=metric_name,
        )
        plot_final_bars(
            args.results_dir,
            str(output_dir / f"{metric_name}_final_bars.pdf"),
            metric_name=metric_name,
        )
        print_final_table(args.results_dir, metric_name=metric_name)


if __name__ == "__main__":
    main()
