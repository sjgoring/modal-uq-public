"""
Plotting selective prediction results with mean +/- SE shading.

Produces:
- Selective MSE-coverage curves (one panel per noise setting).
- AURC bar charts with error bars.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# UMs to plot, with display labels and styles.
# (key, label, linestyle, color)
UM_PLOT_SPEC = [
    ("var_tu",      "Variance TU",        "-",  "tab:blue"),
    ("ent_tu",      "Entropy TU",         "-",  "tab:orange"),
    ("quest_tu_01", r"QUEST TU ($\alpha=0.1$)", "-",  "tab:green"),
    ("quest_tu_05", r"QUEST TU ($\alpha=0.5$)", "--", "tab:green"),
    ("quest_tu_g",  "QUEST TU (global)",  ":",  "tab:green"),
    ("random",      "Random",             "-",  "gray"),
]


def plot_mse_curves(
    results_dir: str = "results",
    output_path: str = "results/mse_coverage.pdf",
    estimator: str = "oracle",
):
    """Plot selective MSE vs coverage with SE shading.
    
    One panel per noise distribution.
    """
    noise_dists = ["gaussian", "t5", "t3"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    
    for ax, nd in zip(axes, noise_dists):
        path = Path(results_dir) / f"results_{nd}_{estimator}.npz"
        if not path.exists():
            ax.set_title(f"{nd} (missing)")
            continue
        
        data = np.load(path)
        coverages = data["coverages"]
        n_seeds = int(data["n_seeds"])
        full_mse = float(data["test_mse_mean"])
        full_mse_se = float(data["test_mse_se"])
        
        for um_key, label, ls, color in UM_PLOT_SPEC:
            mean_key = f"mse_mean_{um_key}"
            se_key = f"mse_se_{um_key}"
            if mean_key not in data.files:
                continue
            mean = data[mean_key]
            se = data[se_key]
            
            ax.plot(coverages, mean, ls, color=color, label=label, linewidth=1.5)
            ax.fill_between(coverages, mean - se, mean + se,
                            color=color, alpha=0.2)
        
        ax.set_xlabel("Coverage (fraction retained)")
        ax.set_ylabel("Selective MSE")
        ax.set_title(f"{nd}  "
                     f"(full MSE = {full_mse:.3f} $\\pm$ {full_mse_se:.3f}, "
                     f"$n_{{seeds}}$={n_seeds})")
        ax.set_xlim(0.05, 1.0)
        ax.grid(True, alpha=0.3)
    
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Estimator: {estimator}", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_aurc_bars(
    results_dir: str = "results",
    output_path: str = "results/aurc_bars.pdf",
    estimator: str = "oracle",
):
    """Bar chart of AURC across noise settings, grouped by UM, with error bars."""
    noise_dists = ["gaussian", "t5", "t3"]
    
    # Order matters for visual layout
    bar_spec = [
        ("var_tu",      "Var TU",            "tab:blue"),
        ("ent_tu",      "Ent TU",            "tab:orange"),
        ("quest_tu_01", "QUEST TU\n(α=0.1)",  "tab:green"),
        ("quest_tu_05", "QUEST TU\n(α=0.5)",  "tab:green"),
        ("quest_tu_g",  "QUEST TU\n(global)", "tab:green"),
        ("random",      "Random",            "gray"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, nd in zip(axes, noise_dists):
        path = Path(results_dir) / f"results_{nd}_{estimator}.npz"
        if not path.exists():
            ax.set_title(f"{nd} (missing)")
            continue
        data = np.load(path)
        n_seeds = int(data["n_seeds"])
        
        names, means, ses, colors = [], [], [], []
        for um_key, label, color in bar_spec:
            mean_key = f"aurc_mean_{um_key}"
            se_key = f"aurc_se_{um_key}"
            if mean_key not in data.files:
                continue
            names.append(label)
            means.append(float(data[mean_key]))
            ses.append(float(data[se_key]))
            colors.append(color)
        
        x_pos = np.arange(len(names))
        ax.bar(x_pos, means, yerr=ses, capsize=4,
               color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(f"{nd} ($n_{{seeds}}$={n_seeds})")
        ax.set_ylabel("AURC = $\\int$ MSE $dc$")
        ax.grid(True, axis="y", alpha=0.3)
    
    fig.suptitle(f"Estimator: {estimator}", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def print_aurc_table(results_dir: str = "results", estimator: str = "oracle"):
    """Print a LaTeX-friendly AURC summary table to stdout."""
    noise_dists = ["gaussian", "t5", "t3"]
    bar_spec = [
        ("random",      "Random"),
        ("var_tu",      "Variance TU"),
        ("ent_tu",      "Entropy TU"),
        ("quest_tu_01", r"QUEST TU ($\alpha=0.1$)"),
        ("quest_tu_05", r"QUEST TU ($\alpha=0.5$)"),
        ("quest_tu_g",  "QUEST TU (global)"),
    ]
    
    print()
    print(f"{'UM':<28}", end="")
    for nd in noise_dists:
        print(f"{nd:<22}", end="")
    print()
    print("-" * (28 + 22 * len(noise_dists)))
    
    for um_key, label in bar_spec:
        print(f"{label:<28}", end="")
        for nd in noise_dists:
            path = Path(results_dir) / f"results_{nd}_{estimator}.npz"
            if not path.exists():
                print(f"{'(no data)':<22}", end="")
                continue
            data = np.load(path)
            mean = float(data[f"aurc_mean_{um_key}"])
            se = float(data[f"aurc_se_{um_key}"])
            cell = f"{mean:.4f} ± {se:.4f}"
            print(f"{cell:<22}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--estimator", type=str, default="oracle",
                        choices=["oracle", "c2", "c3", "all"])
    args = parser.parse_args()
    
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    estimators = ["oracle", "c2", "c3"] if args.estimator == "all" else [args.estimator]
    
    for est in estimators:
        print(f"\n=== Estimator: {est} ===")
        plot_mse_curves(
            args.results_dir,
            str(output_dir / f"mse_coverage_{est}.pdf"),
            estimator=est,
        )
        plot_aurc_bars(
            args.results_dir,
            str(output_dir / f"aurc_bars_{est}.pdf"),
            estimator=est,
        )
        print_aurc_table(args.results_dir, estimator=est)


if __name__ == "__main__":
    main()
