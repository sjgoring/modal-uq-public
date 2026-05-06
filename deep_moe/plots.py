"""
Plot selective loss curves and AURC bars from v2 experiment results.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# UMs to plot, with display labels and styles.
UM_PLOT_SPEC = [
    ("var_tu",      "Variance TU",        "-",  "tab:blue"),
    ("ent_tu",      "Entropy TU",         "-",  "tab:orange"),
    ("quest_tu_01", r"QUEST TU ($\alpha=0.1$)", "-",  "tab:green"),
    ("quest_tu_g",  "QUEST TU (global)",  ":",  "tab:green"),
    ("random",      "Random",             "-",  "gray"),
]


def plot_loss_curves(
    results_dir: str,
    output_path: str,
    estimator: str,
):
    """Three panels: gaussian, bimodal, skewed."""
    noise_dists = ["gaussian", "bimodal", "skewed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    
    for ax, nd in zip(axes, noise_dists):
        path = Path(results_dir) / f"results_{nd}_{estimator}.npz"
        if not path.exists():
            ax.set_title(f"{nd} (missing)")
            continue
        
        data = np.load(path)
        coverages = data["coverages"]
        n_seeds = int(data["n_seeds"])
        full_loss = float(data["test_loss_mean"])
        full_loss_se = float(data["test_loss_se"])
        
        for um_key, label, ls, color in UM_PLOT_SPEC:
            mean_key = f"loss_mean_{um_key}"
            se_key = f"loss_se_{um_key}"
            if mean_key not in data.files:
                continue
            mean = data[mean_key]
            se = data[se_key]
            
            ax.plot(coverages, mean, ls, color=color, label=label, linewidth=1.5)
            ax.fill_between(coverages, mean - se, mean + se,
                            color=color, alpha=0.2)
        
        ax.set_xlabel("Coverage (fraction retained)")
        ax.set_ylabel(r"Selective log-density loss")
        ax.set_title(f"{nd}  "
                     f"(full loss = {full_loss:.3f} $\\pm$ {full_loss_se:.3f}, "
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
    results_dir: str,
    output_path: str,
    estimator: str,
):
    """Three-panel AURC bar charts (gaussian, bimodal, skewed)."""
    noise_dists = ["gaussian", "bimodal", "skewed"]
    bar_spec = [
        ("var_tu",      "Var TU",            "tab:blue"),
        ("ent_tu",      "Ent TU",            "tab:orange"),
        ("quest_tu_01", "QUEST TU\n(α=0.1)",  "tab:green"),
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
        ax.set_ylabel(r"AURC = $\int$ loss $dc$")
        ax.grid(True, axis="y", alpha=0.3)
    
    fig.suptitle(f"Estimator: {estimator}", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def print_aurc_table(results_dir: str, estimator: str):
    print("\n" + "=" * 90)
    print(f"AURC table — estimator: {estimator}")
    print("=" * 90)
    
    noise_dists = ["gaussian", "bimodal", "skewed"]
    bar_spec = [
        ("random",      "Random"),
        ("var_tu",      "Variance TU"),
        ("ent_tu",      "Entropy TU"),
        ("quest_tu_01", r"QUEST TU (a=0.1)"),
        ("quest_tu_g",  "QUEST TU (global)"),
    ]
    
    print(f"\n{'UM':<22} " + " | ".join(f"{nd:^18}" for nd in noise_dists))
    print("-" * (22 + 21 * len(noise_dists)))
    for um_key, label in bar_spec:
        row = [label.ljust(22)]
        for nd in noise_dists:
            path = Path(results_dir) / f"results_{nd}_{estimator}.npz"
            if not path.exists():
                row.append(f"{'(missing)':^18}")
                continue
            data = np.load(path)
            mean_key = f"aurc_mean_{um_key}"
            se_key = f"aurc_se_{um_key}"
            if mean_key not in data.files:
                row.append(f"{'-':^18}")
                continue
            m = float(data[mean_key])
            se = float(data[se_key])
            row.append(f"{m:>8.4f} ± {se:>5.4f} ")
        print(" | ".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--estimator", default="all", choices=["oracle", "c2", "all"])
    args = parser.parse_args()
    
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    estimators = ["oracle", "c2"] if args.estimator == "all" else [args.estimator]
    
    for est in estimators:
        print(f"\n=== Estimator: {est} ===")
        plot_loss_curves(
            args.results_dir,
            str(output_dir / f"loss_coverage_{est}.pdf"),
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
