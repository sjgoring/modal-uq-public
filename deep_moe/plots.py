"""
Plot selective loss curves and AURC bars from selective-prediction experiments.

For each estimator (oracle, MLE), produces two 3-panel figures:
- TU plot: variance TU, entropy TU, QUEST TU (alpha=0.1, global) on each panel.
- EU plot: variance EU, entropy EU, QUEST EU (alpha=0.1, global) on each panel.

Panels are ordered: Gaussian, skewed, bimodal.

Expected input files in results_dir:
    results_<noise>_<estimator>.npz

Expected NPZ keys:
    coverages, n_seeds, test_loss_mean, test_loss_se,
    loss_mean_<um>, loss_se_<um>, aurc_mean_<um>, aurc_se_<um>

Missing files or keys are skipped gracefully and marked in the output plots.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NOISE_ORDER = ["gaussian", "skewed", "bimodal"]


# UM specs for TU and EU plots
TU_SPEC = [
    ("var_tu",      "Variance TU",                 "-",  "tab:blue"),
    ("ent_tu",      "Entropy TU",                  "-",  "tab:orange"),
    ("quest_tu_01", r"QUEST TU ($\alpha=0.1$)",    "-",  "tab:green"),
    ("quest_tu_g",  "QUEST TU (global)",           ":",  "tab:green"),
    ("random",      "Random",                      "-",  "gray"),
]

EU_SPEC = [
    ("var_eu",      "Variance EU",                 "-",  "tab:blue"),
    ("ent_eu",      "Entropy EU",                  "-",  "tab:orange"),
    ("quest_eu_01", r"QUEST EU ($\alpha=0.1$)",    "-",  "tab:green"),
    ("quest_eu_g",  "QUEST EU (global)",           ":",  "tab:green"),
    ("random",      "Random",                      "-",  "gray"),
]


def plot_loss_curves(
    results_dir: str,
    output_path: str,
    estimator: str,
    spec,
    spec_label: str,  # "TU" or "EU"
):
    """Plot selective loss curves for one estimator over all noise settings.

    Args:
        results_dir: Directory containing per-noise NPZ results files.
        output_path: File path for the output PDF/figure.
        estimator: One of {"oracle", "mle"}.
        spec: Sequence of (um_key, label, linestyle, color).
        spec_label: Display label such as "TU" or "EU".
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    
    for ax, nd in zip(axes, NOISE_ORDER):
        path = Path(results_dir) / f"results_{nd}_{estimator}.npz"
        if not path.exists():
            ax.set_title(f"{nd} (missing)")
            continue
        
        data = np.load(path)
        coverages = data["coverages"]
        n_seeds = int(data["n_seeds"])
        full_loss = float(data["test_loss_mean"])
        full_loss_se = float(data["test_loss_se"])
        
        for um_key, label, ls, color in spec:
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
    fig.suptitle(f"{spec_label} ranking — estimator: {estimator}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def plot_aurc_bars(
    results_dir: str,
    output_path: str,
    estimator: str,
    spec,
    spec_label: str,
):
    """Plot three-panel AURC bar charts in NOISE_ORDER."""
    bar_spec = [(k, label.replace(" TU", "").replace(" EU", "").strip(), color)
                for (k, label, _ls, color) in spec]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, nd in zip(axes, NOISE_ORDER):
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
        ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
        ax.set_title(f"{nd} ($n_{{seeds}}$={n_seeds})")
        ax.set_ylabel(r"AURC = $\int$ loss $dc$")
        ax.grid(True, axis="y", alpha=0.3)
    
    fig.suptitle(f"{spec_label} AURC — estimator: {estimator}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close(fig)


def print_aurc_table(results_dir: str, estimator: str):
    print("\n" + "=" * 90)
    print(f"AURC table — estimator: {estimator}")
    print("=" * 90)
    
    rows = [
        ("random",      "Random"),
        ("var_tu",      "Variance TU"),
        ("var_eu",      "Variance EU"),
        ("ent_tu",      "Entropy TU"),
        ("ent_eu",      "Entropy EU"),
        ("quest_tu_01", "QUEST TU (a=0.1)"),
        ("quest_eu_01", "QUEST EU (a=0.1)"),
        ("quest_tu_g",  "QUEST TU (global)"),
        ("quest_eu_g",  "QUEST EU (global)"),
    ]
    
    print(f"\n{'UM':<22} " + " | ".join(f"{nd:^18}" for nd in NOISE_ORDER))
    print("-" * (22 + 21 * len(NOISE_ORDER)))
    for um_key, label in rows:
        row = [label.ljust(22)]
        for nd in NOISE_ORDER:
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
    parser.add_argument("--estimator", default="all", choices=["oracle", "mle", "all"])
    args = parser.parse_args()
    
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    estimators = ["oracle", "mle"] if args.estimator == "all" else [args.estimator]
    
    for est in estimators:
        print(f"\n=== Estimator: {est} ===")
        # TU plots
        plot_loss_curves(
            args.results_dir,
            str(output_dir / f"loss_coverage_TU_{est}.pdf"),
            estimator=est, spec=TU_SPEC, spec_label="TU",
        )
        plot_aurc_bars(
            args.results_dir,
            str(output_dir / f"aurc_bars_TU_{est}.pdf"),
            estimator=est, spec=TU_SPEC, spec_label="TU",
        )
        # EU plots
        plot_loss_curves(
            args.results_dir,
            str(output_dir / f"loss_coverage_EU_{est}.pdf"),
            estimator=est, spec=EU_SPEC, spec_label="EU",
        )
        plot_aurc_bars(
            args.results_dir,
            str(output_dir / f"aurc_bars_EU_{est}.pdf"),
            estimator=est, spec=EU_SPEC, spec_label="EU",
        )
        print_aurc_table(args.results_dir, estimator=est)


if __name__ == "__main__":
    main()
