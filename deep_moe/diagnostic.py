"""
Diagnostic: plot true conditional density vs ensemble predictive density
at several test points.

Critical for verifying the model actually fits the truth before plumbing
into the full pipeline.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgp import generate, make_true_density, mean_function
from moe_ensemble import MoEEnsemble
from deep_ensemble import DeepEnsemble


def run_diagnostic(
    noise_dist: str,
    *,
    model: str = "deep",
    n_train: int = 1000,
    M: int = 10,
    K: int = 3,
    seed: int = 42,
    output_dir: str = "diagnostics",
    # Deep ensemble hyperparameters
    hidden_dim: int = 32,
    n_hidden: int = 2,
    n_epochs: int = 500,
    entropy_bonus: float = 0.0,
    # MoE hyperparameter
    bootstrap: bool = True,
):
    print("=" * 60)
    print(f"DIAGNOSTIC: {noise_dist} (model={model}, n={n_train}, M={M}, K={K})")
    print("=" * 60)
    
    X_train, y_train = generate(n=n_train, noise_dist=noise_dist, seed=seed)
    
    if model == "deep":
        ens = DeepEnsemble(
            input_dim=X_train.shape[1], M=M, hidden_dim=hidden_dim,
            n_hidden=n_hidden, K=K,
        )
        ens.fit(
            X_train, y_train, n_epochs=n_epochs, base_seed=seed,
            verbose=False, entropy_bonus=entropy_bonus,
        )
    elif model == "moe":
        ens = MoEEnsemble(M=M, n_experts=K, bootstrap=bootstrap)
        ens.fit(X_train, y_train, base_seed=seed)
    else:
        raise ValueError(f"Unknown model: {model!r}")
    
    print(f"Trained {M} ensemble members.")
    
    # Test points across the input range
    x_test_values = np.array([-1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8])
    n_panels = len(x_test_values)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, x_val in enumerate(x_test_values):
        ax = axes[i]
        x = np.array([x_val])
        
        # True density
        true_dist = make_true_density(x, noise_dist)
        y_grid = true_dist.y_grid
        true_density = true_dist.density_values
        
        # Predictive density on the same grid
        pred_dist = ens.predictive_distribution(x)
        pred_density = pred_dist.density(y_grid)
        
        # Plot
        ax.plot(y_grid, true_density, 'k-', label='true', linewidth=2)
        ax.plot(y_grid, pred_density, 'tab:blue', label='MoE predictive',
                linewidth=1.5, alpha=0.85)
        ax.fill_between(y_grid, 0, pred_density, color='tab:blue', alpha=0.15)
        
        # Mark conditional mean and modes
        mu_x = mean_function(x)[0] if x.ndim == 1 else mean_function(x[None, :])[0]
        ax.axvline(mu_x, color='gray', linestyle=':', alpha=0.6, linewidth=1)
        
        ax.set_title(f"x = {x_val:+.1f}", fontsize=11)
        ax.set_xlabel("y")
        ax.set_ylabel("density")
        ax.set_xlim(y_grid[0], y_grid[-1])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    
    # Hide unused subplot
    if n_panels < len(axes):
        for j in range(n_panels, len(axes)):
            axes[j].axis("off")
    
    fig.suptitle(
        f"True vs predicted density ({noise_dist}, model={model}, "
        f"n={n_train}, M={M}, K={K})",
        fontsize=12,
    )
    fig.tight_layout()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out = output_path / f"density_check_{noise_dist}_{model}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)
    
    # Quantitative summary: integrated squared error between true and predictive
    print("\nIntegrated squared error (true vs predicted) per test point:")
    for x_val in x_test_values:
        x = np.array([x_val])
        true_dist = make_true_density(x, noise_dist)
        pred_density = ens.predictive_distribution(x).density(true_dist.y_grid)
        ise = float(
            np.trapz(
                (true_dist.density_values - pred_density) ** 2, true_dist.y_grid
            )
        )
        print(f"  x={x_val:+.1f}: ISE = {ise:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-dist", choices=["gaussian", "bimodal", "skewed", "all"],
                        default="all")
    parser.add_argument("--model", choices=["deep", "moe"], default="deep")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--M", type=int, default=10)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="diagnostics")
    args = parser.parse_args()
    
    if args.noise_dist == "all":
        nd_list = ["gaussian", "bimodal", "skewed"]
    else:
        nd_list = [args.noise_dist]
    
    for nd in nd_list:
        run_diagnostic(
            noise_dist=nd,
            model=args.model,
            n_train=args.n_train,
            M=args.M,
            K=args.K,
            seed=args.seed,
            output_dir=args.output_dir,
            hidden_dim=args.hidden_dim,
            n_hidden=args.n_hidden,
            n_epochs=args.n_epochs,
            entropy_bonus=args.entropy_bonus,
            bootstrap=not args.no_bootstrap,
        )


if __name__ == "__main__":
    main()
