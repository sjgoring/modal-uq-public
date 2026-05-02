"""
Diagnostic checks for the variance-vs-error relationship.

Investigates why variance-based UMs may underperform random in the Gaussian setting.
Hypotheses tested:
  H1: Variance has no correlation with squared error (calibration failure).
  H2: Predicted sigma^2 doesn't match actual squared residuals (ensemble miscal).
  H3: EU is noisy with small M, hurting TU ranking; AU alone might do better.
  H4: AU and EU components have different signal quality.

Run as a standalone script. Trains a single ensemble, computes diagnostics, prints
results, and saves a small set of diagnostic plots.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr

from ensemble import DeepEnsemble
from friedman import friedman_mean, generate_friedman
from measures import (
    entropy_au, entropy_eu, entropy_tu,
    variance_au, variance_eu, variance_tu,
)


def run_diagnostics(
    noise_dist: str = "gaussian",
    noise_scale: float = 1.0,
    n_train: int = 1000,
    n_test: int = 500,
    M: int = 5,
    n_epochs: int = 500,
    hidden_dim: int = 50,
    n_hidden: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    output_dir: str = "diagnostics",
):
    print("=" * 60)
    print(f"DIAGNOSTIC: {noise_dist} noise, n_train={n_train}, M={M}")
    print("=" * 60)
    
    # Generate data and train ensemble
    X_train, y_train = generate_friedman(
        n=n_train, noise_dist=noise_dist,
        noise_scale=noise_scale, seed=seed,
    )
    X_test, y_test = generate_friedman(
        n=n_test, noise_dist=noise_dist,
        noise_scale=noise_scale, seed=seed + 1,
    )
    
    print(f"\nTraining ensemble (M={M}, epochs={n_epochs})...")
    t0 = time.time()
    ensemble = DeepEnsemble(
        input_dim=X_train.shape[1],
        M=M, hidden_dim=hidden_dim, n_hidden=n_hidden,
    )
    ensemble.fit(
        X_train, y_train,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        base_seed=seed * M, verbose=False,
    )
    print(f"  Training took {time.time() - t0:.1f}s")
    
    # Predict on test set
    mus, sigmas = ensemble.predict(X_test)  # shapes (n_test, M)
    pred_means = mus.mean(axis=1)
    sq_errors = (pred_means - y_test) ** 2
    
    # True conditional means (signal) and noise level
    true_means = friedman_mean(X_test)
    
    print(f"\nTest set summary:")
    print(f"  True signal range: [{true_means.min():.2f}, {true_means.max():.2f}]")
    print(f"  Pred mean range:   [{pred_means.min():.2f}, {pred_means.max():.2f}]")
    print(f"  MSE = {sq_errors.mean():.3f}")
    print(f"  R^2 = {1 - sq_errors.mean() / y_test.var():.4f}")
    print(f"  Mean predicted sigma: {sigmas.mean():.3f} (per ensemble member, "
          f"averaged across test set)")
    
    # Compute per-test-point UMs
    print("\nComputing UMs per test point...")
    n_test_pts = len(y_test)
    var_au_vals = np.zeros(n_test_pts)
    var_eu_vals = np.zeros(n_test_pts)
    var_tu_vals = np.zeros(n_test_pts)
    ent_au_vals = np.zeros(n_test_pts)
    ent_tu_vals = np.zeros(n_test_pts)
    for i in range(n_test_pts):
        pred = ensemble.predictive_distribution(X_test[i])
        var_au_vals[i] = variance_au(pred)
        var_eu_vals[i] = variance_eu(pred)
        var_tu_vals[i] = variance_tu(pred)
        ent_au_vals[i] = entropy_au(pred)
        ent_tu_vals[i] = entropy_tu(pred)
    
    # ============== H1: Correlation of UMs with squared error ==============
    print("\n--- H1: Do UMs correlate with squared error? ---")
    print("  (Spearman = rank correlation; relevant for selective prediction)")
    print(f"  {'UM':<25} {'Spearman':<12} {'Pearson':<12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    for name, vals in [
        ("Variance AU", var_au_vals),
        ("Variance EU", var_eu_vals),
        ("Variance TU", var_tu_vals),
        ("Entropy AU", ent_au_vals),
        ("Entropy TU", ent_tu_vals),
    ]:
        sp, _ = spearmanr(vals, sq_errors)
        pe, _ = pearsonr(vals, sq_errors)
        print(f"  {name:<25} {sp:<+12.4f} {pe:<+12.4f}")
    print("  (Correlation near 0 means the UM is essentially random for ranking.)")
    print("  (Negative correlation means the UM is anti-predictive — worse than random.)")
    
    # ============== H2: Calibration of ensemble sigma^2 ==============
    print("\n--- H2: Is the ensemble well-calibrated? ---")
    # Predicted variance is mean of per-component sigma^2 plus var of means
    pred_total_var = var_tu_vals
    pred_au = var_au_vals  # mean within-component variance
    
    # Actual squared residuals are sq_errors. For perfect calibration:
    #   E[sq_error | x] = total predictive variance at x.
    # We bin test points by predicted variance and check the within-bin mean
    # of squared errors.
    n_bins = 10
    sort_idx = np.argsort(pred_total_var)
    bin_edges = np.linspace(0, n_test_pts, n_bins + 1).astype(int)
    print(f"  {'Bin':<5} {'Mean pred var':<16} {'Mean sq error':<16} {'Ratio':<8}")
    print(f"  {'-'*5} {'-'*16} {'-'*16} {'-'*8}")
    for b in range(n_bins):
        idx = sort_idx[bin_edges[b]:bin_edges[b+1]]
        mpv = pred_total_var[idx].mean()
        mse = sq_errors[idx].mean()
        ratio = mse / mpv if mpv > 1e-9 else float('nan')
        print(f"  {b:<5} {mpv:<16.4f} {mse:<16.4f} {ratio:<8.3f}")
    print("  (Well-calibrated: ratio ~1.0 across bins. Underconfident: ratio < 1. "
          "Overconfident: ratio > 1.)")
    
    # ============== H3: AU-only ranking vs TU ranking ==============
    print("\n--- H3: Does noisy EU hurt TU? ---")
    # If AU alone has higher Spearman than TU, then EU is adding noise.
    sp_au, _ = spearmanr(var_au_vals, sq_errors)
    sp_tu, _ = spearmanr(var_tu_vals, sq_errors)
    sp_eu, _ = spearmanr(var_eu_vals, sq_errors)
    print(f"  Variance AU rank corr: {sp_au:+.4f}")
    print(f"  Variance EU rank corr: {sp_eu:+.4f}")
    print(f"  Variance TU rank corr: {sp_tu:+.4f}")
    if sp_au > sp_tu:
        print("  -> AU-only ranks better than TU. EU is degrading the signal.")
        print("     Suggests increasing M, more training, or AU as primary UM.")
    else:
        print("  -> TU ranks at least as well as AU alone. EU is contributing useful info.")
    
    # ============== H4: Variance heterogeneity across test set ==============
    print("\n--- H4: How much do predicted variances vary across the test set? ---")
    print(f"  Predicted variance: min={pred_total_var.min():.4f}, "
          f"max={pred_total_var.max():.4f}, "
          f"std={pred_total_var.std():.4f}, "
          f"mean={pred_total_var.mean():.4f}")
    # Coefficient of variation
    cv = pred_total_var.std() / pred_total_var.mean() if pred_total_var.mean() > 0 else float('nan')
    print(f"  Coefficient of variation: {cv:.3f}")
    print("  (Low CV (<0.2) means variance is roughly constant across test set,")
    print("   so it provides little signal for ranking.)")
    
    # ============== Save diagnostic plots ==============
    print("\nSaving diagnostic plots...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # Plot 1: Predicted variance vs squared error (calibration scatter)
    ax = axes[0, 0]
    ax.scatter(pred_total_var, sq_errors, s=8, alpha=0.5)
    lim = max(pred_total_var.max(), sq_errors.max())
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y = x (perfect cal.)")
    ax.set_xlabel("Predicted total variance")
    ax.set_ylabel("Squared error")
    ax.set_title(f"Calibration scatter ({noise_dist})")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Predicted AU only vs squared error
    ax = axes[0, 1]
    ax.scatter(var_au_vals, sq_errors, s=8, alpha=0.5, color="tab:orange")
    ax.set_xlabel("Predicted AU (mean within-component variance)")
    ax.set_ylabel("Squared error")
    ax.set_title("AU vs error")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: EU vs squared error
    ax = axes[1, 0]
    ax.scatter(var_eu_vals, sq_errors, s=8, alpha=0.5, color="tab:green")
    ax.set_xlabel("Predicted EU (variance of ensemble means)")
    ax.set_ylabel("Squared error")
    ax.set_title("EU vs error")
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of predicted variances (heterogeneity)
    ax = axes[1, 1]
    ax.hist(pred_total_var, bins=30, color="tab:blue", alpha=0.7)
    ax.set_xlabel("Predicted total variance")
    ax.set_ylabel("Count")
    ax.set_title(f"Variance distribution (CV = {cv:.3f})")
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Diagnostics: {noise_dist} noise, M={M}, "
                 f"n_train={n_train}, n_epochs={n_epochs}",
                 fontsize=11)
    fig.tight_layout()
    out = output_path / f"diagnostic_{noise_dist}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)
    
    return {
        "spearman_var_au": sp_au,
        "spearman_var_eu": sp_eu,
        "spearman_var_tu": sp_tu,
        "cv_pred_var": cv,
        "mse": float(sq_errors.mean()),
        "r2": float(1 - sq_errors.mean() / y_test.var()),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--noise", default="gaussian", choices=["gaussian", "t5", "t3"])
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--M", type=int, default=5)
    p.add_argument("--n-epochs", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="diagnostics")
    args = p.parse_args()
    
    run_diagnostics(
        noise_dist=args.noise,
        n_train=args.n_train,
        n_test=args.n_test,
        M=args.M,
        n_epochs=args.n_epochs,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
