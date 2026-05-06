# tests/test_oracle_diagnostic.py
import numpy as np
import sys
sys.path.insert(0, '.')

from modal_uq.datasets.synthetic_constant_var import SyntheticConstantVarDataset
from modal_uq.models.oracle import Oracle
from modal_uq.models.base import InferentialChoiceConfig
from modal_uq.analysis.correlation import compute_uncertainty_scores
from scipy.stats import spearmanr

import modal_uq.registry
# add near the top of tests/test_orcale_diagnostic.py
import modal_uq.uncertainty.random
import modal_uq.uncertainty.variance
import modal_uq.uncertainty.differential_entropy
import modal_uq.uncertainty.quest
import modal_uq.uncertainty.scoring_rules

def test_oracle_diagnostic():
    """Diagnose Oracle model issues with uncertainty measures."""
    
    print("=" * 80)
    print("ORACLE MODEL DIAGNOSTIC TEST")
    print("=" * 80)
    
    # 1. Create dataset
    print("\n[STEP 1] Creating dataset...")
    ds = SyntheticConstantVarDataset(n_samples=1000, seed=42)
    print(f"  Dataset y_grid shape: {ds.y_grid.shape}")
    print(f"  Dataset y_grid range: [{ds.y_grid.min():.4f}, {ds.y_grid.max():.4f}]")
    
    # 2. Create Oracle model with BMA inferential choice
    print("\n[STEP 2] Creating Oracle model with BMA...")
    model = Oracle(
        inferential_choice={
            'predict': 'bma',
            'approximate': 'posterior_predictive',
            'point_estimate_criterion': 'mle'
        },
        data_set=ds
    )
    print(f"  Inferential choice config: {model.get_inferential_choice_config()}")
    
    # 3. Fit model (should be no-op for Oracle)
    print("\n[STEP 3] Fitting model...")
    model.fit(ds.X_train, ds.y_train)
    print(f"  Model._y_min: {model._y_min}, Model._y_max: {model._y_max}")
    
    # 4. Get ground truth densities
    print("\n[STEP 4] Computing ground truth densities...")
    true_dens = ds.gt_dens(ds.X_test, ds.y_test)
    print(f"  true_dens shape: {true_dens.shape}")
    print(f"  true_dens sample values [0]: {true_dens[0, :5]}")  # First 5 grid points for first sample
    
    # 5. Get estimated densities from Oracle
    print("\n[STEP 5] Computing estimated densities from Oracle...")
    est_dens = model.predict_density(ds.X_test, ds.y_grid, context='predict')
    print(f"  est_dens shape: {est_dens.shape}")
    print(f"  est_dens sample values [0,0]: {est_dens[0, 0, :5]}")  # First member, first sample
    
    # 6. Compare densities
    print("\n[STEP 6] Comparing densities...")
    
    # For BMA, est_dens is (5, N, G) - check if all members are identical
    if est_dens.ndim == 3:
        print(f"  est_dens has 3 dimensions (BMA case)")
        # Check if all 5 members are identical
        members_identical = np.allclose(est_dens[0], est_dens[1:])
        print(f"  Are all 5 members identical? {members_identical}")
        
        # Average to compare with true_dens
        est_dens_mean = est_dens.mean(axis=0)
    else:
        est_dens_mean = est_dens
    
    print(f"  est_dens_mean shape: {est_dens_mean.shape}")
    print(f"  true_dens shape: {true_dens.shape}")
    
    # Check if they're numerically identical
    if est_dens_mean.shape == true_dens.shape:
        max_diff = np.max(np.abs(est_dens_mean - true_dens))
        print(f"  Max difference between est_dens_mean and true_dens: {max_diff:.2e}")
        are_identical = np.allclose(est_dens_mean, true_dens)
        print(f"  Are they numerically identical (allclose)? {are_identical}")
    
    # 7. Compute uncertainty measures
    print("\n[STEP 7] Computing uncertainty measures...")
    measures = [
        # {"name": "random", "params": {"label": "random"}},
        {"name": "variance", "params": {"decomposition": "total", "label": "variance_total"}},
        {"name": "variance", "params": {"decomposition": "aleatoric", "label": "variance_aleatoric"}},
        {"name": "alpha_volume", "params": {"decomposition": "total", "label": "alpha_volume_total_5%", "alpha": 0.05}},
        {"name": "alpha_volume", "params": {"decomposition": "aleatoric", "label": "alpha_volume_aleatoric_5%", "alpha": 0.05}},
        {"name": "differential_entropy", "params": {"decomposition": "total", "label": "diff_entropy_total"}},
        {"name": "differential_entropy", "params": {"decomposition": "aleatoric", "label": "diff_entropy_aleatoric"}}
    ]
    
    df_scores = compute_uncertainty_scores(measures, model, ds.X_test, ds.y_test, y_grid=ds.y_grid)
    print(f"\n  Uncertainty scores shape: {df_scores.shape}")
    print(f"  Uncertainty scores:\n{df_scores.head(10)}")
    
    # 8. Check if all measures have identical rankings
    print("\n[STEP 8] Analyzing rankings...")
    for col in df_scores.columns:
        ranking = np.argsort(-np.array(df_scores[col]))  # Descending order
        print(f"  Top 5 indices by {col}: {ranking[:5]}")

    # Check if rankings are identical across measures
    # make sure to exclude random for this exercise.
    rankings = {col: np.argsort(-np.array(df_scores[col])) for col in df_scores.columns}
    all_identical = True
    for col1 in df_scores.columns:
        for col2 in df_scores.columns:
            if not np.array_equal(rankings[col1], rankings[col2]):
                all_identical = False
                print(f"  Rankings differ between {col1} and {col2}")
    print(f"  Are all rankings identical? {all_identical}")
    
    # Check correlation between all measures
    print("\n[STEP 9] Correlation analysis...")
    corr_matrix = df_scores.corr()
    print(f"  Correlation matrix:\n{corr_matrix}")
    
    # Check if all measures are constant
    print("\n[STEP 10] Variance in each measure...")
    for col in df_scores.columns:
        values = np.array(df_scores[col])
        print(f"  {col}: min={values.min():.6f}, max={values.max():.6f}, std={values.std():.6f}")
    
    print("\np[STEP 11] Correlation")

    # where df_scores is your DataFrame from the diagnostic run
    cols = list(df_scores.columns)
    print("Spearman rank correlations:")
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            r = spearmanr(df_scores[a].values, df_scores[b].values).correlation
            print(f"  {a:25s} vs {b:25s}: {r:.6f}")

    print("\nUnique value counts and tie fraction:")
    N = len(df_scores)
    for c in cols:
        vals = np.array(df_scores[c])
        nuniq = np.unique(vals).size
        ties_frac = 1.0 - (nuniq / N)
        print(f"  {c:25s}: unique={nuniq:4d}, ties_frac={ties_frac:.3f}")

    print("\nTop-10 indices+values per measure:")
    k = 10
    for c in cols:
        vals = np.array(df_scores[c])
        idx = np.argsort(-vals)[:k]
        print(f"  {c}: ", list(zip(idx.tolist(), vals[idx].round(6).tolist())))

    print("\n[STEP 12] LR repicrocal check...")
    # Check if LR reciprocal is constant across samples (it should be for Oracle) as est_dens = true_dens
    # assuming you already have: true_dens, est_dens_mean (or est_dens), y_grid, y_true, y_mode_pred
    N = true_dens.shape[0]
    y_grid = ds.y_grid
    y_mode_true = y_grid[true_dens.argmax(axis=1)]
    y_true = y_mode_true  # or use y_grid[true_dens.argmax(axis=1)] if you want the mode values directly

    est_dens.shape  # should be (N, G) or (5, N, G) for BMA
    est_dens.mean(axis=0).shape  # should be (N, G)
    est_dens.mean(axis=0).argmax(axis=-1).shape  # should be (N,)

    # Oracle mode predictions (should match true modes)
    y_mode_pred_idx = est_dens.mean(axis=0).argmax(axis=-1)  # shape (N,) - index of mode in y_grid
    y_mode_pred = y_grid[y_mode_pred_idx]  # shape (N,) - predicted mode values


    print("Check: do the Oracle predicted modes match the true modes?")
    mode_match = np.isclose(y_mode_pred, y_true)
    print(f"  Mode match fraction: {mode_match.mean():.3f}")

    idx_true = np.abs(y_grid[None,:] - y_true[:,None]).argmin(axis=1)
    idx_pred = np.abs(y_grid[None,:] - y_mode_pred[:,None]).argmin(axis=1)

    p_true_star = true_dens[np.arange(N), idx_true]
    p_true_hat  = true_dens[np.arange(N), idx_pred]

    lr_true_per_sample = p_true_star / (p_true_hat + 1e-12)
    lr_est_per_sample  = p_true_hat  / (p_true_star + 1e-12)  # same as using est_dens when equal

    print("Mean lr_true, mean lr_est:", lr_true_per_sample.mean(), lr_est_per_sample.mean())
    print("Elementwise product (should be ~1):", np.mean(lr_true_per_sample * lr_est_per_sample))
    # For clearer symmetry plot logs:
    print("Mean log-lr_true, mean log-lr_est:", np.mean(np.log(lr_true_per_sample+1e-12)), np.mean(np.log(lr_est_per_sample+1e-12)))

    print("\n[STEP 13] Subset selection check")
    # Use alpha_volume_total as uncertainty measure
    unc = df_scores['alpha_volume_total_5%'].values
    threshold_10pct = np.percentile(unc, 90)  # bottom 80% by uncertainty

    subset = unc <= threshold_10pct
    print(f"Subset size: {subset.sum()}")

    lr_true_subset = lr_true_per_sample[subset]
    lr_est_subset  = lr_est_per_sample[subset]

    print(f"  mean(lr_true) = {lr_true_subset.mean():.4f}")
    print(f"  mean(lr_est)  = {lr_est_subset.mean():.4f}")
    print(f"  product = {lr_true_subset.mean() * lr_est_subset.mean():.4f} (not 1.0, as expected)")
    print(f"  harmonic_mean(lr_true) = {len(lr_true_subset) / np.sum(1/lr_true_subset):.4f}")


    print("\n[STEP 14] Sample-level reciprocity check")

    # Check per-sample reciprocity directly, before any selective averaging.
    recip = lr_true_per_sample * lr_est_per_sample
    print(f"  Mean product: {recip.mean():.6f}")
    print(f"  Min product:  {recip.min():.6f}")
    print(f"  Max product:  {recip.max():.6f}")

    # Compare the vectors directly.
    print(f"  Max abs diff between lr_true and 1/lr_est: {np.max(np.abs(lr_true_per_sample - 1.0 / (lr_est_per_sample + 1e-12))):.6e}")

    # Optional: show a few samples.
    for i in range(min(10, N)):
        print(
            f"  i={i:3d}  lr_true={lr_true_per_sample[i]:.6f}  "
            f"lr_est={lr_est_per_sample[i]:.6f}  product={recip[i]:.6f}"
        )

    print("\n[STEP 15] Curve-level reciprocity check")

    # Use the same uncertainty score that drives selective ranking.
    unc = df_scores['alpha_volume_total_5%'].values
    steps = 20
    n = len(unc)
    order = np.argsort(-unc)

    print("  coverage  mean(lr_true)  mean(lr_est)  1/mean(lr_true)  mean(product)")
    for k in range(steps + 1):
        keep = order[k * n // (steps + 1):]
        if len(keep) == 0:
            continue

        lt = lr_true_per_sample[keep]
        le = lr_est_per_sample[keep]

        mean_lt = lt.mean()
        mean_le = le.mean()
        mean_inv_lt = 1.0 / (mean_lt + 1e-12)
        mean_prod = np.mean(lt * le)

        print(
            f"  {len(keep)/n:7.3f}   {mean_lt:12.6f}  {mean_le:11.6f}"
            f"  {mean_inv_lt:15.6f}  {mean_prod:12.6f}"
        )

    print("\n[STEP 16] Final reciprocity diagnostic on selective subsets")

    # Use the same uncertainty score that drives the selective curve.
    unc = df_scores['alpha_volume_total_5%'].values
    order = np.argsort(-unc)
    steps = 20
    n = len(unc)

    print("  coverage  mean(lr_true)  mean(lr_est)  1/mean(lr_true)  mean(lr_true*lr_est)")
    for k in range(steps + 1):
        keep = order[k * n // (steps + 1):]
        if len(keep) == 0:
            continue

        lt = lr_true_per_sample[keep]
        le = lr_est_per_sample[keep]

        mean_lt = lt.mean()
        mean_le = le.mean()
        mean_inv_lt = 1.0 / (mean_lt + 1e-12)
        mean_prod = np.mean(lt * le)

        print(
            f"  {len(keep)/n:7.3f}   {mean_lt:12.6f}  {mean_le:11.6f}"
            f"  {mean_inv_lt:15.6f}  {mean_prod:16.6f}"
        )

    print("\n  If reciprocity held after averaging, mean(lr_est) would match 1/mean(lr_true).")
    print("  If only sample-level reciprocity holds, mean(lr_true*lr_est) should stay near 1.")



    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(lr_true_per_sample, label='lr_true', marker='o', markersize=3)
    plt.plot(1 / (lr_est_per_sample + 1e-12), label='1/lr_est', marker='s', markersize=3)
    plt.xlabel('Sample index')
    plt.ylabel('LR value')
    plt.legend()
    plt.title('Per-sample lr_true vs 1/lr_est')
    # plt.savefig('lr_reciprocal_check.png')
    plt.close()

    print("\n[STEP 17] Plot curve-level reciprocity using ACTUAL metric function")

    from modal_uq.metrics.mode_errors import likelihood_ratio_measure
    import matplotlib.pyplot as plt

    unc = df_scores['alpha_volume_total_5%'].values
    order = np.argsort(-unc)
    steps = 20
    n = len(unc)

    coverages = []
    lr_true_vals = []
    lr_est_vals = []

    est_dens_mean = est_dens.mean(axis=0) if est_dens.ndim == 3 else est_dens

    for k in range(steps + 1):
        keep = order[k * n // (steps + 1):]
        if len(keep) == 0:
            continue
        
        # Compute modes for this subset
        y_mode_true_subset = ds.y_grid[true_dens[keep].argmax(axis=1)]
        y_mode_pred_subset = ds.y_grid[est_dens_mean[keep].argmax(axis=1)]
        
        # Call actual metric function for this subset
        kwargs_true = {
            'true_dens': true_dens[keep],
            'est_dens': est_dens_mean[keep],
            'y_grid': ds.y_grid,
            'reference_dist': 'true'
        }
        kwargs_est = {
            'true_dens': true_dens[keep],
            'est_dens': est_dens_mean[keep],
            'y_grid': ds.y_grid,
            'reference_dist': 'est'
        }
        
        # Compute modes for this subset
        y_mode_true_subset = ds.y_grid[true_dens[keep].argmax(axis=1)]
        y_mode_pred_subset = ds.y_grid[est_dens_mean[keep].argmax(axis=1)]

        # DEBUG: Check if modes are identical
        print(f"  Modes match? {np.allclose(y_mode_true_subset, y_mode_pred_subset)}")
        print(f"  Mode diff max: {np.max(np.abs(y_mode_true_subset - y_mode_pred_subset))}")
        print(f"  First 5 true modes: {y_mode_true_subset[:5]}")
        print(f"  First 5 pred modes: {y_mode_pred_subset[:5]}")



        lr_true = likelihood_ratio_measure(y_mode_true_subset, y_mode_pred_subset, kwargs_true)
        lr_est = likelihood_ratio_measure(y_mode_true_subset, y_mode_pred_subset, kwargs_est)
        
        coverages.append(len(keep) / n)
        lr_true_vals.append(lr_true)
        lr_est_vals.append(lr_est)

        # Right after keep = order[k * n // (steps + 1):] in the first iteration:
        if k == 0:  # First iteration only
            print(f"\n[DEBUG] Density values for first iteration:")
            print(f"  true_dens[keep] shape: {true_dens[keep].shape}")
            print(f"  Sample density sums (should be ~1 if normalized): {np.sum(true_dens[keep], axis=1)[:5]}")
            print(f"  true_dens[keep] min/max: {true_dens[keep].min():.6e}, {true_dens[keep].max():.6e}")
            print(f"  est_dens_mean[keep] min/max: {est_dens_mean[keep].min():.6e}, {est_dens_mean[keep].max():.6e}")


        print(f"  Coverage: {len(keep)/n:.3f}  LR_true: {lr_true:.6f}  LR_est: {lr_est:.6f}")

    # DEBUG: Plotting data verification
    print("\n[DEBUG] Plotting data verification:")
    print(f"  lr_true_vals: {lr_true_vals}")
    print(f"  lr_est_vals: {lr_est_vals}")
    print(f"  coverages: {coverages}")
    print(f"  100 * (1.0 - np.array(coverages)): {100 * (1.0 - np.array(coverages))}")

    

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(100 * (1.0 - np.array(coverages)), lr_true_vals, marker='o', label='LR_true')
    plt.plot(100 * (1.0 - np.array(coverages)), lr_est_vals, marker='s', label='LR_est')
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Expected (1.0)')
    plt.xlabel('% abstention')
    plt.ylabel('Likelihood Ratio')
    plt.title('Oracle Selective Prediction - LR should be 1.0')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig('selective_reciprocity_diagnostic_fixed.png', dpi=150)
    plt.close()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    test_oracle_diagnostic()