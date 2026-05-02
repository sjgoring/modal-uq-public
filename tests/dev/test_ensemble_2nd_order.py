import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import os
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
if __name__ == "__main__":
    for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import io
try:
    from matplotlib.widgets import Slider
    HAS_SLIDER = True
except Exception:
    HAS_SLIDER = False

# ----- Helper utilities for transforms, KDE, and Jacobians (Phase 1) -----
def alr_transform(weights):
    """Additive log-ratio transform for a weight vector or matrix.

    weights: array-like, shape (K,) or (K, n_samples)
    returns z: shape (K-1, n_samples)
    """
    w = np.asarray(weights)
    if w.ndim == 1:
        w = w[:, None]
    # Ensure numerical stability
    w = np.clip(w, 1e-12, 1.0)
    # reference is last component
    ref = -1
    w_ref = w[ref, :]
    z = np.log(w[:-1, :] / w_ref[None, :])
    return z


def alr_inverse(z):
    """Inverse ALR: map z (K-1, n) back to simplex weights (K, n)."""
    z = np.asarray(z)
    if z.ndim == 1:
        z = z[:, None]
    expz = np.exp(z)
    denom = 1.0 + np.sum(expz, axis=0)
    w_top = expz / denom[None, :]
    w_ref = 1.0 / denom
    w = np.vstack([w_top, w_ref[None, :]])
    return w


def safe_log(x):
    a = np.asarray(x)
    return np.log(np.clip(a, 1e-12, None))


def safe_exp(z):
    return np.exp(z)


def numerical_jacobian(func, x, eps=1e-6):
    """Numerical Jacobian of func at x using central differences.

    func: callable that maps R^d -> R^m
    x: array-like shape (d,)
    returns J shape (m,d)
    """
    x = np.asarray(x, dtype=float)
    fx = np.asarray(func(x))
    m = fx.size
    d = x.size
    J = np.zeros((m, d), dtype=float)
    for j in range(d):
        dx = np.zeros_like(x)
        dx[j] = eps
        f_plus = np.asarray(func(x + dx))
        f_minus = np.asarray(func(x - dx))
        J[:, j] = (f_plus.flatten() - f_minus.flatten()) / (2 * eps)
    return J


def safe_kde_2d(samples2d, grid_x, grid_y, bandwidth=None):
    """Evaluate a 2D KDE on the provided meshgrid.

    samples2d: shape (2, n_samples) OR (n_samples, 2)
    grid_x, grid_y: 1D arrays to build mesh
    returns: X, Y, Z where Z shape is (len(grid_y), len(grid_x)) matching meshgrid convention
    """
    try:
        from scipy.stats import gaussian_kde
        s = np.asarray(samples2d)
        if s.ndim == 2 and s.shape[0] != 2 and s.shape[1] == 2:
            s = s.T
        if s.shape[0] != 2:
            s = s.T
        kde = gaussian_kde(s)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_coords = np.vstack([gx.ravel(), gy.ravel()])
        z = kde(grid_coords)
        Z = z.reshape(gx.shape)
        return gx, gy, Z
    except Exception:
        # fallback to sklearn KernelDensity
        from sklearn.neighbors import KernelDensity
        s = np.asarray(samples2d)
        if s.ndim == 2 and s.shape[0] != 2 and s.shape[1] == 2:
            s = s
        if s.shape[0] != 2:
            s = s.T
        X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
        pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
        kd = KernelDensity(bandwidth=bandwidth or 1.0)
        kd.fit(s.T)
        logdens = kd.score_samples(pts)
        Z = np.exp(logdens).reshape(X_grid.shape)
        return X_grid, Y_grid, Z


def make_2d_grid(min1, max1, min2, max2, n1=100, n2=100, pad=0.1):
    r1 = max1 - min1
    r2 = max2 - min2
    g1 = np.linspace(min1 - pad * r1, max1 + pad * r1, n1)
    g2 = np.linspace(min2 - pad * r2, max2 + pad * r2, n2)
    return g1, g2


def get_param_blocks_and_labels(param_length, n_components=None):
    """Return offsets and label function for parameter vector of CondGMM.

    Assumes layout: [weights (K), means (K), covariances (K)] flattened per sample.
    """
    if n_components is None:
        # infer K assuming layout is 3K
        K = param_length // 3
    else:
        K = n_components
    offsets = {
        'weights': (0, K),
        'means': (K, 2 * K),
        'covs': (2 * K, 3 * K)
    }

    def label_for_index(idx):
        if offsets['weights'][0] <= idx < offsets['weights'][1]:
            i = idx - offsets['weights'][0]
            return f"weights_{i}"
        if offsets['means'][0] <= idx < offsets['means'][1]:
            i = idx - offsets['means'][0]
            return f"mu_{i}"
        if offsets['covs'][0] <= idx < offsets['covs'][1]:
            i = idx - offsets['covs'][0]
            return f"Sigma_{i}"
        return f"param_{idx}"

    return offsets, label_for_index

# ----- End helpers -----


def select_projection_and_compute_surface(params, offsets, n_components=None, grid_n=80, bandwidth=None):
    """Select top-2 transformed dims by variance and compute back-transformed density surface.

    params: array-like, shape (n_params, n_members) or (n_members, n_params)
    offsets: from `get_param_blocks_and_labels`
    returns: dict with keys `gx, gy, Z, labels, proj_dims`
    """
    P = np.asarray(params)
    if P.ndim == 2 and P.shape[0] < P.shape[1]:
        # probably (n_params, n_members) already; if not, try to transpose
        if P.shape[0] < 10 and P.shape[1] > 10:
            # leave as-is
            pass
    # Ensure shape (n_params, n_members)
    if P.shape[0] > P.shape[1]:
        # more params than members -> assume params are rows
        if P.shape[1] < 10:
            P = P
    if P.shape[1] < P.shape[0]:
        # If it's (n_members, n_params) transpose
        if P.shape[0] != P.shape[1]:
            P = P.T

    n_params, n_members = P.shape
    # extract blocks
    w0, w1 = offsets['weights']
    m0, m1 = offsets['means']
    c0, c1 = offsets['covs']
    weights = P[w0:w1, :]
    mus = P[m0:m1, :]
    covs = P[c0:c1, :]

    # transform blocks
    z_w = alr_transform(weights)  # (K-1, n_members)
    z_mu = mus  # identity
    z_cov = safe_log(covs)

    transformed = np.vstack([z_w, z_mu, z_cov])

    # compute per-dimension variance and pick top-2
    var_per_dim = np.var(transformed, axis=1)
    # if there are fewer than 2 dims, pad
    d = transformed.shape[0]
    top2 = np.argsort(var_per_dim)[-2:][::-1]
    if top2.size < 2:
        top2 = np.arange(min(2, d))

    # samples for KDE
    samples2d = transformed[top2, :]

    # grid
    g1, g2 = make_2d_grid(np.min(samples2d[0, :]), np.max(samples2d[0, :]),
                          np.min(samples2d[1, :]), np.max(samples2d[1, :]),
                          n1=grid_n, n2=grid_n)

    gx, gy, Z = safe_kde_2d(samples2d, g1, g2, bandwidth=bandwidth)

    # Back-transform each grid point to original parameter subset while holding other transformed dims at their mean
    mean_trans = np.mean(transformed, axis=1)

    # derived counts for mapping transformed indices back to original param vector indices
    K = w1 - w0
    mu_count = m1 - m0

    def transformed_index_to_original(tidx):
        if tidx < (K - 1):
            return tidx
        if tidx < (K - 1) + mu_count:
            return (K + (tidx - (K - 1)))
        return (2 * K + (tidx - (K - 1) - mu_count))


    def map_uv_to_output(uv):
        # uv: (2,) in transformed coords for selected dims
        full = mean_trans.copy()
        full[top2[0]] = uv[0]
        full[top2[1]] = uv[1]
        # now map full transformed vector back to original parameters
        # inverse for weights
        K = w1 - w0
        z_w_full = full[0:(K - 1)] if K - 1 > 0 else np.array([])
        if z_w_full.size > 0:
            w_full = alr_inverse(z_w_full)
            w_full = w_full.ravel()
        else:
            w_full = np.ones(K) / K
        # inverse for mus
        mu_start = (K - 1)
        mu_end = mu_start + (m1 - m0)
        mu_full = full[mu_start:mu_end]
        # inverse for covs
        cov_start = mu_end
        cov_end = cov_start + (c1 - c0)
        cov_full = safe_exp(full[cov_start:cov_end])
        # assemble original param vector (weights, mus, covs)
        out = np.concatenate([w_full, mu_full, cov_full])
        # choose to return only the two original quantities corresponding to the selected transformed dims
        # find mapping from transformed index -> original index (approx)
        o1 = transformed_index_to_original(top2[0])
        o2 = transformed_index_to_original(top2[1])
        # clamp
        o1 = int(np.clip(o1, 0, out.size - 1))
        o2 = int(np.clip(o2, 0, out.size - 1))
        return np.array([out[o1], out[o2]])

    # compute Jacobian correction and apply to Z
    X_flat = gx.ravel()
    Y_flat = gy.ravel()
    pts = np.vstack([X_flat, Y_flat]).T
    J_abs = np.zeros(pts.shape[0], dtype=float)
    for i, p in enumerate(pts):
        try:
            J = numerical_jacobian(map_uv_to_output, p)
            det = np.linalg.det(J)
            J_abs[i] = abs(det) if np.isfinite(det) and det != 0 else 1.0
        except Exception:
            J_abs[i] = 1.0

    Z_flat = Z.ravel()
    Z_corrected = Z_flat / (J_abs + 1e-18)
    Z_corrected = Z_corrected.reshape(Z.shape)
    # normalize
    Z_corrected = Z_corrected / (np.sum(Z_corrected) + 1e-18)

    # labels
    offsets_local, label_fn = get_param_blocks_and_labels(n_params, n_components)
    labels = (label_fn(transformed_index_to_original(top2[0])),
              label_fn(transformed_index_to_original(top2[1])))

    return dict(gx=gx, gy=gy, Z=Z_corrected, labels=labels, proj_dims=top2)

from modal_uq.datasets.synthetic_constant_var import SyntheticConstantVarDataset
from modal_uq.datasets.moons_synthetic import MoonsSyntheticDataset
from modal_uq.models.ensemble import Ensemble
from modal_uq.models.condGMM import CondGMM


def make_small_dataset(seed, n_samples=200, test_size=0.3, source="MOONS"):
    if source == "MOONS":
        ds = MoonsSyntheticDataset(n_samples=int(n_samples / (1 - test_size)), noise=0.1, random_state=seed)
        X, y = ds.sample()
    elif source == "SYNTH_MULTI_MODAL":
        ds = SyntheticConstantVarDataset(n_samples=n_samples)
        X, y, _, _, _ = ds.get_data()
    else:
        raise ValueError(f"Unknown dataset: {source}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test


def test_ensemble_2nd_order(capsys=None):
    show_plots = capsys is None
    seed = 0
    n_samples = 2000
    data_source = "SYNTH_MULTI_MODAL"

    ensemble_kwargs = {
        "base_model": "condgmm",
        "base_params": {"n_components": 2},
        "n_members": 10, #must be greater than number of params for KDE estimate.
        "bootstrap": True,
        "seed": 42,
    }

    X_train, X_test, y_train, y_test = make_small_dataset(seed=seed, n_samples=n_samples, source=data_source)

    model = Ensemble(**ensemble_kwargs)
    model.fit(X_train, y_train)

    y_grid = model.default_y_grid(X_test, grid_points=128)
    dens = model.predict_density(X_test, y_grid)
    theta_grid = model.default_theta_grid(X_test, num_points=100)
    second_order, _ = model.get_second_order_distribution(X_test, theta_grid)

    assert dens.ndim in (2, 3)
    assert second_order is not None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join("runs", "_ensemble_2nd_order", timestamp)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "timestamp": timestamp,
        "data_source": data_source,
        "n_samples": n_samples,
        "ensemble_kwargs": ensemble_kwargs,
        "y_grid_points": 128,
        "x_test_shape": list(X_test.shape),
        "y_test_shape": list(y_test.shape),
        "dens_shape": list(dens.shape),
    }

    config_path = os.path.join(out_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, sort_keys=True))

    output_path = os.path.join(out_dir, "output.txt")
    if capsys is not None:
        captured = capsys.readouterr()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(captured.out)
            if captured.err:
                f.write("\n\n[stderr]\n")
                f.write(captured.err)

    # ===== Ensemble theta / predictive visualization =====
    try:
        if capsys is None:
            theta_grid = model.default_theta_grid(X_test)
            theta_dens_by_X, theta_grid = model.get_second_order_distribution(X_test, theta_grid)
            captured_plot_info = ""
        else:
            buf = io.StringIO()
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = buf
            sys.stderr = buf
            try:
                theta_grid = model.default_theta_grid(X_test)
                theta_dens_by_X, theta_grid = model.get_second_order_distribution(X_test, theta_grid)
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
            captured_plot_info = buf.getvalue()
    except Exception as e:
        theta_dens_by_X = None
        captured_plot_info = f"[ERROR] computing theta densities: {e}\n"

    plot_path = os.path.join(out_dir, "ensemble_density.png")
    try:
        if theta_dens_by_X is not None:
            # prepare predictive BMA (ensure shape [N,G])
            if dens.ndim == 3:
                predictive_bma = dens.mean(axis=0)
            else:
                predictive_bma = dens

            # Extract n_components for labeling
            try:
                n_components = model.members[0].model.n_components
            except (AttributeError, IndexError):
                n_components = 2  # fallback
            
            def get_param_label(param_idx, n_comp):
                """Return label for parameter at param_idx (after dropping first weight).
                
                Parameters are concatenated as: [weights | means | covariances]
                After dropping weights[0], indices map as:
                - 0 to n_comp-2: weights_{1} to weights_{n_comp-1}
                - n_comp-1 to 2*n_comp-2: mu_0 to mu_{n_comp-1}
                - 2*n_comp-1 onward: Sigma_0 to Sigma_{n_comp-1}
                """
                if param_idx < n_comp - 1:
                    # weights[1:]
                    return f"weights_{param_idx + 1}"
                elif param_idx < 2 * n_comp - 1:
                    # means
                    mu_idx = param_idx - (n_comp - 1)
                    return f"mu_{mu_idx}"
                else:
                    # covariances
                    sigma_idx = param_idx - (2 * n_comp - 1)
                    return f"Sigma_{sigma_idx}"

            def plot_sample(idx):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # left: up to first 5 theta-dim curves for the chosen sample
                # theta_dens_by_X expected shapes: [n_X, n_params, n_points] or [n_X, n_points]
                if theta_dens_by_X.ndim == 3:
                    n_params = theta_dens_by_X.shape[1]
                    n_thetas = min(n_params, 5)
                    for t in range(n_thetas):
                        y_vals = theta_dens_by_X[idx, t, :]
                        ax1.plot(theta_grid[t], y_vals, alpha=0.8, label=f"theta_{t}")
                elif theta_dens_by_X.ndim == 2:
                    # fallback: single parameter densities per X
                    y_vals = theta_dens_by_X[idx, :]
                    x_for_theta = theta_grid if np.ndim(theta_grid) == 1 else theta_grid[0]
                    ax1.plot(x_for_theta, y_vals, alpha=0.8, label="theta_0")
                ax1.set_xlabel("theta")
                ax1.set_ylabel("density")
                ax1.set_title(f"Theta densities (sample {idx})")
                ax1.legend()

                # right: predictive BMA for same sample (use y_grid as x-axis)
                ax2.plot(y_grid, predictive_bma[idx, :], color="black", lw=2)
                ax2.fill_between(y_grid, predictive_bma[idx, :], alpha=0.2, color="grey")
                ax2.set_xlabel("y")
                ax2.set_ylabel("density")
                ax2.set_title(f"Predictive BMA (sample {idx})")

                plt.tight_layout()
                fig.savefig(plot_path, dpi=100, bbox_inches="tight")
                if not show_plots:
                    plt.close(fig)
            
            def update_axes_grid(axes_grid, sample_idx, theta_grid_vals):
                """Update subplot grid with marginal parameter densities for sample_idx.
                
                Parameters
                ----------
                axes_grid : list of lists or 1D array of Axes
                    Flattened subplot axes
                sample_idx : int
                    Index into X_test samples (0 to n_X-1)
                theta_grid_vals : 2D array
                    Parameter grids, shape [n_params-1, num_points]
                """
                # Extract marginals for this sample: [n_params-1, num_points]
                marginals = theta_dens_by_X[sample_idx, :, :]
                n_params = marginals.shape[0]
                
                # Flatten axes for easy iteration
                if isinstance(axes_grid, np.ndarray):
                    axes_flat = axes_grid.flatten()
                else:
                    axes_flat = [ax for row in axes_grid for ax in row] if isinstance(axes_grid[0], (list, tuple)) else axes_grid
                
                # Plot each marginal
                for param_idx in range(n_params):
                    ax = axes_flat[param_idx]
                    ax.clear()
                    
                    # Get grid values for this parameter
                    if theta_grid_vals.ndim == 2:
                        x_grid = theta_grid_vals[param_idx, :]
                    else:
                        x_grid = theta_grid_vals
                    
                    y_vals = marginals[param_idx, :]
                    ax.plot(x_grid, y_vals, color="steelblue", lw=1.5)
                    ax.fill_between(x_grid, y_vals, alpha=0.3, color="steelblue")
                    
                    param_label = get_param_label(param_idx, n_components)
                    ax.set_ylabel(param_label, fontsize=9)
                    ax.set_xlabel("value", fontsize=8)
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for param_idx in range(n_params, len(axes_flat)):
                    axes_flat[param_idx].set_visible(False)
                
                plt.suptitle(f"Marginal Parameter Densities (X sample {sample_idx})", fontsize=12, y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])

            # interactive slider if available; otherwise static plot of sample 0
            if HAS_SLIDER and X_test.shape[0] > 1:
                print("Creating interactive plot with slider over X_test samples and 3D surface...")
                print(f"DEBUG: theta_dens_by_X shape = {theta_dens_by_X.shape}, ndim = {theta_dens_by_X.ndim}")
                print(f"DEBUG: theta_grid shape = {theta_grid.shape}")
                print(f"DEBUG: X_test.shape[0] = {X_test.shape[0]}")
                try:
                    # Prepare marginals grid
                    # Handle both 2D and 3D cases
                    if theta_dens_by_X.ndim == 2:
                        # Only 1 sample; reshape to (1, n_params, n_points)
                        theta_dens_by_X = theta_dens_by_X[np.newaxis, :, :]
                    n_params = theta_dens_by_X.shape[1]
                    n_cols = int(np.ceil((n_params) / 2))
                    n_rows = 2
                    # Build a figure with extra column for 3D surface
                    from matplotlib import gridspec
                    fig = plt.figure(figsize=(4 * n_cols + 6, 3 * n_rows))
                    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1] * n_cols + [1.4])

                    axes = np.empty((n_rows, n_cols), dtype=object)
                    for r in range(n_rows):
                        for c in range(n_cols):
                            axes[r, c] = fig.add_subplot(gs[r, c])

                    # 3D axis occupies the last column across rows
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                    ax3d = fig.add_subplot(gs[:, -1], projection='3d')

                    # params across members: [n_params, n_X, n_members]
                    params_stack = np.stack([m.get_params(X_test) for m in model.members], axis=-1)
                    n_params_total = params_stack.shape[0]
                    offsets, label_fn = get_param_blocks_and_labels(n_params_total, n_components)

                    # initial plots
                    update_axes_grid(axes, 0, theta_grid)

                    # compute initial surface for sample 0
                    sample_params = params_stack[:, 0, :]
                    surf_payload = select_projection_and_compute_surface(sample_params, offsets, n_components=n_components, grid_n=80)
                    surf = ax3d.plot_surface(surf_payload['gx'], surf_payload['gy'], surf_payload['Z'], cmap='viridis', linewidth=0, antialiased=True)
                    ax3d.set_xlabel(surf_payload['labels'][0])
                    ax3d.set_ylabel(surf_payload['labels'][1])
                    ax3d.set_zlabel('density')

                    # Add slider at the bottom
                    ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03])
                    slider = Slider(ax_slider, "X sample", 0, X_test.shape[0] - 1, valinit=0, valstep=1)

                    def update(val):
                        nonlocal surf
                        sample_idx = int(slider.val)
                        update_axes_grid(axes, sample_idx, theta_grid)
                        # recompute surface for new sample
                        sample_params = params_stack[:, sample_idx, :]
                        try:
                            new_payload = select_projection_and_compute_surface(sample_params, offsets, n_components=n_components, grid_n=60)
                            try:
                                surf.remove()
                            except Exception:
                                pass
                            surf = ax3d.plot_surface(new_payload['gx'], new_payload['gy'], new_payload['Z'], cmap='viridis', linewidth=0, antialiased=True)
                            ax3d.set_xlabel(new_payload['labels'][0])
                            ax3d.set_ylabel(new_payload['labels'][1])
                        except Exception as e:
                            print(f"Warning: failed to update 3D surface for sample {sample_idx}: {e}")
                        fig.canvas.draw_idle()

                    slider.on_changed(update)
                except Exception as e:
                    print(f"Error in interactive plot: {e}")
                    # fallback to static plot
                    plot_sample(0)
            else:
                # static saved plot of sample 0
                print("Creating static plot (HAS_SLIDER={}, X_test.shape[0]={})...".format(HAS_SLIDER, X_test.shape[0]))
                plot_sample(0)
            if show_plots:
                plt.show()
        else:
            captured_plot_info += "[INFO] theta_dens_by_X is None; skipping plotting.\n"
    except Exception as e:
        captured_plot_info += f"[ERROR] plotting: {e}\n"

    # Append plot info to output.txt
    if capsys is not None:
        try:
            if captured_plot_info:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write("\n\n[Ensemble Visualization]\n")
                    f.write(captured_plot_info)
                assert os.path.exists(output_path)
        except Exception:
            pass

    assert os.path.exists(config_path)
    


if __name__ == "__main__":
    capsys = None  # No capturing when running directly
    test_ensemble_2nd_order(capsys)