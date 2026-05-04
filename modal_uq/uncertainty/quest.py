import numpy as np
from .base import UncertaintyBase
from ..models.base import InferentialChoiceConfig
from ..registry import register
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

from modal_uq.models.ensemble import Ensemble

@register('uncertainty','alpha_volume')
@register('uncertainty','integrated_volume')
class QUESTUncertainty(UncertaintyBase):
    """
    QUEST uncertainty using Highest Density Regions (HDR).
    
    Computes either alpha volume or intergrated volume .
    Alpha volume - the Lebesgue measure (total length) of the highest density region
    containing (1 - alpha) probability mass.
    Integrated volume - curve of alpha volume against alpha, integrated over alpha in (0,1).
    
    Uses dual inferential_choice contexts for uncertainty decomposition:
    - total:      Lebesgue measure of HDR from predict context
    - aleatoric:  Lebesgue measure of HDR from approximate context (true DGP)
    - epistemic:  TBD - HDR-based measure does not naturally decompose via subtraction
    
    The epistemic component for HDR-based measures requires domain-specific analysis
    and is left as a stub for future implementation.
    """
    def __init__(self, alpha=None, decomposition='total', scope='local', grid_points=512, y_pad=1.0, n_param_samples=20, mc_n_samples=100000, mc_random_state=None, bounds_q_low=1e-3, bounds_q_high=1.0-1e-3, bounds_pad_frac=0.05, n_alpha=100, n_jobs=None):
    # def __init__(self, alpha, decomposition='total', grid_points=10000, y_pad=1.0, n_param_samples=20):
        assert decomposition in {'total','aleatoric','epistemic'}
        self.alpha = alpha
        self.decomposition = decomposition
        self.scope = scope
        self.grid_points = grid_points
        self.y_pad = y_pad
        self.n_param_samples = n_param_samples
        # Monte Carlo settings for HDR / Lebesgue estimation
        self.mc_n_samples = mc_n_samples
        self.mc_random_state = mc_random_state
        self.bounds_q_low = bounds_q_low
        self.bounds_q_high = bounds_q_high
        self.bounds_pad_frac = bounds_pad_frac
        self.n_alpha = n_alpha
        self.n_jobs = n_jobs  # Number of parallel jobs for per-input loops

        if self.alpha == None and scope == 'local':
            raise ValueError("alpha must be provided for local scope.")

    @staticmethod
    def _normalize_last_axis(arr, y_grid):
        """
        Normalize densities along the last axis (G), regardless of arr being [N,G] or [S,N,G].
        """
        Z = integrate.trapezoid(arr, y_grid, axis=-1)     # shape: [N] or [S,N]
        Z = np.expand_dims(Z, axis=-1)         # -> [N,1] or [S,N,1]
        return arr / (Z + 1e-12)

    def _validate_inferential_choices(self, model):
        """Validate inferential choice configuration for BMA-based QUEST.

        Skip validation for `epistemic` decomposition which uses Ensemble-specific
        second-order KDE/MC logic and does not require predict/approximate contexts.
        """
        if self.decomposition == 'epistemic':
            return

        cfg = model.get_inferential_choice_config()
        predict = InferentialChoiceConfig.canonicalize_strategy(cfg.predict)
        approximate = InferentialChoiceConfig.canonicalize_strategy(cfg.approximate)

        if predict != 'bma' or approximate != 'posterior_predictive':
            raise NotImplementedError(
                "QUESTUncertainty requires inferential choices predict='bma' and "
                "approximate='posterior_predictive' for total/aleatoric decompositions. "
                f"Current settings: predict='{cfg.predict}', approximate='{cfg.approximate}'."
            )

    # @staticmethod
    # def _lebesgue_measure_hdr(mask, y_grid):
    #     """
    #     Compute the Lebesgue measure (total length) of the Highest Density Regions.

    #     Parameters
    #     ----------
    #     mask : bool array of shape [N, G]
    #         Boolean mask where True indicates points in the HDR.
    #     y_grid : array of shape [G]
    #         Grid over which density is defined.

    #     Returns
    #     -------
    #     lebesgue_measure : array of shape [N]
    #         Total length (Lebesgue measure) of the HDR for each sample.
    #     """
    #     y_grid = np.asarray(y_grid)
    #     dy = np.diff(y_grid)
    #     dy = np.concatenate([dy, [dy[-1]]])
    #     lebesgue_measure = np.sum(mask * dy, axis=1)
    #     return lebesgue_measure
    
    # @staticmethod
    def _lebesgue_measure_hdr(self, mask, grid):
        """
        Compute the Lebesgue measure for multidimensional HDR, where grid is shape [D,G] and mask is shape [G].
        Note this does not just accept output from HDR, need to do call this per x_i, not per X.

        Shapes
        ------
        G - number of grid points (per dimension)
        D - dimensionality of output space (e.g. D=2 for bivariate density)

        Parameters
        ----------
        mask : bool array of shape [G]
            Boolean mask where True indicates points in the HDR.
        grid : array of shape [D, G]
            Grid over which density is defined
        """

        # Deprecated grid-based Lebesgue helper kept for reference.
        raise NotImplementedError("Grid-based Lebesgue measure is deprecated. Use Monte Carlo methods instead.")

    # @staticmethod
    # def _hdr_from_density(dens, y_grid, alpha):
    #     """
    #     Compute the Highest Density Region (HDR) containing (1 - alpha) probability,
    #     following Hyndman (1996) "Computing and Graphing Highest Density Regions".

    #     Parameters
    #     ----------
    #     dens : array
    #         Normalized densities over y_grid
    #         Shape can be [G], [N,G], or [S,N,G].
    #         If [S,N,G] we compute HDR per [N,G] mixture density.
    #     y_grid : array of shape [G]
    #         Grid over which density is defined.
    #     alpha : float
    #         Tail probability, i.e. HDR contains probability mass (1 - alpha).

    #     Returns
    #     -------
    #     threshold : array of shape [N]
    #         Density threshold t_alpha such that HDR = { y : p(y) >= t_alpha }.
    #     mask : bool array of shape [N, G]
    #         True where y_grid belongs to the HDR.
    #     """

    #     # if support more than 1D, raise an error for now - need to adapt sorting and cumulative mass logic
    #     if dens.ndim > 3:
    #         raise ValueError("Density must be at most 3D: [G], [N,G], or [S,N,G].")

    #     # Ensure last axis is grid axis
    #     dens = np.asarray(dens)

    #     # If dens is [S,N,G], average over S to get mixture density
    #     if dens.ndim == 3:
    #         dens = dens.mean(axis=0)   # -> [N,G]
    #     elif dens.ndim == 1:
    #         dens = dens[None, :]       # -> [1,G]

    #     # At this point:
    #     # dens shape is [N,G]
    #     N, G = dens.shape
    #     y_grid = np.asarray(y_grid)

    #     # Normalize densities along grid axis
    #     Z = integrate.trapezoid(dens, y_grid, axis=-1)[:, None] + 1e-12
    #     p = dens / Z  # now integrates to 1

    #     # Sort densities descending along grid axis
    #     # idx_sorted: [N,G]
    #     idx_sorted = np.argsort(-p, axis=1)
    #     p_sorted = np.take_along_axis(p, idx_sorted, axis=1)

    #     # Cumulative integral of sorted densities
    #     # Note: spacing varies across grid, so trapz can't be used directly.
    #     # Approximate cumulative mass via cumulative sum of p_sorted * delta_y.
    #     # Compute delta_y for each grid point:
    #     dy = np.diff(y_grid)
    #     # pad last value so shapes broadcast: [G]
    #     dy = np.concatenate([dy, [dy[-1]]])

    #     # cumulative mass: [N,G]
    #     cum_mass = np.cumsum(p_sorted * dy, axis=1)

    #     # Find minimal j where cum_mass >= (1 - alpha)
    #     target = 1.0 - alpha
    #     # idx_thresh[n] = threshold index for density of sample n
    #     idx_thresh = np.argmin(cum_mass < target, axis=1)  # [N]

    #     # Extract threshold values t_alpha[n] = p_sorted[n, idx_thresh[n]]
    #     threshold = p_sorted[np.arange(N), idx_thresh]

    #     # Build HDR mask: p[n,g] >= threshold[n]
    #     mask = p >= threshold[:, None]

    #     return threshold, mask
    

    # @staticmethod
    def _hdr_from_density(self, dens, grid, alpha):
        """Compute the Highest Density Region (HDR) containing (1 - alpha) probability,
        
        Shapes
        ------        
        grid: requires a numpy meshgrid of shape [D,G,...,G] for D-dimensional output space, with G grid points per dimension. This allows for non-axis-aligned HDRs in multivariate spaces.
        dens: expects dens aligned with meshgrid shape [1, G, ..., G] 

        """

        raise NotImplementedError("Grid-based HDR computation is deprecated. Use Monte Carlo methods via model.sample_output, model.output_bounds and model.density_function_for_input.")

    def _hdr_resampler_from_samples(self, samples, sample_mask):
        """Create a sampler function that resamples from HDR-accepted MC samples.

        Parameters
        - samples: array of shape (S, N, d) (MC samples produced earlier)
        - sample_mask: boolean array of shape (S, N) indicating HDR membership

        Returns
        - sampler(X, n_samples, rng) -> (n_samples, N, d)
        """
        samples = np.asarray(samples)
        sample_mask = np.asarray(sample_mask)

        def sampler(X_, n_samples, rng):
            rng = np.random.default_rng(rng)
            N = X_.shape[0]
            d = samples.shape[-1]
            out = np.zeros((n_samples, N, d))
            for i in range(N):
                hdr_samples = samples[sample_mask[:, i], i, :]
                if hdr_samples.shape[0] > 0:
                    idx = rng.choice(hdr_samples.shape[0], size=n_samples, replace=True)
                    out[:, i, :] = hdr_samples[idx]
                else:
                    out[:, i, :] = rng.standard_normal((n_samples, d))
            return out

        return sampler

    def _build_bounds_from_samples(self, samples, q_low=None, q_high=None, pad_frac=None):
        q_low = self.bounds_q_low if q_low is None else q_low
        q_high = self.bounds_q_high if q_high is None else q_high
        pad_frac = self.bounds_pad_frac if pad_frac is None else pad_frac
        # samples: (n_samples, N, d)
        lows = np.quantile(samples, q_low, axis=0)  # (N, d)
        highs = np.quantile(samples, q_high, axis=0)  # (N, d)
        pad = (highs - lows) * pad_frac
        bounds = np.stack([lows - pad, highs + pad], axis=-1)  # (N, d, 2)
        return bounds

    def _hdr_from_density_grid_1d(self, density_func, X, alpha=0.05, grid=None):
        """Compute 1D HDR volume on a grid.

        This helper is only for the simple 1D case. Multivariate densities must
        continue using the Monte Carlo HDR path.
        """
        X = np.asarray(X)
        if grid is None:
            grid = np.asarray(self._default_density_grid(X))
        else:
            grid = np.asarray(grid)

        if grid.ndim != 1:
            raise ValueError("1D grid HDR helper requires a 1D grid.")

        y_grid = grid
        dy = np.diff(y_grid)
        if dy.size == 0:
            raise ValueError("Grid must contain at least two points for HDR computation.")
        dy = np.concatenate([dy, [dy[-1]]])

        volumes = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            dens = np.asarray(density_func(y_grid, input_index=i)).reshape(-1)
            if dens.shape[0] != y_grid.shape[0]:
                raise ValueError("1D grid density callable must return one density value per grid point.")

            z = integrate.trapezoid(dens, y_grid)
            p = dens / (z + 1e-12)
            idx_sorted = np.argsort(-p)
            p_sorted = p[idx_sorted]
            cum_mass = np.cumsum(p_sorted * dy)
            cutoff_idx = np.searchsorted(cum_mass, 1.0 - alpha, side='left')
            cutoff_idx = min(cutoff_idx, p_sorted.shape[0] - 1)
            threshold = p_sorted[cutoff_idx]
            mask = p >= threshold
            volumes[i] = np.sum(mask * dy)

        return volumes

    def _default_density_grid(self, X):
        """Return the local 1D evaluation grid used by QUEST helpers."""
        # The 1D helper is intentionally local and deterministic.
        # If the model provides a custom grid, the caller should pass it in.
        return np.linspace(-self.y_pad, self.y_pad, self.grid_points)

    def _hdr_from_density_function(self, density_func, sampler, X, alpha=0.05, n_samples=None, random_state=None):
        """
        Monte Carlo HDR threshold estimation using samples from the target density.

        sampler(X, n_samples, rng) -> (n_samples, N, d)
        density_func(points, input_index) -> (M,) densities
        Returns (c_alpha, hdr_indicator, samples, sample_mask)
        """
        n_samples = int(n_samples or self.mc_n_samples)
        rng = np.random.default_rng(random_state or self.mc_random_state)
        samples = sampler(X, n_samples, rng)  # (n_samples, N, d)
        samples = np.asarray(samples)
        n_samples_local, N, d = samples.shape

        # Helper function for per-input computation (parallelizable)
        def _compute_hdr_threshold_for_input(i):
            pts = samples[:, i, :]
            dens_vals = density_func(pts, input_index=i)
            dens_vals = np.asarray(dens_vals).ravel()
            # sort densities descending
            idx = np.argsort(-dens_vals)
            sorted_dens = dens_vals[idx]
            cutoff_idx = max(0, int(np.ceil((1.0 - alpha) * n_samples_local)) - 1)
            thresh = sorted_dens[cutoff_idx]
            mask = dens_vals >= thresh
            return thresh, mask

        # Parallelize over inputs
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') and self.n_jobs is not None else 1
        if n_jobs == 1:
            # Serial execution
            results = [_compute_hdr_threshold_for_input(i) for i in range(N)]
        else:
            # Parallel execution
            results = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_compute_hdr_threshold_for_input)(i) for i in range(N)
            )

        c_alpha = np.array([r[0] for r in results])
        sample_mask = np.stack([r[1] for r in results], axis=1)  # (n_samples, N)

        def hdr_indicator(query_points, input_index=0):
            q = np.asarray(query_points)
            return density_func(q, input_index=input_index) >= c_alpha[input_index]

        return c_alpha, hdr_indicator, samples, sample_mask

    def _lebesgue_measure_hdr_mc(self, density_func, c_alpha, bounds, n_samples=None, random_state=None):
        """
        Estimate Lebesgue measure via uniform sampling inside per-input axis-aligned bounds.

        bounds: (N, d, 2)
        Returns: volumes (N,), fractions_inside (N,)
        """
        n_samples = int(n_samples or self.mc_n_samples)
        rng = np.random.default_rng(random_state or self.mc_random_state)
        bounds = np.asarray(bounds)
        N, d, two = bounds.shape

        # Helper function for per-input computation (parallelizable)
        def _compute_volume_for_input(i):
            lows = bounds[i, :, 0]
            highs = bounds[i, :, 1]
            U = rng.uniform(lows, highs, size=(n_samples, d))
            dens_vals = density_func(U, input_index=i)
            inside = np.asarray(dens_vals).ravel() >= c_alpha[i]
            fraction_inside = np.mean(inside)
            box_vol = float(np.prod(highs - lows))
            volume = box_vol * fraction_inside
            return volume, fraction_inside

        # Parallelize over inputs
        n_jobs = self.n_jobs if hasattr(self, 'n_jobs') and self.n_jobs is not None else 1
        if n_jobs == 1:
            # Serial execution
            results = [_compute_volume_for_input(i) for i in range(N)]
        else:
            # Parallel execution
            results = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_compute_volume_for_input)(i) for i in range(N)
            )

        volumes = np.array([r[0] for r in results])
        fractions = np.array([r[1] for r in results])
        return volumes, fractions

    def _grid_helper(self, grid):
        """
        Expects a numpy meshgrid of shape [D,G,...,G] for D-dimensional output space, with G grid points per dimension.
        Returns (is_uniform, cell_volume) where is_uniform is True if grid is uniformly spaced in each dimension, and cell_volume is the volume of each grid cell.
        """
        # take slice of grid along each dimension, check uniform spacing, and compute cell volume as product of spacings.
        



    def _grid_helper(self, grid):
        """
        Check if grid is uniformly spaced in each dimension, and return volume.

        Shapes
        grid: expects a full meshgrid of shape [D,G,...,G] for D-dimensional output space, with G grid points per dimension.

        """
        grid = np.asarray(grid)
        #Ensure grid is shape [D,G]
        if grid.shape[0] != grid.ndim-1:
            raise ValueError("Grid must be shape [D,G] for D-dimensional output space. Received shape: {}".format(grid.shape))
        # if grid.ndim == 1:
        #     grid = grid[None, :]
        # diffs = []
        grid_unif = True

        # check uniform spacing in each dimension
        diffs = []

        for d in range(grid.shape[0]):
            diffs_i = np.diff(grid[d,:])
            if not np.all(np.isclose(diffs_i, diffs_i[0,])):
                return False, None
            diffs.append(diffs_i[0])
        # find cell volume
        cell_vol = np.prod(diffs)
        print("Grid diffs: {}, cell volume: {}".format(diffs, cell_vol))
        return grid_unif, cell_vol



    def _compute_total(self, model, X):
        if self.scope == 'local':
            return self._compute_total_helper(model, X, alpha=self.alpha)
        else:
            # integral under curve for various alpha
            f = []
            alphas = np.linspace(0.01, 0.99, num=100)
            for alpha in alphas:
                f.append(np.asarray(self._compute_total_helper(model, X, alpha=alpha)))
            f = np.stack(f, axis=0)
            return np.trapz(f, alphas, axis=0)

    def _compute_total_helper(self, model, X, alpha):
        """Compute local total uncertainty as BMA of (alpha_volume / (1 - TV)) over predict samples.
        
        For each predict sample s:
        1. Compute alpha_volume(predict_s, alpha)
        2. Compute TV distance between approximate (reference) and predict_s
        3. Compute result_s = alpha_volume_s / (1 - tv_s)
        Average across all predict samples: TU_local = mean([result_s])
        """
        # Override alpha and scope
        alpha_old = self.alpha
        scope_old = self.scope
        self.alpha = alpha
        self.scope = 'local'

        try:
            y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
            y_grid = np.asarray(y_grid)

            if y_grid.ndim == 1:
                predict_dens = self._as_density_collection(model.predict_density(X, y_grid, context='predict'))
                approximate_dens = self._as_density_collection(model.predict_density(X, y_grid, context='approximate'))

                predict_mean = np.mean(predict_dens, axis=0)
                approximate_mean = np.mean(approximate_dens, axis=0)

                def predict_density_1d(points, input_index=0):
                    points = np.asarray(points).reshape(-1)
                    return np.interp(points, y_grid, predict_mean[input_index])

                def approximate_density_1d(points, input_index=0):
                    points = np.asarray(points).reshape(-1)
                    return np.interp(points, y_grid, approximate_mean[input_index])

                _ = approximate_density_1d  # retained for symmetry with the total decomposition
                return self._hdr_from_density_grid_1d(predict_density_1d, X, alpha=alpha, grid=y_grid)

            # Multivariate fallback: preserve existing Monte Carlo behaviour unchanged.
            sampler = lambda X_, n_samples, rng: model.sample_output(X_, n_samples, rng)
            bounds = model.output_bounds(X, q_low=self.bounds_q_low, q_high=self.bounds_q_high,
                                         pad_frac=self.bounds_pad_frac, n_samples=min(10000, self.mc_n_samples),
                                         rng=self.mc_random_state)
            predict_dens = self._as_density_collection(model.predict_density(X, y_grid, context='predict'))
            S_p = predict_dens.shape[0]

            result_list = []
            pointwise_density = model.density_function_for_input(X) if hasattr(model, 'density_function_for_input') else None
            for s in range(S_p):
                def p_hat_s(points, input_index=0):
                    if pointwise_density is not None:
                        return np.asarray(pointwise_density(points, input_index=input_index)).reshape(-1)
                    x_i = np.asarray(X)[input_index:input_index + 1]
                    dens = model.predict_density(x_i, np.asarray(points), context='predict')
                    dens = self._as_density_collection(dens)
                    return np.asarray(dens[s, 0]).reshape(-1)

                def p_star_ref(points, input_index=0):
                    if pointwise_density is not None:
                        return np.asarray(pointwise_density(points, input_index=input_index)).reshape(-1)
                    x_i = np.asarray(X)[input_index:input_index + 1]
                    dens = model.predict_density(x_i, np.asarray(points), context='approximate')
                    dens = self._as_density_collection(dens)
                    return np.mean(np.asarray(dens[:, 0]), axis=0).reshape(-1)

                alpha_vol_s = self.alpha_volume(p_hat_s, sampler, bounds, X, alpha=alpha,
                                                n_samples=self.mc_n_samples, random_state=self.mc_random_state,
                                                method='monte_carlo')

                c_alpha_star, _, samples_star, samples_star_mask = self._hdr_from_density_function(
                    p_star_ref, sampler, X, alpha)
                c_alpha_hat, _, samples_hat, sample_hat_mask = self._hdr_from_density_function(
                    p_hat_s, sampler, X, alpha)

                sampler_p_star = self._hdr_resampler_from_samples(samples_star, samples_star_mask)
                sampler_p_hat = self._hdr_resampler_from_samples(samples_hat, sample_hat_mask)

                def p_star_trunc(points, input_index=0):
                    vals = np.asarray(p_star_ref(points, input_index=input_index)).ravel()
                    return np.where(vals >= c_alpha_star[input_index], vals, 0.0)

                def p_hat_trunc(points, input_index=0):
                    vals = np.asarray(p_hat_s(points, input_index=input_index)).ravel()
                    return np.where(vals >= c_alpha_hat[input_index], vals, 0.0)

                tv_s = self.tv_distance_mc(
                    p_star_trunc,
                    p_hat_trunc,
                    sampler_p_star,
                    sampler_p_hat,
                    X,
                    n_samples=self.mc_n_samples,
                    random_state=self.mc_random_state,
                )

                tv_s_safe = np.minimum(tv_s, 1.0 - 1e-10)
                result_list.append(alpha_vol_s / (1.0 - tv_s_safe))

            results = np.stack(result_list, axis=0)
            return np.mean(results, axis=0)
        finally:
            # Post - reset scopes
            self.alpha = alpha_old
            self.scope = scope_old

    def _compute_aleatoric(self, model, X):
        """Compute aleatoric uncertainty as BMA of QUEST measure over predict samples.
        
        Uses predict context with BMA decomposition:
        - Obtains predict densities [S_p, N, G] or [N, G]
        - For each predict sample: computes alpha_volume
        - Averages across all predict samples
        """
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        y_grid = np.asarray(y_grid)

        if y_grid.ndim == 1:
            predict_dens = self._as_density_collection(model.predict_density(X, y_grid, context='predict'))
            predict_mean = np.mean(predict_dens, axis=0)

            def density_func_1d(points, input_index=0):
                points = np.asarray(points).reshape(-1)
                return np.interp(points, y_grid, predict_mean[input_index])

            if self.scope == 'local':
                return self._hdr_from_density_grid_1d(density_func_1d, X, alpha=self.alpha, grid=y_grid)
            if self.scope == 'global':
                alpha_grid = np.linspace(0.01, 0.99, num=self.n_alpha)
                curve = []
                for alpha in alpha_grid:
                    curve.append(self._hdr_from_density_grid_1d(density_func_1d, X, alpha=alpha, grid=y_grid))
                curve = np.stack(curve, axis=0)
                return np.trapz(curve, alpha_grid, axis=0)
            raise ValueError(f"Unknown scope: {self.scope}")

        # Multivariate fallback: preserve existing Monte Carlo behavior unchanged.
        if not hasattr(model, 'sample_output') or not hasattr(model, 'output_bounds'):
            raise NotImplementedError("Model must implement sample_output and output_bounds for Monte Carlo HDR estimation.")

        sampler = lambda X_, n_samples, rng: model.sample_output(X_, n_samples, rng)
        bounds = model.output_bounds(X, q_low=self.bounds_q_low, q_high=self.bounds_q_high,
                                      pad_frac=self.bounds_pad_frac, n_samples=min(10000, self.mc_n_samples),
                                      rng=self.mc_random_state)

        predict_dens = self._as_density_collection(model.predict_density(X, y_grid, context='predict'))
        S_p = predict_dens.shape[0]
        uncert_list = []
        pointwise_density = model.density_function_for_input(X) if hasattr(model, 'density_function_for_input') else None

        for s in range(S_p):
            def density_func_s(points, input_index=0):
                if pointwise_density is not None:
                    return np.asarray(pointwise_density(points, input_index=input_index)).reshape(-1)
                x_i = np.asarray(X)[input_index:input_index + 1]
                dens_vals = model.predict_density(x_i, np.asarray(points), context='predict')
                dens_vals = self._as_density_collection(dens_vals)
                return np.asarray(dens_vals[s, 0]).reshape(-1)

            if self.scope == 'local':
                av_s = self.alpha_volume(density_func_s, sampler, bounds, X,
                                         alpha=self.alpha, n_samples=self.mc_n_samples,
                                         random_state=self.mc_random_state, method='monte_carlo')
                uncert_list.append(av_s)
            elif self.scope == 'global':
                iv_s = self.integrated_volume(density_func_s, sampler, bounds, X,
                                              n_alpha=self.n_alpha, n_samples=self.mc_n_samples,
                                              random_state=self.mc_random_state, method='monte_carlo')
                uncert_list.append(iv_s)

        uncert_s = np.stack(uncert_list, axis=0)
        return np.mean(uncert_s, axis=0)

    def _compute_epistemic(self, model, X):
        # Epistemic uncertainty via second-order (parameter-space) HDR
        if not isinstance(model, Ensemble):
            raise ValueError("Model must be an instance of Ensemble for epistemic uncertainty computation.")
        
        # Get KDE + MC samples for each input from the second-order distribution
        kdes_list, samples_list = model.get_second_order_distribution(X, n_mc_samples=self.mc_n_samples, random_state=self.mc_random_state)
        
        # Build per-input density functions and sampler wrappers
        def density_func_epistemic(points, input_index=0):
            """Evaluate the KDE at arbitrary points for a given input index."""
            kde = kdes_list[input_index]
            points = np.asarray(points)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            return kde(points.T).ravel()
        
        def sampler_epistemic(X_, n_samples, rng):
            """
            Return pre-sampled MC samples from the KDEs.
            
            For epistemic, we use the pre-sampled arrays stored in samples_list.
            Shape of return: (n_samples, N, d)
            """
            rng = np.random.default_rng(rng)
            N = X_.shape[0]
            max_available = samples_list[0].shape[0]
            
            # If n_samples exceeds what's available, resample or cycle
            if n_samples <= max_available:
                # Randomly select n_samples from the pre-sampled pool
                result = []
                for i in range(N):
                    idx = rng.choice(max_available, size=n_samples, replace=False)
                    result.append(samples_list[i][idx])
                return np.stack(result, axis=1)  # (n_samples, N, d)
            else:
                # Resample with replacement from the pre-sampled pool
                result = []
                for i in range(N):
                    idx = rng.choice(max_available, size=n_samples, replace=True)
                    result.append(samples_list[i][idx])
                return np.stack(result, axis=1)  # (n_samples, N, d)
        
        # Build bounds for the second-order parameter space
        # Bounds are per-input based on the pre-sampled MC samples
        bounds_list = []
        for i in range(X.shape[0]):
            mc_samples = samples_list[i]  # (n_mc_samples, n_params-1)
            lows = np.quantile(mc_samples, self.bounds_q_low, axis=0)  # (n_params-1,)
            highs = np.quantile(mc_samples, self.bounds_q_high, axis=0)  # (n_params-1,)
            pad = (highs - lows) * self.bounds_pad_frac
            bounds_i = np.stack([lows - pad, highs + pad], axis=-1)  # (n_params-1, 2)
            bounds_list.append(bounds_i)
        bounds = np.stack(bounds_list, axis=0)  # (N, n_params-1, 2)
        
        uncert_s = []
        if self.scope == 'local':
            av = self.alpha_volume(density_func_epistemic, sampler_epistemic, bounds, X, alpha=self.alpha, n_samples=self.mc_n_samples, random_state=self.mc_random_state)
            uncert_s.append(av)
        elif self.scope == 'global':
            iv = self.integrated_volume(density_func_epistemic, sampler_epistemic, bounds, X, n_alpha=self.n_alpha, n_samples=self.mc_n_samples, random_state=self.mc_random_state)
            uncert_s.append(iv)
        return np.mean(np.stack(uncert_s, axis=0), axis=0)

        #     param_samples = model.sample_full_network_parameters(n_samples)
        #     mdn_param_samples = []
        #     for param_sample in param_samples:
        #         mdn_params = model.mdn_params_for_input_full(x, param_sample)
        #         mdn_param_samples.append(mdn_params.flatten())
        #     mdn_param_samples = np.array(mdn_param_samples)
        #     alpha_volume_val = self.integrated_volume_from_parameter_samples(mdn_param_samples, model=model)
        #     alpha_volume_scores.append(alpha_volume_val)
        # return np.array(alpha_volume_scores)

    def score_total(self, model, X, y_true=None):
        return self._compute_total(model, X)

    def score_aleatoric(self, model, X, y_true=None):
        return self._compute_aleatoric(model, X)

    def score_epistemic(self, model, X, y_true=None):
        return self._compute_epistemic(model, X)

    def score(self, model, X, y_true=None):
        """Dispatch to total/aleatoric/epistemic score by decomposition."""
        # Validate inferential choices upfront
        self._validate_inferential_choices(model)
        
        if self.decomposition == 'total':
            return self._compute_total(model, X)
        if self.decomposition == 'aleatoric':
            return self._compute_aleatoric(model, X)
        if self.decomposition == 'epistemic':
            return self._compute_epistemic(model, X)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def alpha_volume(self, density_func, sampler=None, bounds=None, X=None, alpha=None, n_samples=None, random_state=None, method='auto', grid=None):
        """
        Alpha-volume with selectable HDR method.

        Parameters
        - density_func: callable(points, input_index) -> (M,) densities
        - sampler: callable(X, n_samples, rng) -> (n_samples, N, d) [MC only]
        - bounds: (N, d, 2) per-input axis-aligned box [MC only]
        - X: inputs array shape (N, ...)
        """
        if X is None:
            raise ValueError("X must be provided.")
        alpha_val = self.alpha if alpha is None else alpha
        X = np.asarray(X)
        if method == 'grid':
            return self._hdr_from_density_grid_1d(density_func, X, alpha=alpha_val, grid=grid)
        if method not in {'auto', 'monte_carlo'}:
            raise ValueError("method must be 'auto', 'grid', or 'monte_carlo'.")
        if sampler is None or bounds is None:
            raise ValueError("sampler and bounds are required for Monte Carlo HDR computation.")
        n_samples = int(n_samples or self.mc_n_samples)
        c_alpha, hdr_indicator, samples, sample_mask = self._hdr_from_density_function(
            density_func, sampler, X, alpha=alpha_val, n_samples=n_samples, random_state=random_state
        )
        volumes, fractions = self._lebesgue_measure_hdr_mc(
            density_func, c_alpha, bounds, n_samples=n_samples, random_state=random_state
        )
        return volumes

    def integrated_volume(self, density_func, sampler=None, bounds=None, X=None, n_alpha=100, n_samples=None, random_state=None, method='auto', grid=None):
        if X is None:
            raise ValueError("X must be provided.")
        X = np.asarray(X)
        if method == 'grid':
            alpha_volume_curve = []
            alpha_grid = np.linspace(0, 1, n_alpha)
            for alpha in alpha_grid:
                av = self.alpha_volume(
                    density_func,
                    X=X,
                    alpha=alpha,
                    method='grid',
                    grid=grid,
                )
                alpha_volume_curve.append(av)
            alpha_volume_curve = np.stack(alpha_volume_curve, axis=0)
            meta = integrate.trapezoid(alpha_volume_curve, alpha_grid, axis=0)
            return np.maximum(meta, 0.0)

        if method not in {'auto', 'monte_carlo'}:
            raise ValueError("method must be 'auto', 'grid', or 'monte_carlo'.")
        alpha_volume_curve = []
        alpha_grid = np.linspace(0, 1, n_alpha)
        for alpha in alpha_grid:
            av = self.alpha_volume(density_func, sampler, bounds, X, alpha=alpha, n_samples=n_samples, random_state=random_state, method='monte_carlo')
            alpha_volume_curve.append(av)
        alpha_volume_curve = np.stack(alpha_volume_curve, axis=0)
        meta = integrate.trapezoid(alpha_volume_curve, alpha_grid, axis=0)
        return np.maximum(meta, 0.0)

    # def integrated_volume_from_parameter_samples(self, theta_samples, model=None, n_alpha=100):
    #     """
    #     Compute integrated volume as the area under the curve of alpha_volume(alpha) for alpha in (0,1).
    #     """
    #     theta = np.asarray(theta_samples)
    #     if theta.ndim != 2:
    #         raise ValueError("theta_samples must be shape [S, P]")
    #     if model is None:
    #         raise ValueError("model must be provided for analytic density evaluation.")
    #     log_density = model.mdn_parameter_log_density(theta)
    #     density = np.exp(log_density)
    #     idx = np.argsort(-density)
    #     density_sorted = density[idx]
    #     dv = np.ones_like(density_sorted) / len(density_sorted)
    #     alpha_volume_curve = []
    #     alpha_grid = np.linspace(0, 1, n_alpha)
    #     for alpha in alpha_grid:
    #         target = 1.0 - alpha
    #         idx_thresh = np.argmin(np.cumsum(density_sorted * dv) < target)
    #         threshold = density_sorted[idx_thresh]
    #         mask = density >= threshold
    #         lebesgue_measure = np.sum(mask * dv)
    #         alpha_volume_curve.append(lebesgue_measure)
    #     meta = integrate.trapezoid(alpha_volume_curve, alpha_grid)
    #     return float(np.maximum(meta, 0.0))

    # def integrated_volume(self, model=None, theta_samples=None, n_param_samples=None):
    #     """
    #     Compute integrated volume using parameter samples.
    #     """
    #     if theta_samples is None:
    #         if model is None:
    #             raise ValueError("Provide either `model` or `theta_samples`.")
    #         n = n_param_samples or self.n_param_samples
    #         # For input-dependent, call from score for each input
    #         raise NotImplementedError("Call integrated_volume_from_parameter_samples for each input in score.")
    #     return self.integrated_volume_from_parameter_samples(theta_samples, model)

    def tv_distance_mc(
        self,
        p_star_func,
        p_hat_func,
        sampler_p_star,
        sampler_p_hat,
        X,
        n_samples=100_000,
        random_state=None,
        eps=1e-12
    ):
        rng = np.random.default_rng(random_state)
        X = np.asarray(X)
        N = X.shape[0]

        # Decide mixture allocation
        choose = rng.uniform(size=n_samples) < 0.5
        n_p = int(choose.sum())
        n_q = int(n_samples - n_p)

        # Draw pooled samples from each truncated sampler (may be 0)
        d = None
        sp = sampler_p_star(X, n_p, rng) if n_p > 0 else np.empty((0, N, 0))
        if n_p > 0:
            d = sp.shape[-1]
        sq = sampler_p_hat(X, n_q, rng) if n_q > 0 else np.empty((0, N, d if d is not None else 0))
        if d is None:
            # zero samples from both -> nothing to do
            return np.zeros(N) if N > 1 else 0.0

        # Assemble mixture samples shape (n_samples, N, d)
        mix_samples = np.empty((n_samples, N, d))
        if n_p > 0:
            mix_samples[choose, :, :] = sp
        if n_q > 0:
            mix_samples[~choose, :, :] = sq

        # Compute TV per input
        tvs = np.zeros(N)
        for i in range(N):
            pts = mix_samples[:, i, :]
            # Ensure shape (M, d) or (M,) as accepted by density funcs
            p_vals = np.asarray(p_star_func(pts, input_index=i)).ravel()
            q_vals = np.asarray(p_hat_func(pts, input_index=i)).ravel()

            mix = 0.5 * (p_vals + q_vals)
            # where mix is extremely small, set weight to 0 (both densities ~0)
            denom = np.maximum(mix, eps)
            weights = np.abs(p_vals - q_vals) / denom
            tvs[i] = 0.5 * np.mean(weights)

        return float(tvs[0]) if N == 1 else tvs