import numpy as np
from .base import UncertaintyBase
from ..registry import register
import scipy.integrate as integrate

from modal_uq.models.ensemble import Ensemble

@register('uncertainty','alpha_volume')
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
    def __init__(self, alpha=None, decomposition='total', scope='local', grid_points=512, y_pad=1.0, n_param_samples=20, mc_n_samples=100000, mc_random_state=None, bounds_q_low=1e-3, bounds_q_high=1.0-1e-3, bounds_pad_frac=0.05):
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
        n_samples, N, d = samples.shape

        c_alpha = np.zeros(N)
        sample_mask = np.zeros((n_samples, N), dtype=bool)

        for i in range(N):
            pts = samples[:, i, :]
            dens_vals = density_func(pts, input_index=i)
            dens_vals = np.asarray(dens_vals).ravel()
            # sort densities descending
            idx = np.argsort(-dens_vals)
            sorted_dens = dens_vals[idx]
            cutoff_idx = max(0, int(np.ceil((1.0 - alpha) * n_samples)) - 1)
            thresh = sorted_dens[cutoff_idx]
            c_alpha[i] = thresh
            sample_mask[:, i] = dens_vals >= thresh

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
        volumes = np.zeros(N)
        fractions = np.zeros(N)
        for i in range(N):
            lows = bounds[i, :, 0]
            highs = bounds[i, :, 1]
            U = rng.uniform(lows, highs, size=(n_samples, d))
            dens_vals = density_func(U, input_index=i)
            inside = np.asarray(dens_vals).ravel() >= c_alpha[i]
            fraction_inside = np.mean(inside)
            box_vol = float(np.prod(highs - lows))
            volumes[i] = box_vol * fraction_inside
            fractions[i] = fraction_inside
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
        raise NotImplementedError("Total uncertainty for QUEST not yet determined. See output file for values of EU and AU. TU will be a function of these.")
        # y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        # dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        # alpha_volume_s = []
        # for s in range(dens_pred.shape[0]):
        #     _, mask_pred = self._hdr_from_density(dens_pred[s], y_grid, self.alpha)
        #     alpha_volume_s.append(self._lebesgue_measure_hdr(mask_pred, y_grid))
        # return np.mean(np.stack(alpha_volume_s, axis=0), axis=0)

    def _compute_aleatoric(self, model, X):
        # Local scope, do alpha vol, global, do integrated vol
        # take average at end to account for possibility of BMA (post pred will only be one dist, average will not matter here).
        # Monte Carlo HDR-only path: require model to provide sampler, bounds, and density callable
        if not hasattr(model, 'sample_output') or not hasattr(model, 'output_bounds') or not hasattr(model, 'density_function_for_input'):
            raise NotImplementedError("Model must implement sample_output, output_bounds, and density_function_for_input for Monte Carlo HDR estimation.")
        sampler = lambda X_, n_samples, rng: model.sample_output(X_, n_samples, rng)
        bounds = model.output_bounds(X, q_low=self.bounds_q_low, q_high=self.bounds_q_high, pad_frac=self.bounds_pad_frac, n_samples=min(10000, self.mc_n_samples), rng=self.mc_random_state)
        density_func = model.density_function_for_input(X)
        uncert_s = []
        if self.scope == 'local':
            av = self.alpha_volume(density_func, sampler, bounds, X, alpha=self.alpha, n_samples=self.mc_n_samples, random_state=self.mc_random_state)
            uncert_s.append(av)
        elif self.scope == 'global':
            iv = self.integrated_volume(density_func, sampler, bounds, X, n_alpha=100, n_samples=self.mc_n_samples, random_state=self.mc_random_state)
            uncert_s.append(iv)
        return np.mean(np.stack(uncert_s, axis=0), axis=0)

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
            iv = self.integrated_volume(density_func_epistemic, sampler_epistemic, bounds, X, n_alpha=100, n_samples=self.mc_n_samples, random_state=self.mc_random_state)
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
        if self.decomposition == 'total':
            return self._compute_total(model, X)
        if self.decomposition == 'aleatoric':
            return self._compute_aleatoric(model, X)
        if self.decomposition == 'epistemic':
            return self._compute_epistemic(model, X)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def alpha_volume(self, density_func, sampler, bounds, X, alpha=None, n_samples=None, random_state=None):
        """
        Monte Carlo alpha-volume: estimate Lebesgue measure of HDR for each input in X.

        Parameters
        - density_func: callable(points, input_index) -> (M,) densities
        - sampler: callable(X, n_samples, rng) -> (n_samples, N, d)
        - bounds: (N, d, 2) per-input axis-aligned box
        - X: inputs array shape (N, ...)
        """
        alpha_val = self.alpha if alpha is None else alpha
        n_samples = int(n_samples or self.mc_n_samples)
        c_alpha, hdr_indicator, samples, sample_mask = self._hdr_from_density_function(density_func, sampler, X, alpha=alpha_val, n_samples=n_samples, random_state=random_state)
        volumes, fractions = self._lebesgue_measure_hdr_mc(density_func, c_alpha, bounds, n_samples=n_samples, random_state=random_state)
        return volumes

    def integrated_volume(self, density_func, sampler, bounds, X, n_alpha=100, n_samples=None, random_state=None):
        alpha_volume_curve = []
        alpha_grid = np.linspace(0, 1, n_alpha)
        for alpha in alpha_grid:
            av = self.alpha_volume(density_func, sampler, bounds, X, alpha=alpha, n_samples=n_samples, random_state=random_state)
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
