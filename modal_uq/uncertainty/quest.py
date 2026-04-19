import numpy as np
from .base import UncertaintyBase
from ..registry import register
import scipy.integrate as integrate

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
    def __init__(self, alpha, decomposition='total', grid_points=512, y_pad=1.0, n_param_samples=20):
    # def __init__(self, alpha, decomposition='total', grid_points=10000, y_pad=1.0, n_param_samples=20):
        assert decomposition in {'total','aleatoric','epistemic'}
        self.alpha = alpha
        self.decomposition = decomposition
        self.grid_points = grid_points
        self.y_pad = y_pad
        self.n_param_samples = n_param_samples

    @staticmethod
    def _normalize_last_axis(arr, y_grid):
        """
        Normalize densities along the last axis (G), regardless of arr being [N,G] or [S,N,G].
        """
        Z = integrate.trapezoid(arr, y_grid, axis=-1)     # shape: [N] or [S,N]
        Z = np.expand_dims(Z, axis=-1)         # -> [N,1] or [S,N,1]
        return arr / (Z + 1e-12)

    @staticmethod
    def _lebesgue_measure_hdr(mask, y_grid):
        """
        Compute the Lebesgue measure (total length) of the Highest Density Regions.

        Parameters
        ----------
        mask : bool array of shape [N, G]
            Boolean mask where True indicates points in the HDR.
        y_grid : array of shape [G]
            Grid over which density is defined.

        Returns
        -------
        lebesgue_measure : array of shape [N]
            Total length (Lebesgue measure) of the HDR for each sample.
        """
        y_grid = np.asarray(y_grid)
        dy = np.diff(y_grid)
        dy = np.concatenate([dy, [dy[-1]]])
        lebesgue_measure = np.sum(mask * dy, axis=1)
        return lebesgue_measure

    @staticmethod
    def _hdr_from_density(dens, y_grid, alpha):
        """
        Compute the Highest Density Region (HDR) containing (1 - alpha) probability,
        following Hyndman (1996) "Computing and Graphing Highest Density Regions".

        Parameters
        ----------
        dens : array
            Normalized densities over y_grid
            Shape can be [G], [N,G], or [S,N,G].
            If [S,N,G] we compute HDR per [N,G] mixture density.
        y_grid : array of shape [G]
            Grid over which density is defined.
        alpha : float
            Tail probability, i.e. HDR contains probability mass (1 - alpha).

        Returns
        -------
        threshold : array of shape [N]
            Density threshold t_alpha such that HDR = { y : p(y) >= t_alpha }.
        mask : bool array of shape [N, G]
            True where y_grid belongs to the HDR.
        """

        # Ensure last axis is grid axis
        dens = np.asarray(dens)

        # If dens is [S,N,G], average over S to get mixture density
        if dens.ndim == 3:
            dens = dens.mean(axis=0)   # -> [N,G]
        elif dens.ndim == 1:
            dens = dens[None, :]       # -> [1,G]

        # At this point:
        # dens shape is [N,G]
        N, G = dens.shape
        y_grid = np.asarray(y_grid)

        # Normalize densities along grid axis
        Z = integrate.trapezoid(dens, y_grid, axis=-1)[:, None] + 1e-12
        p = dens / Z  # now integrates to 1

        # Sort densities descending along grid axis
        # idx_sorted: [N,G]
        idx_sorted = np.argsort(-p, axis=1)
        p_sorted = np.take_along_axis(p, idx_sorted, axis=1)

        # Cumulative integral of sorted densities
        # Note: spacing varies across grid, so trapz can't be used directly.
        # Approximate cumulative mass via cumulative sum of p_sorted * delta_y.
        # Compute delta_y for each grid point:
        dy = np.diff(y_grid)
        # pad last value so shapes broadcast: [G]
        dy = np.concatenate([dy, [dy[-1]]])

        # cumulative mass: [N,G]
        cum_mass = np.cumsum(p_sorted * dy, axis=1)

        # Find minimal j where cum_mass >= (1 - alpha)
        target = 1.0 - alpha
        # idx_thresh[n] = threshold index for density of sample n
        idx_thresh = np.argmin(cum_mass < target, axis=1)  # [N]

        # Extract threshold values t_alpha[n] = p_sorted[n, idx_thresh[n]]
        threshold = p_sorted[np.arange(N), idx_thresh]

        # Build HDR mask: p[n,g] >= threshold[n]
        mask = p >= threshold[:, None]

        return threshold, mask
    
    def _compute_total(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        alpha_volume_s = []
        for s in range(dens_pred.shape[0]):
            _, mask_pred = self._hdr_from_density(dens_pred[s], y_grid, self.alpha)
            alpha_volume_s.append(self._lebesgue_measure_hdr(mask_pred, y_grid))
        return np.mean(np.stack(alpha_volume_s, axis=0), axis=0)

    def _compute_aleatoric(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_approx = self._predict_density_collection(model, X, y_grid, context='approximate')
        alpha_volume_s = []
        for s in range(dens_approx.shape[0]):
            _, mask_approx = self._hdr_from_density(dens_approx[s], y_grid, self.alpha)
            alpha_volume_s.append(self._lebesgue_measure_hdr(mask_approx, y_grid))
        return np.mean(np.stack(alpha_volume_s, axis=0), axis=0)

    def _compute_epistemic(self, model, X):
        # Integrated volume on parameter posterior per input.
        n_samples = self.n_param_samples
        alpha_volume_scores = []
        for x in X:
            param_samples = model.sample_full_network_parameters(n_samples)
            mdn_param_samples = []
            for param_sample in param_samples:
                mdn_params = model.mdn_params_for_input_full(x, param_sample)
                mdn_param_samples.append(mdn_params.flatten())
            mdn_param_samples = np.array(mdn_param_samples)
            alpha_volume_val = self.integrated_volume_from_parameter_samples(mdn_param_samples, model=model)
            alpha_volume_scores.append(alpha_volume_val)
        return np.array(alpha_volume_scores)

    def score_total(self, model, X, y_true=None):
        return self._compute_total(model, X)

    def score_aleatoric(self, model, X, y_true=None):
        return self._compute_aleatoric(model, X)

    def score_epistemic(self, model, X, y_true=None):
        return self._compute_epistemic(model, X)

    def score(self, model, X, y_true=None):
        """Dispatch to total/aleatoric/epistemic score by decomposition."""
        if self.decomposition == 'total':
            return self.score_total(model, X, y_true=y_true)
        if self.decomposition == 'aleatoric':
            return self.score_aleatoric(model, X, y_true=y_true)
        if self.decomposition == 'epistemic':
            return self.score_epistemic(model, X, y_true=y_true)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")

    def integrated_volume_from_parameter_samples(self, theta_samples, model=None, n_alpha=100):
        """
        Compute integrated volume as the area under the curve of alpha_volume(alpha) for alpha in (0,1).
        """
        theta = np.asarray(theta_samples)
        if theta.ndim != 2:
            raise ValueError("theta_samples must be shape [S, P]")
        if model is None:
            raise ValueError("model must be provided for analytic density evaluation.")
        log_density = model.mdn_parameter_log_density(theta)
        density = np.exp(log_density)
        idx = np.argsort(-density)
        density_sorted = density[idx]
        dv = np.ones_like(density_sorted) / len(density_sorted)
        alpha_volume_curve = []
        alpha_grid = np.linspace(0, 1, n_alpha)
        for alpha in alpha_grid:
            target = 1.0 - alpha
            idx_thresh = np.argmin(np.cumsum(density_sorted * dv) < target)
            threshold = density_sorted[idx_thresh]
            mask = density >= threshold
            lebesgue_measure = np.sum(mask * dv)
            alpha_volume_curve.append(lebesgue_measure)
        meta = integrate.trapezoid(alpha_volume_curve, alpha_grid)
        return float(np.maximum(meta, 0.0))

    def integrated_volume(self, model=None, theta_samples=None, n_param_samples=None):
        """
        Compute integrated volume using parameter samples.
        """
        if theta_samples is None:
            if model is None:
                raise ValueError("Provide either `model` or `theta_samples`.")
            n = n_param_samples or self.n_param_samples
            # For input-dependent, call from score for each input
            raise NotImplementedError("Call integrated_volume_from_parameter_samples for each input in score.")
        return self.integrated_volume_from_parameter_samples(theta_samples, model)
