import numpy as np
from .base import UncertaintyBase
from ..registry import register
from scipy.stats import gaussian_kde

@register('uncertainty','quest')
class QUESTUncertainty(UncertaintyBase):
    """
    QUEST uncertainty using Highest Density Regions (HDR).
    
    Computes the Lebesgue measure (total length) of the highest density region
    containing (1 - alpha) probability mass.
    
    Uses dual marginalization contexts for uncertainty decomposition:
    - total:      Lebesgue measure of HDR from predict context
    - aleatoric:  Lebesgue measure of HDR from approximate context (true DGP)
    - epistemic:  TBD - HDR-based measure does not naturally decompose via subtraction
    
    The epistemic component for HDR-based measures requires domain-specific analysis
    and is left as a stub for future implementation.
    """
    def __init__(self, alpha, decomposition='total', grid_points=512, y_pad=1.0, n_param_samples=20):
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
        Z = np.trapz(arr, y_grid, axis=-1)     # shape: [N] or [S,N]
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
        Z = np.trapz(dens, y_grid, axis=-1)[:, None] + 1e-12
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
    
    def score(self, model, X, y_true=None):
        """Compute QUEST uncertainty using predict and approximate contexts.
        
        Returns Lebesgue measure of the HDR. Decomposition:
        - aleatoric: HDR measure from approximate context
        - total: HDR measure from predict context
        - epistemic: NotImplementedError (TBD - domain-specific analysis needed)
        """
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        
        try:
            # Sample densities from both contexts
            dens_pred = model.predict_density_samples(X, y_grid, context='predict', n_samples=self.n_param_samples)   # [S,N,G]
            dens_approx = model.predict_density_samples(X, y_grid, context='approximate', n_samples=self.n_param_samples)   # [S,N,G]
            
            # Compute HDR from both contexts
            threshold_pred, mask_pred = self._hdr_from_density(dens_pred, y_grid, self.alpha)
            threshold_approx, mask_approx = self._hdr_from_density(dens_approx, y_grid, self.alpha)
            
            # Compute Lebesgue measures
            lebesgue_pred = self._lebesgue_measure_hdr(mask_pred, y_grid)   # [N]
            lebesgue_approx = self._lebesgue_measure_hdr(mask_approx, y_grid)  # [N]
            
            aleatoric = lebesgue_approx
            total = lebesgue_pred
            
        except Exception:
            # Deterministic fallback
            dens_pred = model.predict_density(X, y_grid, context='predict')
            dens_approx = model.predict_density(X, y_grid, context='approximate')
            
            threshold_pred, mask_pred = self._hdr_from_density(dens_pred, y_grid, self.alpha)
            threshold_approx, mask_approx = self._hdr_from_density(dens_approx, y_grid, self.alpha)
            
            lebesgue_pred = self._lebesgue_measure_hdr(mask_pred, y_grid)
            lebesgue_approx = self._lebesgue_measure_hdr(mask_approx, y_grid)
            
            aleatoric = lebesgue_approx
            total = lebesgue_pred
        
        # Return depending on decomposition
        if self.decomposition == 'epistemic':
            # Compute meta-QUEST on the parameter posterior as epistemic uncertainty.
            # meta_quest operates on parameter samples (KDE-imputed posterior) and
            # returns a scalar. We return this scalar repeated for each input in X
            # so the shape matches other uncertainty measures ([N], one value per X).
            n_samples = self.n_param_samples
            quest_scores = []
            for x in X:
                param_samples = model.sample_full_network_parameters(n_samples)
                mdn_param_samples = []
                for param_sample in param_samples:
                    mdn_params = model.mdn_params_for_input_full(x, param_sample)
                    mdn_param_samples.append(mdn_params.flatten())
                mdn_param_samples = np.array(mdn_param_samples)
                quest_val = self.meta_quest_from_parameter_samples(mdn_param_samples, model=model)
                quest_scores.append(quest_val)
            return np.array(quest_scores)
            
        elif self.decomposition == 'aleatoric':
            return aleatoric
        else:  # total
            return total

    def meta_quest_from_parameter_samples(self, theta_samples, model=None, n_alpha=100):
        """
        Compute meta-QUEST as the area under the curve of QUEST(alpha) for alpha in (0,1).
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
        quest_curve = []
        alpha_grid = np.linspace(0, 1, n_alpha)
        for alpha in alpha_grid:
            target = 1.0 - alpha
            idx_thresh = np.argmin(np.cumsum(density_sorted * dv) < target)
            threshold = density_sorted[idx_thresh]
            mask = density >= threshold
            lebesgue_measure = np.sum(mask * dv)
            quest_curve.append(lebesgue_measure)
        meta = np.trapz(quest_curve, alpha_grid)
        return float(np.maximum(meta, 0.0))

    def meta_quest(self, model=None, theta_samples=None, n_param_samples=None):
        """
        Compute meta-QUEST using parameter samples.
        """
        if theta_samples is None:
            if model is None:
                raise ValueError("Provide either `model` or `theta_samples`.")
            n = n_param_samples or self.n_param_samples
            # For input-dependent, call from score for each input
            raise NotImplementedError("Call meta_quest_from_parameter_samples for each input in score.")
        return self.meta_quest_from_parameter_samples(theta_samples, model)