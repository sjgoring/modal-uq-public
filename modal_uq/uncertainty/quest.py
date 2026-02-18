
import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','quest')
class QUESTUncertainty(UncertaintyBase):
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
        y_grid = model.default_y_grid(X)
        
        dens_s = model.predict_density_samples(X, y_grid, n_samples=self.n_param_samples)   # [S,N,G]
        dens_mix = dens_s.mean(axis=0)                                   # [N,G]

        threshold, mask = self._hdr_from_density(dens_mix, y_grid, self.alpha)
        lebesgue_measure = self._lebesgue_measure_hdr(mask, y_grid)
        
        # Return the Lebesgue measure of the highest density regions as the uncertainty score.
        if self.decomposition == 'epistemic':
            return np.zeros_like(lebesgue_measure)  # epistemic component is not defined for HDR measure (yet)
        else:
            return lebesgue_measure
