import numpy as np
from .base import UncertaintyBase
from ..registry import register
import scipy.integrate as integrate

@register('uncertainty','differential_entropy')
class DifferentialEntropy(UncertaintyBase):
    """
    Differential entropy with optional decomposition.

    Uses dual inferential_choice contexts for uncertainty decomposition:
    
    - total:      H[Y | x] computed from predict context
    - aleatoric:  E_θ[ H(Y | x, θ) ] from approximate context (true DGP with known params)
    - epistemic:  total - aleatoric (difference from approximate to predict)

    The approximate context represents the true data generating process with point-estimate
    parameters (minimal epistemic uncertainty), while predict includes parameter uncertainty.
    
    Requires model.predict_density_samples(X, y_grid, context='predict'|'approximate', n_samples)
    to return [S,N,G] densities. Falls back to deterministic density if unavailable.
    """
    def __init__(self, base=np.e, decomposition='total', grid_points=512, y_pad=1.0, n_param_samples=20):
    # def __init__(self, base=np.e, decomposition='total', grid_points=10000, y_pad=1.0, n_param_samples=20):
        assert decomposition in {'total','aleatoric','epistemic'}
        self.base = base
        self.decomposition = decomposition
        self.grid_points = grid_points
        self.y_pad = y_pad
        self.n_param_samples = n_param_samples

    @staticmethod
    def _normalize_last_axis(arr, y_grid):
        """
        Normalize densities along the last axis (G), regardless of arr being [N,G] or [S,N,G].
        """
        Z = integrate.trapezoid(arr, y_grid, axis=-1)
        Z = np.expand_dims(Z, axis=-1)         # -> [N,1] or [S,N,1]
        return arr / (Z + 1e-12)

    @staticmethod
    def _entropy_from_density(dens, y_grid, base):
        """
        Compute differential entropy H[p] = -∫ p log_base p dy
        for dens of shape [N,G] or [S,N,G].
        Implements the limit p*log(p)=0 when p=0.
        """
        # Normalize
        dens = DifferentialEntropy._normalize_last_axis(dens, y_grid)

        # Safe log: substitute only inside the log, not in dens
        eps = 1e-40
        logp = np.log(dens + eps)
        if base != np.e:
            logp = logp / np.log(base)

        # Compute integrand p * log p
        integrand = dens * logp

        # Force 0·log0 = 0 where dens == 0
        integrand = np.where(dens > 0, integrand, 0.0)

        # Integrate
        H = -integrate.trapezoid(integrand, y_grid, axis=-1)
        return H

    @staticmethod
    def _kl_divergence(p, q, y_grid):
        """
        Compute KL divergence KL(p || q) = ∫ p log(p / q) dy.
        
        Parameters
        ----------
        p : array of shape [G]
            Reference density (not necessarily normalized)
        q : array of shape [G]
            Comparison density (not necessarily normalized)
        y_grid : array of shape [G]
            Grid points
        
        Returns
        -------
        kl : float
            KL(p || q), clipped to be non-negative for numerical stability
        """
        # Normalize both densities
        p = p / (integrate.trapezoid(p, y_grid) + 1e-12)
        q = q / (integrate.trapezoid(q, y_grid) + 1e-12)
        
        # Clip q to avoid log(0)
        q = np.clip(q, 1e-40, None)
        
        # Compute integrand p * log(p / q)
        integrand = p * np.log(p / q)
        integrand = np.where(p > 0, integrand, 0.0)  # Force 0*log(0) = 0
        
        # Integrate
        kl = integrate.trapezoid(integrand, y_grid)
        return np.maximum(kl, 0)  # Ensure non-negative

    def _compute_total(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        try:
            dens_pred = model.predict_density_samples(
                X, y_grid, context='predict', n_samples=self.n_param_samples
            )
            if dens_pred.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")
            dens_pred_mix = self._normalize_last_axis(dens_pred.mean(axis=0), y_grid)
            return self._entropy_from_density(dens_pred_mix, y_grid, self.base)
        except Exception:
            dens_pred = model.predict_density(X, y_grid, context='predict')
            return self._entropy_from_density(dens_pred, y_grid, self.base)

    def _compute_aleatoric(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        try:
            dens_approx = model.predict_density_samples(
                X, y_grid, context='approximate', n_samples=self.n_param_samples
            )
            if dens_approx.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")
            H_approx = self._entropy_from_density(dens_approx, y_grid, self.base)
            return H_approx.mean(axis=0)
        except Exception:
            dens_approx = model.predict_density(X, y_grid, context='approximate')
            return self._entropy_from_density(dens_approx, y_grid, self.base)

    def _compute_epistemic(self, model, X):
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        try:
            dens_pred = model.predict_density_samples(
                X, y_grid, context='predict', n_samples=self.n_param_samples
            )
            dens_approx = model.predict_density_samples(
                X, y_grid, context='approximate', n_samples=self.n_param_samples
            )
            if dens_pred.ndim != 3 or dens_approx.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")
        except Exception:
            dens_pred = model.predict_density(X, y_grid, context='predict')[None, :, :]
            dens_approx = model.predict_density(X, y_grid, context='approximate')[None, :, :]

        S_pred = dens_pred.shape[0]
        S_approx = dens_approx.shape[0]
        kl_div = []

        if S_pred == 1 and S_approx == 1:
            for i in range(len(X)):
                kl = self._kl_divergence(dens_approx[0, i, :], dens_pred[0, i, :], y_grid)
                kl_div.append(kl)
        elif S_pred > 1 and S_approx == 1:
            for i in range(len(X)):
                kl_samples = []
                for s in range(S_pred):
                    kl = self._kl_divergence(dens_approx[0, i, :], dens_pred[s, i, :], y_grid)
                    kl_samples.append(kl)
                kl_div.append(np.mean(kl_samples))
        elif S_pred == 1 and S_approx > 1:
            for i in range(len(X)):
                kl_samples = []
                for s in range(S_approx):
                    kl = self._kl_divergence(dens_approx[s, i, :], dens_pred[0, i, :], y_grid)
                    kl_samples.append(kl)
                kl_div.append(np.mean(kl_samples))
        else:
            for i in range(len(X)):
                kl_pred_samples = []
                for s_pred in range(S_pred):
                    kl_approx_samples = []
                    for s_approx in range(S_approx):
                        kl = self._kl_divergence(dens_approx[s_approx, i, :], dens_pred[s_pred, i, :], y_grid)
                        kl_approx_samples.append(kl)
                    kl_pred_samples.append(np.mean(kl_approx_samples))
                kl_div.append(np.mean(kl_pred_samples))

        return np.array(kl_div)

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