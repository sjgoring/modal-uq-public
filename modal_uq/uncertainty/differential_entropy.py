import numpy as np
import scipy.integrate as integrate

from .base import UncertaintyBase
from ..models.base import InferentialChoiceConfig
from ..registry import register

@register('uncertainty','differential_entropy')
class DifferentialEntropy(UncertaintyBase):
    """
    Differential entropy with optional decomposition.

    Uses dual inferential_choice contexts for uncertainty decomposition:

    - predict: candidate distributions, returned as [S,N,G] or [N,G]
    - approximate: posterior predictive reference, returned as [N,G] or [S,N,G]

    Entropy decompositions: (Corresponding to C2 of Schweighofer et al.)
    - total:      BMA of cross-entropy between predict samples and the approximate reference
    - aleatoric:  BMA of the entropy measure over candidate distributions
    - epistemic:  BMA of KL(predict sample || approximate reference)
    """
    def __init__(self, base=np.e, decomposition='total', grid_points=512, y_pad=1.0, n_param_samples=20, n_jobs=None):
        assert decomposition in {'total','aleatoric','epistemic'}
        self.base = base
        self.decomposition = decomposition
        self.grid_points = grid_points
        self.y_pad = y_pad
        self.n_param_samples = n_param_samples
        self.n_jobs = n_jobs  # Number of parallel jobs for compute-heavy operations

    @staticmethod
    def _normalize_last_axis(arr, y_grid):
        """
        Normalize densities along the last axis (G), regardless of arr being [N,G] or [S,N,G].
        """
        Z = integrate.trapezoid(arr, y_grid, axis=-1)
        Z = np.expand_dims(Z, axis=-1)         # -> [N,1] or [S,N,1]
        return arr / (Z + 1e-12)

    @staticmethod
    def _posterior_predictive_density(dens):
        """Collapse a density collection to its posterior predictive density."""
        dens = np.asarray(dens)
        if dens.ndim == 3:
            return dens.mean(axis=0)
        return dens

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
    def _cross_entropy_from_density(p, q, y_grid, base):
        """
        Compute cross-entropy CE[p || q] = -∫ p log_base q dy.

        Supports p and q with shapes [N,G] or [S,N,G]. When q is a [N,G] reference,
        it broadcasts across the sample axis of p.
        """
        p = DifferentialEntropy._normalize_last_axis(np.asarray(p), y_grid)
        q = DifferentialEntropy._normalize_last_axis(np.asarray(q), y_grid)

        eps = 1e-40
        logq = np.log(q + eps)
        if base != np.e:
            logq = logq / np.log(base)

        integrand = np.where(p > 0, p * logq, 0.0)
        return -integrate.trapezoid(integrand, y_grid, axis=-1)

    @staticmethod
    def _kl_divergence(p, q, y_grid, base=np.e):
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
        
        # If distributions are effectively identical, return 0.0 early
        if np.allclose(p, q, atol=1e-12, rtol=1e-12):
            return 0.0

        # Clip q to avoid log(0)
        q = np.clip(q, 1e-40, None)
        
        # Clip q to avoid log(0); keep p exact so p==0 stays handled below
        q_clip = np.clip(q, 1e-40, None)

        # Compute integrand safely: p * log(p / q_clip), but define 0*log0 := 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = p / q_clip
            integrand = np.where(p > 0, p * np.log(ratio), 0.0)

        kl = integrate.trapezoid(integrand, y_grid)
        kl = np.maximum(kl, 0.0)  # numeric safety

        if base != np.e:
            kl = kl / np.log(base)
        return kl

    @staticmethod
    def _validate_inferential_choices(model):
        cfg = model.get_inferential_choice_config()
        predict = InferentialChoiceConfig.canonicalize_strategy(cfg.predict)
        approximate = InferentialChoiceConfig.canonicalize_strategy(cfg.approximate)

        if predict != 'bma' or approximate != 'posterior_predictive':
            raise NotImplementedError(
                "DifferentialEntropy requires inferential choices predict='bma' and "
                "approximate='posterior_predictive'. "
                f"Current settings: predict='{cfg.predict}', approximate='{cfg.approximate}'."
            )

    def _compute_total(self, model, X, y_grid=None):
        if y_grid is None:
            y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        dens_ref = self._posterior_predictive_density(
            self._predict_density_collection(model, X, y_grid, context='approximate')
        )
        ce = self._cross_entropy_from_density(dens_pred, dens_ref, y_grid, self.base)
        return ce.mean(axis=0)

    def _compute_aleatoric(self, model, X, y_grid=None):
        if y_grid is None:
            y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        H_pred = self._entropy_from_density(dens_pred, y_grid, self.base)
        return H_pred.mean(axis=0)

    def _compute_epistemic(self, model, X, y_grid=None):
        if y_grid is None:
            y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)
        dens_pred = self._predict_density_collection(model, X, y_grid, context='predict')
        dens_ref = self._posterior_predictive_density(
            self._predict_density_collection(model, X, y_grid, context='approximate')
        )
        if dens_pred.ndim == 2:
            dens_pred = dens_pred[None, ...]

        kl_divs = []
        for dens_s in dens_pred:
            kl_per_input = []
            for idx in range(X.shape[0]):
                kl_per_input.append(self._kl_divergence(dens_s[idx, :], dens_ref[idx, :], y_grid, self.base))
            kl_divs.append(kl_per_input)

        return np.mean(np.asarray(kl_divs), axis=0)

    def score_total(self, model, X, y_true=None, y_grid=None):
        return self._compute_total(model, X, y_grid=y_grid)

    def score_aleatoric(self, model, X, y_true=None, y_grid=None):
        return self._compute_aleatoric(model, X, y_grid=y_grid)

    def score_epistemic(self, model, X, y_true=None, y_grid=None):
        return self._compute_epistemic(model, X, y_grid=y_grid)

    def score(self, model, X, y_true=None, y_grid=None):
        self._validate_inferential_choices(model)
        if self.decomposition == 'total':
            return self.score_total(model, X, y_true=y_true, y_grid=y_grid)
        if self.decomposition == 'aleatoric':
            return self.score_aleatoric(model, X, y_true=y_true, y_grid=y_grid)
        if self.decomposition == 'epistemic':
            return self.score_epistemic(model, X, y_true=y_true, y_grid=y_grid)
        raise ValueError(f"Unknown decomposition: {self.decomposition}")