import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','differential_entropy')
class DifferentialEntropy(UncertaintyBase):
    """
    Differential entropy with optional decomposition. Derived from variance.py

    - total:      H[Y | x]
    - aleatoric:  E_θ[ H(Y | x, θ) ]
    - epistemic:  E_θ[ d_KL((Y | x, θ), E_θ(Y | x, θ)) ]

    Uses model.predict_density_samples(X, y_grid, n_samples)-> [S,N,G] if available;
    otherwise falls back to deterministic density [N,G] (epistemic=0).
    """
    def __init__(self, base=np.e, decomposition='total', grid_points=512, y_pad=1.0, n_param_samples=20):
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
        Z = np.trapz(arr, y_grid, axis=-1)     # shape: [N] or [S,N]
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
        H = -np.trapz(integrand, y_grid, axis=-1)
        return H

    def score(self, model, X, y_true=None):
        # Build a default grid per model (shared across X in this batch)
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)

        try:
            # Monte-Carlo parameter samples: [S,N,G]
            dens_s = model.predict_density_samples(X, y_grid, n_samples=self.n_param_samples)
            if dens_s.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")

            H_s = self._entropy_from_density(dens_s, y_grid, self.base)  # [S,N] each

            dens_mix = dens_s.mean(axis=0)                              # [N,G]
            dens_mix = self._normalize_last_axis(dens_mix, y_grid)      # (optional) renormalize
            H_mix = self._entropy_from_density(dens_mix, y_grid, self.base) # [N]

            # Decomposition:
            total     = H_mix # H(Y | x)                 -> [N]
            aleatoric = H_s.mean(axis=0)    # E_theta[H(Y | x, θ)]     -> [N]
            epistemic = total - aleatoric      # E_theta[d_KL(Y|X,theta),E_theta(Y | x, θ)]     -> [N]

        except Exception:
            # Deterministic density fallback: [N,G]   (epistemic = 0)
            dens = model.predict_density(X, y_grid)
            H = self._entropy_from_density(dens, y_grid, self.base)  # [N]
            aleatoric = H
            epistemic = np.zeros_like(H)
            total = aleatoric

        if self.decomposition == 'aleatoric':
            return aleatoric
        elif self.decomposition == 'epistemic':
            return epistemic
        else:
            return total