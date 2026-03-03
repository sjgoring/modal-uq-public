import numpy as np
from .base import UncertaintyBase
from ..registry import register

@register('uncertainty','differential_entropy')
class DifferentialEntropy(UncertaintyBase):
    """
    Differential entropy with optional decomposition.

    Uses dual marginalization contexts for uncertainty decomposition:
    
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
        Z = np.trapz(arr, y_grid, axis=-1)
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

        print("Test prints - differential_entropy.py - _entropy_from_density()")
        print(dens.shape, y_grid.shape)
        print(dens[:5,:5], y_grid[:5])
        # quit()

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
        p = p / (np.trapz(p, y_grid) + 1e-12)
        q = q / (np.trapz(q, y_grid) + 1e-12)
        
        # Clip q to avoid log(0)
        q = np.clip(q, 1e-40, None)
        
        # Compute integrand p * log(p / q)
        integrand = p * np.log(p / q)
        integrand = np.where(p > 0, integrand, 0.0)  # Force 0*log(0) = 0
        
        # Integrate
        kl = np.trapz(integrand, y_grid)
        return np.maximum(kl, 0)  # Ensure non-negative

    def score(self, model, X, y_true=None):
        """Compute differential entropy using predict and approximate contexts.
        
        Entropy components (NOT additive decomposition):
        - aleatoric:  Entropy computed from approximate context (true DGP with known params)
        - total:      Entropy computed from predict context (includes parameter uncertainty)
        - epistemic:  NOT DEFINED - decomposition depends on marginalization strategy choice
        
        Note: Epistemic uncertainty cannot be universally defined as (total - aleatoric) because
        the relationship between these quantities depends on the specific marginalization strategies
        chosen for predict vs. approximate contexts. Information-theoretic decomposition semantics
        vary by marginalization approach.
        """
        # Build a default grid per model (shared across X in this batch)
        y_grid = model.default_y_grid(X, grid_points=self.grid_points, y_pad=self.y_pad)

        # print("Test prints - differential_entropy.py - score")
        # print(X.shape, y_grid.shape)
        # print(X[:5], y_grid[:5])

        try:
            # Sample from both prediction and approximation contexts
            dens_pred = model.predict_density_samples(X, y_grid, context='predict', n_samples=self.n_param_samples)  # [S,N,G]
            dens_approx = model.predict_density_samples(X, y_grid, context='approximate', n_samples=self.n_param_samples)  # [S,N,G]
            
            print("differential_entropy.py - score() - test prints")
            print(model.__class__.__name__)
            print(dens_pred.shape, dens_approx.shape, y_grid.shape)
            print(dens_pred[:1, :5, :5], dens_approx[:1, :5, :5])
            # print(np.sum(dens_pred[:1,:1,:,])) # 3.097?

            if dens_pred.ndim != 3 or dens_approx.ndim != 3:
                raise ValueError("predict_density_samples must return [S,N,G]")

            # Compute entropies
            H_pred = self._entropy_from_density(dens_pred, y_grid, self.base)  # [S,N]
            H_approx = self._entropy_from_density(dens_approx, y_grid, self.base)  # [S,N]

            # Entropy components:
            # Aleatoric is expected entropy from approximate context
            aleatoric = H_approx.mean(axis=0)  # E_theta[H(Y | x, θ)] from approximate -> [N]
            
            # Total entropy from predict context
            dens_pred_mix = dens_pred.mean(axis=0)  # [N,G]
            # print(dens_pred_mix.shape)
            # print(np.sum(dens_pred_mix[:1,:])) #3.097

            dens_pred_mix = self._normalize_last_axis(dens_pred_mix, y_grid)
            # print(dens_pred_mix.shape)
            # print(dens_pred_mix[:5,:5])

            total = self._entropy_from_density(dens_pred_mix, y_grid, self.base)  # [N]
            # print(total.shape)
            # print(total[:5])

        except Exception:
            # Deterministic density fallback: [N,G]
            dens_pred = model.predict_density(X, y_grid, context='predict')
            dens_approx = model.predict_density(X, y_grid, context='approximate')
            
            H_pred = self._entropy_from_density(dens_pred, y_grid, self.base)  # [N]
            H_approx = self._entropy_from_density(dens_approx, y_grid, self.base)  # [N]
            
            aleatoric = H_approx
            total = H_pred

        if self.decomposition == 'aleatoric':
            return aleatoric
        elif self.decomposition == 'epistemic':
            # Epistemic entropy: KL divergence between approximate and predict distributions
            # Computation depends on marginalization strategies (number of expectations)
            #
            # Case 1: No expectation (both deterministic)
            #   predict=point/bma, approximate=point/bma → KL(approx || predict)
            # Case 2: 1 expectation (over predict)
            #   predict=posterior_weighted, approximate=point/bma → E_θ_p[KL(approx || predict_θ)]
            # Case 3: 1 expectation (over approximate)
            #   predict=point/bma, approximate=posterior_weighted → E_θ_a[KL(approx_θ || predict)]
            # Case 4: 2 expectations (double integral)
            #   predict=posterior_weighted, approximate=posterior_weighted
            #   → E_θ_p[E_θ_a[KL(approx_θ_a || predict_θ_p)]]
            try:
                S_pred = dens_pred.shape[0]
                S_approx = dens_approx.shape[0]
                
                kl_div = []
                
                if S_pred == 1 and S_approx == 1:
                    # Case 1: No expectation - simple KL divergence
                    for i in range(len(X)):
                        kl = self._kl_divergence(dens_approx[0, i, :], dens_pred[0, i, :], y_grid)
                        kl_div.append(kl)
                
                elif S_pred > 1 and S_approx == 1:
                    # Case 2: 1 expectation over predict samples
                    for i in range(len(X)):
                        kl_samples = []
                        for s in range(S_pred):
                            kl = self._kl_divergence(dens_approx[0, i, :], dens_pred[s, i, :], y_grid)
                            kl_samples.append(kl)
                        kl_div.append(np.mean(kl_samples))
                
                elif S_pred == 1 and S_approx > 1:
                    # Case 3: 1 expectation over approximate samples
                    for i in range(len(X)):
                        kl_samples = []
                        for s in range(S_approx):
                            kl = self._kl_divergence(dens_approx[s, i, :], dens_pred[0, i, :], y_grid)
                            kl_samples.append(kl)
                        kl_div.append(np.mean(kl_samples))
                
                else:  # S_pred > 1 and S_approx > 1
                    # Case 4: 2 expectations (double integral)
                    for i in range(len(X)):
                        kl_pred_samples = []
                        for s_pred in range(S_pred):
                            kl_approx_samples = []
                            for s_approx in range(S_approx):
                                kl = self._kl_divergence(dens_approx[s_approx, i, :], dens_pred[s_pred, i, :], y_grid)
                                kl_approx_samples.append(kl)
                            kl_pred_samples.append(np.mean(kl_approx_samples))
                        kl_div.append(np.mean(kl_pred_samples))
                
                epistemic = np.array(kl_div)
            except NameError:
                # Fallback for deterministic case: no parameter uncertainty
                epistemic = np.zeros(len(X))
            return epistemic
        else:
            return total