"""
Ensemble of cgmm Mixture-of-Experts regressors.

Each member is a `MixtureOfExpertsRegressor` trained on a bootstrap sample
(or the full data, configurable). The ensemble's predictive distribution
at a query point x is the equally-weighted mixture of the M individual
predictives.

cgmm's MoE uses linear-in-x mean functions per expert, which cannot capture
nonlinearities like sin(pi x) directly. We address this by augmenting the
input with engineered features (sin(pi x) and |x|) before passing to the MoE.
This is transparent to the rest of the pipeline: the wrapper accepts raw x,
augments internally, and returns predictives compatible with our 1D output.

Exposes the interface our QUEST pipeline expects:
  - `fit(X, y)`
  - `predictive_distribution(x) -> GaussianMixture1D`
  - `parameter_samples(x) -> np.ndarray` (M × 2 array of [mu_m, log_sigma_m])

The latter is used only for QUEST's oracle EU computation.
"""

import numpy as np
from cgmm import MixtureOfExpertsRegressor

from .predictive import GaussianMixture1D


def augment_features(X: np.ndarray) -> np.ndarray:
    """Augment 1D input X with engineered features so linear MoE can capture
    nonlinearities in the DGP.
    
    Args:
        X: shape (n,) or (n, 1).
    
    Returns:
        X_aug of shape (n, 3): [x, sin(pi x), |x|].
    """
    if X.ndim == 1:
        x = X
    else:
        x = X[:, 0]
    return np.column_stack([x, np.sin(np.pi * x), np.abs(x)])


class MoEEnsemble:
    def __init__(
        self,
        M: int = 10,
        n_experts: int = 2,
        bootstrap: bool = True,
        max_iter: int = 200,
        reg_covar: float = 1e-4,
    ):
        """
        Args:
            M: number of ensemble members.
            n_experts: experts per member (passed as cgmm n_components).
            bootstrap: if True, each member is trained on a bootstrap sample.
                Otherwise, all members train on the full data and differ only
                via random_state.
            max_iter: cgmm max EM iterations per member.
            reg_covar: covariance regularization. Slightly higher than cgmm
                default for numerical stability with small bootstrap samples.
        """
        self.M = M
        self.n_experts = n_experts
        self.bootstrap = bootstrap
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.models: list[MixtureOfExpertsRegressor] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, base_seed: int = 0) -> "MoEEnsemble":
        """Train M MoE regressors with different random states."""
        n = X.shape[0]
        X_aug = augment_features(X)
        self.models = []
        rng = np.random.default_rng(base_seed)
        
        for m in range(self.M):
            seed_m = base_seed * 1000 + m
            
            if self.bootstrap:
                idx = rng.integers(0, n, size=n)
                X_m, y_m = X_aug[idx], y[idx]
            else:
                X_m, y_m = X_aug, y
            
            model = MixtureOfExpertsRegressor(
                n_components=self.n_experts,
                random_state=seed_m,
                max_iter=self.max_iter,
                reg_covar=self.reg_covar,
            )
            model.fit(X_m, y_m)
            self.models.append(model)
        
        return self
    
    def _conditional_at(self, x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """For each ensemble member, return (means, sigmas, weights) at input x.
        
        Each tuple has shape (n_experts,).
        """
        x_aug = augment_features(np.atleast_2d(x))  # (1, 3)
        out = []
        for model in self.models:
            gmm = model.condition(x_aug)
            means = gmm.means_.ravel()
            sigmas = np.sqrt(gmm.covariances_.ravel())
            weights = gmm.weights_
            out.append((means, sigmas, weights))
        return out
    
    def predictive_distribution(self, x: np.ndarray) -> GaussianMixture1D:
        """Get the predictive Gaussian mixture for input x.
        
        The predictive is bar_p(y|x) = (1/M) sum_m p_m(y|x) where each p_m is
        an n_experts-component Gaussian mixture from member m. Total of
        M * n_experts components.
        
        Component (m, k) gets weight pi_{m,k} / M (so all weights sum to 1).
        """
        per_member = self._conditional_at(x)
        all_means = []
        all_sigmas = []
        all_weights = []
        for means, sigmas, weights in per_member:
            all_means.append(means)
            all_sigmas.append(sigmas)
            all_weights.append(weights / self.M)
        
        flat_means = np.concatenate(all_means)
        flat_sigmas = np.concatenate(all_sigmas)
        flat_weights = np.concatenate(all_weights)
        return GaussianMixture1D(
            mus=flat_means, sigmas=flat_sigmas, weights=flat_weights,
        )
    
    def member_distribution(self, x: np.ndarray, m: int) -> GaussianMixture1D:
        """Get just the m-th ensemble member's predictive at x.
        
        Used for the MLE estimator (where one selected member plays truth approximation).
        """
        means, sigmas, weights = self._conditional_at(x)[m]
        return GaussianMixture1D(mus=means, sigmas=sigmas, weights=weights)
    
    def member_log_likelihood(self, X: np.ndarray, y: np.ndarray, m: int) -> float:
        """Mean log-likelihood of (X, y) under the m-th ensemble member.
        
        Used for MLE selection.
        """
        from scipy.special import logsumexp
        x_aug = augment_features(X)
        gmm = self.models[m].condition(x_aug)
        # gmm.score_samples gives log p(y | X) for each row; need to handle 1D output
        # cgmm's GaussianMixture stores per-sample conditional GMM; iterate.
        # Actually .condition(X_aug) returns a single sklearn.GaussianMixture
        # whose means and covariances depend on X. Let's evaluate log p_m(y | x_i)
        # for each i separately by using cgmm's log_prob if it exists, else manual.
        n = X.shape[0]
        means_per_x, covs_per_x, weights_per_x = [], [], []
        for i in range(n):
            g = self.models[m].condition(x_aug[i:i+1])
            means_per_x.append(g.means_.ravel())
            covs_per_x.append(g.covariances_.ravel())
            weights_per_x.append(g.weights_)
        means_per_x = np.array(means_per_x)        # (n, K)
        covs_per_x = np.array(covs_per_x)
        weights_per_x = np.array(weights_per_x)
        sigmas_per_x = np.sqrt(covs_per_x)
        diff = (y[:, None] - means_per_x) / sigmas_per_x
        log_normal = (
            -0.5 * np.log(2 * np.pi)
            - np.log(sigmas_per_x)
            - 0.5 * diff ** 2
        )
        log_weighted = np.log(np.maximum(weights_per_x, 1e-300)) + log_normal
        return float(logsumexp(log_weighted, axis=1).mean())
    
    def parameter_samples(self, x: np.ndarray) -> np.ndarray:
        """For QUEST oracle EU: return M × 2 array of [mu_m, log_sigma_m].
        
        Each member is summarized by its mixture mean and log-std (where
        sigma is the std of the per-member mixture, accounting for both
        within-component variance and between-component spread).
        """
        per_member = self._conditional_at(x)
        out = np.zeros((self.M, 2))
        for m, (means, sigmas, weights) in enumerate(per_member):
            mu = float((weights * means).sum())
            second = float((weights * (sigmas ** 2 + means ** 2)).sum())
            var = max(second - mu ** 2, 1e-12)
            out[m, 0] = mu
            out[m, 1] = 0.5 * np.log(var)
        return out


# ---------- Sanity check ----------

if __name__ == "__main__":
    from dgp import generate
    
    print("MoE ensemble sanity check")
    print("=" * 50)
    
    X, y = generate(n=500, noise_dist="bimodal", seed=42)
    print(f"Data: X shape {X.shape}, y shape {y.shape}")
    
    ens = MoEEnsemble(M=5, n_experts=3, bootstrap=True)
    ens.fit(X, y, base_seed=0)
    print(f"Trained {ens.M} members with {ens.n_experts} experts each.")
    
    for x_val in [-1.5, 0.0, 1.5]:
        x = np.array([x_val])
        pred = ens.predictive_distribution(x)
        print(f"\nx={x_val:+.1f}: predictive has {pred.M} components")
        print(f"  component means: {pred.mus.round(3)}")
        print(f"  component weights: {pred.weights.round(3)}")
        print(f"  predictive mean = {pred.mean():.3f}, var = {pred.variance():.3f}")
        
        theta = ens.parameter_samples(x)
        print(f"  parameter samples (per-member mu, log_sigma):")
        for m, (mu, log_s) in enumerate(theta):
            print(f"    member {m}: mu={mu:+.3f}, sigma={np.exp(log_s):.3f}")
