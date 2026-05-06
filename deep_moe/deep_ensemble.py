"""
Deep ensemble of K-component Gaussian-mixture-output regression networks.

Each ensemble member is a feedforward neural network whose output head produces
parameters of a K-component Gaussian mixture: K means, K log-variances, and K
mixture-weight logits (passed through softmax). At K=1 the architecture reduces
to a standard single-Gaussian deep ensemble.

The ensemble's predictive distribution at input x is the equally-weighted
mixture of the M individual mixtures, giving an M*K-component Gaussian mixture
overall. This is the BMA-style aggregate used throughout the QUEST pipeline.

Implements the same interface as moe_ensemble.MoEEnsemble:
  - fit(X, y, base_seed)
  - predictive_distribution(x) -> GaussianMixture1D
  - parameter_samples(x) -> ndarray of shape (M, 2) summarizing each member.

Reference: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles", NeurIPS 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predictive import GaussianMixture1D


class GaussianMLP(nn.Module):
    """A feedforward network outputting K-component Gaussian mixture parameters.
    
    For each input x, the network outputs:
      - K means (mu_1, ..., mu_K)
      - K log-variances (log_var_1, ..., log_var_K)
      - K mixture-weight logits (later passed through softmax to get pi_1, ..., pi_K)
    
    With K=1, this reduces to a standard Gaussian-output network.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 50,
        n_hidden: int = 2,
        K: int = 1,
    ):
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K
        
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        
        self.mu_head = nn.Linear(hidden_dim, K)
        self.log_var_head = nn.Linear(hidden_dim, K)
        self.logit_head = nn.Linear(hidden_dim, K) if K > 1 else None
        
        # Initialize the means head's bias to spread components apart slightly.
        # This breaks symmetry so components don't collapse during early training.
        if K > 1:
            with torch.no_grad():
                spread = torch.linspace(-0.5, 0.5, K)
                self.mu_head.bias.copy_(spread)
                # Start with relatively wide variance (sigma ~ 1.28) so that
                # both components have non-negligible support early on. This
                # mitigates the failure mode where one component collapses to
                # a tight fit and the other goes unused.
                self.log_var_head.bias.fill_(0.5)
    
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mus, log_vars, log_pis), each of shape (batch, K).
        
        log_pis are log-mixture-weights (passed through log_softmax for numerical
        stability). For K=1 these are trivially all zero.
        """
        h = self.backbone(x)
        mus = self.mu_head(h)               # (batch, K)
        log_vars = self.log_var_head(h)     # (batch, K)
        log_vars = torch.clamp(log_vars, min=-10.0, max=10.0)
        
        if self.K == 1:
            log_pis = torch.zeros_like(mus)
        else:
            logits = self.logit_head(h)
            log_pis = torch.log_softmax(logits, dim=-1)
        
        return mus, log_vars, log_pis


def gaussian_mixture_nll(
    mus: torch.Tensor,
    log_vars: torch.Tensor,
    log_pis: torch.Tensor,
    y: torch.Tensor,
    entropy_bonus: float = 0.0,
) -> torch.Tensor:
    """Negative log-likelihood under K-component Gaussian mixture per sample.
    
    For each sample i:
        log p(y_i | x_i) = logsumexp_k [log_pi_{i,k}
                                         - 0.5 * log(2*pi)
                                         - 0.5 * log_var_{i,k}
                                         - 0.5 * (y_i - mu_{i,k})^2 / exp(log_var_{i,k})]
    
    Optionally adds an entropy bonus on the mixture weights, which prevents
    component collapse in the K > 1 case. The total loss is:
        loss = mean_i (-log p_i) - entropy_bonus * mean_i H(pi_i)
    where H(pi) = -sum_k pi_k log pi_k. Larger entropy_bonus => stronger
    pressure to keep all components active.
    
    Args:
        mus, log_vars, log_pis: tensors of shape (batch, K).
        y: targets of shape (batch,).
        entropy_bonus: weight on the mixture-weight entropy term (default 0).
    
    Returns:
        scalar mean loss across the batch.
    """
    y_expanded = y.unsqueeze(-1)  # (batch, 1)
    log_normal = (
        -0.5 * np.log(2 * np.pi)
        - 0.5 * log_vars
        - 0.5 * (y_expanded - mus) ** 2 / log_vars.exp()
    )  # (batch, K)
    log_components = log_pis + log_normal
    log_p = torch.logsumexp(log_components, dim=-1)  # (batch,)
    nll = -log_p.mean()
    
    if entropy_bonus > 0.0 and log_pis.shape[-1] > 1:
        pis = log_pis.exp()
        entropy = -(pis * log_pis).sum(dim=-1).mean()  # mean over batch
        return nll - entropy_bonus * entropy
    return nll


def train_single_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    hidden_dim: int = 50,
    n_hidden: int = 2,
    K: int = 1,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
    entropy_bonus: float = 0.0,
) -> GaussianMLP:
    """Train a single Gaussian-mixture-output network.
    
    With K=1, behaves identically to the standard Gaussian-output network.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GaussianMLP(input_dim, hidden_dim, n_hidden, K=K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mus, log_vars, log_pis = model(xb)
            loss = gaussian_mixture_nll(
                mus, log_vars, log_pis, yb, entropy_bonus=entropy_bonus,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: NLL = {epoch_loss / len(dataset):.4f}")
    
    model.eval()
    return model


class DeepEnsemble:
    """A deep ensemble of M Gaussian-mixture-output networks.
    
    Each member outputs a K-component Gaussian mixture; the ensemble's predictive
    is the M*K-component mixture obtained by averaging across members. With K=1
    this reduces to the standard deep ensemble of single-Gaussian outputs.
    """
    
    def __init__(
        self,
        input_dim: int,
        M: int = 5,
        hidden_dim: int = 50,
        n_hidden: int = 2,
        K: int = 1,
    ):
        self.input_dim = input_dim
        self.M = M
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.K = K
        self.models: list[GaussianMLP] = []
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        base_seed: int = 0,
        verbose: bool = True,
        entropy_bonus: float = 0.0,
    ) -> "DeepEnsemble":
        """Train all M components with different random seeds.
        
        Args:
            entropy_bonus: weight on the mixture-weight entropy term in NLL
                (only relevant when K > 1; default 0 disables it).
        """
        self.models = []
        for m in range(self.M):
            if verbose:
                print(f"Training ensemble member {m+1}/{self.M}...")
            model = train_single_network(
                X_train, y_train,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                n_hidden=self.n_hidden,
                K=self.K,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                seed=base_seed + m,
                device=device,
                verbose=False,
                entropy_bonus=entropy_bonus,
            )
            self.models.append(model)
        return self
    
    def predict(
        self, X: np.ndarray, device: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict mixture-component parameters for each input.
        
        Args:
            X: array of shape (n, d).
        
        Returns:
            (mus, sigmas, weights): each of shape (n, M, K).
            mus: per-component means.
            sigmas: per-component standard deviations.
            weights: per-component mixture weights pi_{m,k} (sum to 1 across k for each m).
        """
        X_t = torch.from_numpy(X).float().to(device)
        all_mus = []
        all_sigmas = []
        all_weights = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                mu, log_var, log_pi = model(X_t)  # each (n, K)
                sigma = (0.5 * log_var).exp()
                pi = log_pi.exp()
            all_mus.append(mu.cpu().numpy())
            all_sigmas.append(sigma.cpu().numpy())
            all_weights.append(pi.cpu().numpy())
        
        # Stack to shape (n, M, K)
        mus = np.stack(all_mus, axis=1)
        sigmas = np.stack(all_sigmas, axis=1)
        weights = np.stack(all_weights, axis=1)
        return mus, sigmas, weights
    
    def predictive_distribution(
        self, x: np.ndarray, device: str = "cpu"
    ) -> GaussianMixture1D:
        """Get the predictive Gaussian mixture for a single input.
        
        Returns a GaussianMixture1D with M*K components. Component-(m, k) has
        weight pi_{m, k} / M (each ensemble member contributes equally; within
        a member, the K components are weighted by the network's softmax pi).
        """
        x_batch = x[None, :]
        mus, sigmas, weights = self.predict(x_batch, device=device)
        # Shapes: (1, M, K) -> flatten to (M*K,)
        flat_mus = mus[0].reshape(-1)
        flat_sigmas = sigmas[0].reshape(-1)
        flat_weights = weights[0].reshape(-1) / self.M  # sum_{m,k} = 1
        return GaussianMixture1D(
            mus=flat_mus, sigmas=flat_sigmas, weights=flat_weights,
        )
    
    def parameter_samples(
        self, x: np.ndarray, device: str = "cpu"
    ) -> np.ndarray:
        """Get the M ensemble parameter samples (mu, log_sigma) at input x.
        
        Used for QUEST EU computation in oracle mode (2D KDE on parameter space).
        For K > 1, we summarize each member by its expected mean and log-std under
        its own mixture, since the parameter-space EU framework assumes one
        (mu, sigma) per posterior sample.
        
        Args:
            x: array of shape (d,).
        
        Returns:
            array of shape (M, 2) with rows [mu_m, log_sigma_m].
        """
        x_batch = x[None, :]
        mus, sigmas, weights = self.predict(x_batch, device=device)
        # Per-member mixture mean and variance
        # E[Y | member m] = sum_k pi_{m,k} * mu_{m,k}
        # Var[Y | member m] = sum_k pi_{m,k} * (sigma_{m,k}^2 + mu_{m,k}^2) - mean^2
        member_mus = (mus[0] * weights[0]).sum(axis=-1)         # (M,)
        member_second = (
            (sigmas[0] ** 2 + mus[0] ** 2) * weights[0]
        ).sum(axis=-1)                                          # (M,)
        member_vars = np.maximum(member_second - member_mus ** 2, 1e-12)
        member_log_sigmas = 0.5 * np.log(member_vars)
        return np.stack([member_mus, member_log_sigmas], axis=1)


# ==================== Sanity checks ====================

if __name__ == "__main__":
    """Quick smoke test: train a tiny ensemble on a simple problem and verify
    that the predictive distribution looks reasonable."""
    
    from friedman import generate_friedman, friedman_mean, true_conditional_density
    from predictive import compute_hdr
    
    print("Smoke test: training a tiny ensemble on Friedman with Gaussian noise...")
    print("(Using small n and few epochs for speed; results will be noisy.)")
    
    X_train, y_train = generate_friedman(n=500, noise_dist="gaussian", seed=42)
    X_test, y_test = generate_friedman(n=50, noise_dist="gaussian", seed=43)
    
    ensemble = DeepEnsemble(input_dim=10, M=3, hidden_dim=30, n_hidden=2, K=1)
    ensemble.fit(
        X_train, y_train,
        n_epochs=100,  # very small for smoke test
        batch_size=32,
        lr=1e-3,
        verbose=True,
    )
    
    # Check predictions on a few test points (K=1 case)
    mus, sigmas, weights = ensemble.predict(X_test[:5])
    # mus, sigmas, weights shape (5, M, K). For K=1, squeeze last dim.
    mus_s = mus.squeeze(-1)
    sigmas_s = sigmas.squeeze(-1)
    true_means = friedman_mean(X_test[:5])
    print("\nPredictions vs truth (first 5 test points; K=1):")
    for i in range(5):
        ensemble_mean = mus_s[i].mean()
        ensemble_std_of_means = mus_s[i].std()
        ensemble_avg_sigma = sigmas_s[i].mean()
        print(f"  x_{i}: true_mean={true_means[i]:6.2f}, "
              f"ensemble_mean={ensemble_mean:6.2f}, "
              f"std_of_means={ensemble_std_of_means:.3f}, "
              f"avg_sigma={ensemble_avg_sigma:.3f}, "
              f"y_obs={y_test[i]:6.2f}")
    
    pred = ensemble.predictive_distribution(X_test[0])
    print(f"\nPredictive at x_0 (K=1): mean={pred.mean():.3f}, var={pred.variance():.3f}, "
          f"M_components={pred.M}")
    v_50, _, _ = compute_hdr(pred, alpha=0.5)
    print(f"  V_0.5 = {v_50:.3f}")
    
    theta = ensemble.parameter_samples(X_test[0])
    print(f"  parameter_samples: mus={theta[:, 0]}, log_sigmas={theta[:, 1]}")
    
    # Now train a K=2 ensemble and check predictive structure
    print("\n--- Now training K=2 ensemble ---")
    ensemble2 = DeepEnsemble(input_dim=10, M=3, hidden_dim=30, n_hidden=2, K=2)
    ensemble2.fit(
        X_train, y_train, n_epochs=100, batch_size=32, lr=1e-3, verbose=False,
    )
    pred2 = ensemble2.predictive_distribution(X_test[0])
    print(f"Predictive at x_0 (K=2): mean={pred2.mean():.3f}, var={pred2.variance():.3f}, "
          f"M_components={pred2.M}  (should be M*K = 6)")
    
    # Inspect per-member mixture parameters
    mus2, sigmas2, weights2 = ensemble2.predict(X_test[:1])
    print(f"\nPer-member mixture parameters at x_0 (K=2):")
    for m in range(ensemble2.M):
        for k in range(ensemble2.K):
            print(f"  member {m}, comp {k}: mu={mus2[0, m, k]:.3f}, "
                  f"sigma={sigmas2[0, m, k]:.3f}, pi={weights2[0, m, k]:.3f}")
