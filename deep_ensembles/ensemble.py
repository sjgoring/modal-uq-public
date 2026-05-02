"""
Deep ensemble for regression.

Each component is a feedforward neural network that outputs (mu, log_sigma^2)
for a Gaussian likelihood. The ensemble is trained by independent random
initialization and minimization of NLL on each component.

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
    """A feedforward network outputting (mu, log_sigma^2) for Gaussian likelihood."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 50, n_hidden: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_var), each of shape (batch,)."""
        h = self.backbone(x)
        mu = self.mu_head(h).squeeze(-1)
        # Soft-clamp log_var to a reasonable range to avoid numerical issues
        log_var = self.log_var_head(h).squeeze(-1)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var


def gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under Gaussian with predicted mean and variance.
    
    NLL = 0.5 * log(2*pi) + 0.5 * log_var + 0.5 * (y - mu)^2 / exp(log_var)
    """
    return 0.5 * (np.log(2 * np.pi) + log_var + (y - mu) ** 2 / log_var.exp()).mean()


def train_single_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    hidden_dim: int = 50,
    n_hidden: int = 2,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> GaussianMLP:
    """Train a single Gaussian-output network."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_t = torch.from_numpy(X_train).float()
    y_t = torch.from_numpy(y_train).float()
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GaussianMLP(input_dim, hidden_dim, n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            mu, log_var = model(xb)
            loss = gaussian_nll(mu, log_var, yb)
            loss.backward()
            # Clip gradients for stability with heavy-tailed targets
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: NLL = {epoch_loss / len(dataset):.4f}")
    
    model.eval()
    return model


class DeepEnsemble:
    """A deep ensemble of M Gaussian-output networks."""
    
    def __init__(
        self,
        input_dim: int,
        M: int = 5,
        hidden_dim: int = 50,
        n_hidden: int = 2,
    ):
        self.input_dim = input_dim
        self.M = M
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
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
    ) -> "DeepEnsemble":
        """Train all M components with different random seeds."""
        self.models = []
        for m in range(self.M):
            if verbose:
                print(f"Training ensemble member {m+1}/{self.M}...")
            model = train_single_network(
                X_train, y_train,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                n_hidden=self.n_hidden,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr=lr,
                seed=base_seed + m,
                device=device,
                verbose=False,  # Avoid clutter; show only ensemble-level progress
            )
            self.models.append(model)
        return self
    
    def predict(self, X: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
        """Predict (mus, sigmas) for each input.
        
        Args:
            X: array of shape (n, d).
        
        Returns:
            (mus, sigmas): each of shape (n, M), where M is ensemble size.
        """
        X_t = torch.from_numpy(X).float().to(device)
        all_mus = []
        all_sigmas = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                mu, log_var = model(X_t)
                sigma = (0.5 * log_var).exp()
            all_mus.append(mu.cpu().numpy())
            all_sigmas.append(sigma.cpu().numpy())
        
        # Stack to shape (n, M)
        mus = np.stack(all_mus, axis=1)
        sigmas = np.stack(all_sigmas, axis=1)
        return mus, sigmas
    
    def predictive_distribution(
        self, x: np.ndarray, device: str = "cpu"
    ) -> GaussianMixture1D:
        """Get the predictive Gaussian mixture for a single input.
        
        Args:
            x: array of shape (d,).
        
        Returns:
            GaussianMixture1D representing the M-component predictive mixture.
        """
        x_batch = x[None, :]  # add batch dim
        mus, sigmas = self.predict(x_batch, device=device)
        # mus, sigmas have shape (1, M)
        return GaussianMixture1D(mus=mus[0], sigmas=sigmas[0])
    
    def parameter_samples(
        self, x: np.ndarray, device: str = "cpu"
    ) -> np.ndarray:
        """Get the M ensemble parameter samples (mu, log_sigma) at input x.
        
        Used for QUEST EU computation via 2D KDE on parameter space.
        
        Args:
            x: array of shape (d,).
        
        Returns:
            array of shape (M, 2) with rows [mu_m, log_sigma_m].
        """
        x_batch = x[None, :]
        mus, sigmas = self.predict(x_batch, device=device)
        log_sigmas = np.log(sigmas[0])
        return np.stack([mus[0], log_sigmas], axis=1)


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
    
    ensemble = DeepEnsemble(input_dim=10, M=3, hidden_dim=30, n_hidden=2)
    ensemble.fit(
        X_train, y_train,
        n_epochs=100,  # very small for smoke test
        batch_size=32,
        lr=1e-3,
        verbose=True,
    )
    
    # Check predictions on a few test points
    mus, sigmas = ensemble.predict(X_test[:5])
    true_means = friedman_mean(X_test[:5])
    print("\nPredictions vs truth (first 5 test points):")
    for i in range(5):
        ensemble_mean = mus[i].mean()
        ensemble_std_of_means = mus[i].std()
        ensemble_avg_sigma = sigmas[i].mean()
        print(f"  x_{i}: true_mean={true_means[i]:6.2f}, "
              f"ensemble_mean={ensemble_mean:6.2f}, "
              f"std_of_means={ensemble_std_of_means:.3f}, "
              f"avg_sigma={ensemble_avg_sigma:.3f}, "
              f"y_obs={y_test[i]:6.2f}")
    
    # Verify predictive distribution at a single test point
    pred = ensemble.predictive_distribution(X_test[0])
    print(f"\nPredictive at x_0: mean={pred.mean():.3f}, var={pred.variance():.3f}")
    v_50, _, _ = compute_hdr(pred, alpha=0.5)
    print(f"  V_0.5 = {v_50:.3f}")
    
    # Parameter samples for EU
    theta = ensemble.parameter_samples(X_test[0])
    print(f"\nParameter samples (M={ensemble.M}):")
    print(f"  mus:        {theta[:, 0]}")
    print(f"  log_sigmas: {theta[:, 1]}")
    print(f"  spread of mus: {theta[:, 0].std():.3f}")
    print(f"  spread of log_sigmas: {theta[:, 1].std():.3f}")
