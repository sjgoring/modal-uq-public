
import numpy as np
from .base import ModelBase, MarginalizationConfig
from ..registry import register

# @register('model','mdn_mc_dropout')
class MDN_MCDropout(ModelBase):
    def __init__(self, input_dim=None, hidden_sizes=(64,64), dropout_p=0.1,
                 num_components=5, lr=1e-3, epochs=200, batch_size=128, marginalization=None):
        super().__init__(marginalization=marginalization)
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.dropout_p = dropout_p
        self.num_components = num_components
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self._torch = None; self._net = None
        self._y_min = None; self._y_max = None
        self._dropout_losses = None  # For selection by criterion

    def _ensure_torch(self):
        if self._torch is None:
            import torch, torch.nn as nn
            self._torch = (torch, nn)
        return self._torch

    def _build_net(self, input_dim):
        torch, nn = self._ensure_torch()
        layers = []
        d = input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh(), nn.Dropout(p=self.dropout_p)]
            d = h
        head = nn.Linear(d, 3*self.num_components)
        net = nn.Sequential(*layers, head)
        return net

    def fit(self, X, y, X_val=None, y_val=None):
        torch, nn = self._ensure_torch()
        input_dim = X.shape[1] if self.input_dim is None else self.input_dim
        self._net = self._build_net(input_dim)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1,1)

        def split(out):
            K = self.num_components
            pi = torch.softmax(out[:, :K], dim=-1)
            mu = out[:, K:2*K]
            sigma = torch.exp(out[:, 2*K:3*K]).clamp_min(1e-6)
            return pi, mu, sigma

        for _ in range(self.epochs):
            for i in range(0, len(X_t), self.batch_size):
                xb = X_t[i:i+self.batch_size]; yb = y_t[i:i+self.batch_size]
                out = self._net(xb)
                pi, mu, sigma = split(out)       
                comp = torch.distributions.Normal(mu, sigma)  # mu,sigma: [B,K]
                yb_expanded = yb.expand_as(mu)                # [B,K]
                log_comp = comp.log_prob(yb_expanded)         # [B,K]
                log_prob = torch.logsumexp(torch.log(pi + 1e-12) + log_comp, dim=1)  # [B]
                loss = -log_prob.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self._y_min = float(y.min()); self._y_max = float(y.max())
        
        # Compute dropout sample losses for criterion-based selection
        self._compute_dropout_sample_losses(X, y)

    def _forward_density(self, X, y_grid):
        torch, nn = self._ensure_torch()
        with torch.no_grad():
            self._net.train()  # activate dropout
            X_t = torch.tensor(X, dtype=torch.float32)
            out = self._net(X_t)
            K = self.num_components
            pi = torch.softmax(out[:, :K], dim=-1)
            mu = out[:, K:2*K]
            sigma = torch.exp(out[:, 2*K:3*K]).clamp_min(1e-6)
            Yg = torch.tensor(y_grid, dtype=torch.float32)[None, None, :]
            mu = mu[:, :, None]; sigma = sigma[:, :, None]; pi = pi[:, :, None]
            comp = torch.distributions.Normal(mu, sigma)
            dens = torch.sum(pi * torch.exp(comp.log_prob(Yg)), dim=1)
            return dens.numpy()
    
    def _compute_dropout_sample_losses(self, X, y, y_grid=None, n_samples=20):
        """Compute average losses across dropout samples for criterion-based selection."""
        if y_grid is None:
            y_grid = self.default_y_grid(X)
        
        losses = []
        for _ in range(n_samples):
            dens = self._forward_density(X, y_grid)  # [N, G]
            # Evaluate density at true labels for each sample
            dens_at_y = []
            for i in range(len(y)):
                density_at_i = np.interp(y[i], y_grid, dens[i], left=0, right=0)
                dens_at_y.append(density_at_i)
            dens_at_y = np.array(dens_at_y)
            dens_at_y = np.clip(dens_at_y, 1e-12, None)  # Avoid log(0)
            nll = -np.mean(np.log(dens_at_y))  # Negative log likelihood
            losses.append(nll)
        self._dropout_losses = np.array(losses)  # [n_samples]
    
    def _select_dropout_by_criterion(self, criterion):
        """Select which dropout sample to use based on criterion.
        
        Parameters
        ----------
        criterion : str
            Criterion for selection: 'mle' (best by NLL), 'map' (posterior probability)
            
        Returns
        -------
        idx : int
            Index of selected dropout sample
        """
        if self._dropout_losses is None:
            raise RuntimeError("Dropout sample losses not computed. Call fit() first.")
        
        if criterion == 'mle':
            return np.argmin(self._dropout_losses)
        elif criterion == 'map':
            weights = np.exp(-self._dropout_losses)
            weights = weights / weights.sum()
            return np.argmax(weights)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")


    def predict_density(self, X, y_grid, context='predict'):
        """Predict density using specified marginalization context.
        
        Parameters
        ----------
        X : array
            Input features
        y_grid : array
            Output grid
        context : {'predict', 'approximate'}, default='predict'
            Marginalization context
            
        Returns
        -------
        dens : array of shape [N, G]
            Predicted density
        """
        config = self.get_marginalization_config()
        
        # Select which strategy to use based on context
        strategy = config.predict if context == 'predict' else config.approximate
        
        torch, nn = self._ensure_torch()
        
        if strategy == 'bma_expected':
            # Average multiple dropout samples
            samples = [self._forward_density(X, y_grid) for _ in range(20)]
            return np.mean(np.stack(samples, axis=0), axis=0)  # [N, G]
        
        elif strategy == 'point_estimate':
            # Select single best dropout sample
            if config.point_estimate_criterion == 'mle' or config.point_estimate_criterion == 'map':
                # Use precomputed losses to select
                idx = self._select_dropout_by_criterion(config.point_estimate_criterion)
                # Re-generate that specific sample (approximate)
                samples = [self._forward_density(X, y_grid) for _ in range(len(self._dropout_losses))]
                return samples[idx]  # [N, G]
            else:
                raise ValueError(f"Unknown criterion: {config.point_estimate_criterion}")
        
        elif strategy == 'posterior_weighted':
            # Generate multiple samples and return average with posterior weights
            samples = [self._forward_density(X, y_grid) for _ in range(20)]
            samples = np.stack(samples, axis=0)  # [20, N, G]
            if self._dropout_losses is not None:
                weights = np.exp(-self._dropout_losses[:len(samples)])
                weights = weights / weights.sum()
                return np.average(samples, axis=0, weights=weights)  # [N, G]
            else:
                return np.mean(samples, axis=0)  # [N, G]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def predict_density_samples(self, X, y_grid, context='predict', n_samples: int = 20):
        """Get dropout samples of density.
        
        Parameters
        ----------
        X : array
            Input features
        y_grid : array
            Output grid
        context : {'predict', 'approximate'}, default='predict'
            Marginalization context
        n_samples : int
            Number of dropout samples to draw
            
        Returns
        -------
        dens : array of shape [S, N, G]
            Sampled densities
        """
        config = self.get_marginalization_config()
        
        # Select which strategy to use based on context
        strategy = config.predict if context == 'predict' else config.approximate
        
        if strategy == 'bma_expected':
            # Return averaged density as single sample
            avg_dens = self.predict_density(X, y_grid, context=context)
            return avg_dens[None, :, :]  # [1, N, G]
        
        elif strategy == 'point_estimate':
            # Return single best sample
            idx = self._select_dropout_by_criterion(config.point_estimate_criterion)
            samples = [self._forward_density(X, y_grid) for _ in range(len(self._dropout_losses))]
            return samples[idx][None, :, :]  # [1, N, G]
        
        elif strategy == 'posterior_weighted':
            # Return individual dropout samples (unaveraged)
            samples = [self._forward_density(X, y_grid) for _ in range(n_samples)]
            return np.stack(samples, axis=0)  # [S, N, G]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
