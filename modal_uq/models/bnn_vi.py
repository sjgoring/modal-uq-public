import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base import ModelBase, MarginalizationConfig
from ..registry import register

@register('model','bnn_vi')
class BayesianNNVI(ModelBase):
    """
    Bayesian Neural Network with Variational Inference.
    Mixture Density Network output: predicts GMM parameters (means, variances, weights).
    Supports GPU acceleration and learns multi-modal distribution via mixture components.
    Implements flexible marginalization strategies for prediction and ground truth approximation.
    """
    def __init__(self, input_dim=None, hidden_dims=[64, 64], n_components=3,
                 prior_sigma=1.0, n_mc_samples=20,
                 learning_rate=1e-3, n_epochs=100, batch_size=32, seed=42,
                 marginalization=None, device=None):
        super().__init__(marginalization=marginalization)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_components = n_components
        self.prior_sigma = prior_sigma
        self.n_mc_samples = n_mc_samples
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.marginalization = marginalization
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = None
        self.optimizer = None
        self._y_min = None
        self._y_max = None

    def _build_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim must be set before building model")
        layers = []
        dims = [self.input_dim] + self.hidden_dims
        for i in range(len(dims) - 1):
            layers.append(BayesianLinear(dims[i], dims[i+1], prior_sigma=self.prior_sigma))
            layers.append(nn.ReLU())
        mdn_output_dim = self.n_components * 3
        layers.append(BayesianLinear(dims[-1], mdn_output_dim, prior_sigma=self.prior_sigma))
        return nn.Sequential(*layers)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if self.input_dim is None:
            self.input_dim = X.shape[1]
            print(f"  Auto-detected input_dim: {self.input_dim}")
            self.model = self._build_model().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print(f"BayesianNNVI (MDN) initialized on device: {self.device}")
        print(f"  Input dim: {self.input_dim}, Hidden dims: {self.hidden_dims}")
        print(f"  Number of mixture components: {self.n_components}")
        self._y_min = float(y.min())
        self._y_max = float(y.max())
        N = len(X)
        n_batches = max(1, N // self.batch_size)
        for epoch in range(self.n_epochs):
            idx = np.random.permutation(N)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, N)
                X_batch = torch.from_numpy(X_shuffled[start:end]).to(self.device)
                y_batch = torch.from_numpy(y_shuffled[start:end]).to(self.device)
                self.optimizer.zero_grad()
                loss = self._elbo_loss(X_batch, y_batch, N)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss/n_batches:.4f}")

    def _elbo_loss(self, X_batch, y_batch, N):
        mdn_params = self.model(X_batch)  # [B, n_components*3]
        B = mdn_params.shape[0]
        mdn_params = mdn_params.reshape(B, self.n_components, 3)
        means = mdn_params[:, :, 0]
        log_variances = mdn_params[:, :, 1]
        log_weights = mdn_params[:, :, 2]
        variances = torch.exp(torch.clamp(log_variances, min=-10.0, max=2.0))
        weights = torch.softmax(log_weights, dim=1)
        component_likelihoods = []
        for k in range(self.n_components):
            mu_k = means[:, k:k+1]
            sigma_k = torch.sqrt(variances[:, k:k+1])
            normalized = (y_batch - mu_k) / sigma_k
            exponent = -0.5 * normalized ** 2
            log_prob = exponent - 0.5 * np.log(2 * np.pi) - torch.log(sigma_k)
            component_likelihoods.append(log_prob)
        component_log_probs = torch.cat(component_likelihoods, dim=1)
        log_mixture = torch.logsumexp(
            torch.log(torch.clamp(weights, min=1e-10)) + component_log_probs,
            dim=1
        )
        nll = -torch.sum(log_mixture)
        kl = 0.0
        for layer in self.model:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        kl_scaled = kl / N
        return nll + kl_scaled

    def get_marginalization_config(self):
        if hasattr(self, 'marginalization') and self.marginalization is not None:
            return self.marginalization
        return MarginalizationConfig()

    def predict_density(self, X, y_grid, context='predict'):
        """
        Predict the density over y_grid for each X, using the marginalization strategy.
        """
        config = self.get_marginalization_config()
        strategy = getattr(config, context, 'bma_expected')
        if strategy == 'bma_expected':
            samples = []
            for _ in range(self.n_mc_samples):
                dens = self._predict_density_single(X, y_grid, sample=True)
                samples.append(dens)
            return np.mean(np.stack(samples, axis=0), axis=0)
        elif strategy == 'point_estimate':
            return self._predict_density_single(X, y_grid, sample=False)
        elif strategy == 'posterior_weighted':
            # For diagonal Gaussian, this is usually same as BMA
            samples = []
            for _ in range(self.n_mc_samples):
                dens = self._predict_density_single(X, y_grid, sample=True)
                samples.append(dens)
            return np.mean(np.stack(samples, axis=0), axis=0)
        else:
            raise ValueError(f"Unknown marginalization strategy: {strategy}")

    def _predict_density_single(self, X, y_grid, sample=True):
        X = np.asarray(X, dtype=np.float32)
        y_grid = np.asarray(y_grid, dtype=np.float32)
        N = len(X)
        G = len(y_grid)
        X_torch = torch.from_numpy(X).to(self.device)
        y_grid_torch = torch.from_numpy(y_grid).to(self.device)
        with torch.no_grad():
            if sample:
                mdn_params = self.model(X_torch)
            else:
                # Use mean of variational posterior for each layer
                layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
                for layer in layers:
                    layer.weight_mu.data.copy_(layer.weight_mu.data)
                    layer.bias_mu.data.copy_(layer.bias_mu.data)
                mdn_params = self.model(X_torch)
            mdn_params = mdn_params.reshape(N, self.n_components, 3)
            means = mdn_params[:, :, 0]
            log_variances = mdn_params[:, :, 1]
            log_weights = mdn_params[:, :, 2]
            variances = torch.exp(torch.clamp(log_variances, min=-10.0, max=2.0))
            weights = torch.softmax(log_weights, dim=1)
            means_reshaped = means.unsqueeze(2)
            sigma_reshaped = torch.sqrt(variances).unsqueeze(2)
            y_grid_reshaped = y_grid_torch.view(1, 1, -1)
            exponent = -0.5 * ((y_grid_reshaped - means_reshaped) / sigma_reshaped) ** 2
            component_dens = torch.exp(exponent) / (sigma_reshaped * np.sqrt(2 * np.pi))
            weighted_dens = weights.unsqueeze(2) * component_dens
            mixture_dens = torch.sum(weighted_dens, dim=1)
        return mixture_dens.cpu().numpy()

    def predict_density_samples(self, X, y_grid, context='predict', n_samples=None):
        config = self.get_marginalization_config()
        strategy = getattr(config, context, 'bma_expected')
        if n_samples is None:
            n_samples = self.n_mc_samples
        samples = []
        for _ in range(n_samples):
            dens = self._predict_density_single(X, y_grid, sample=(strategy != 'point_estimate'))
            samples.append(dens)
        return np.array(samples)

    def default_y_grid(self, X, grid_points=512, y_pad=1.0):
        if self._y_min is None or self._y_max is None:
            return np.linspace(-10, 10, grid_points)
        range_y = self._y_max - self._y_min
        y_min = self._y_min - y_pad * range_y
        y_max = self._y_max + y_pad * range_y
        return np.linspace(y_min, y_max, grid_points)

    def sample_mdn_parameters(self, n_samples=100):
        """
        Sample MDN mixture parameters (means, log_variances, log_weights) from the output layer's bias posterior.
        Returns shape: [n_samples, n_components * 3]
        """
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        if not layers:
            raise RuntimeError("No BayesianLinear layers found in model.")
        layer = layers[-1]
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                b = layer.sample_bias().cpu().numpy()
                samples.append(b)
        return np.array(samples)  # shape [n_samples, n_components * 3]

    def mdn_parameter_log_density(self, theta_samples):
        """
        Compute log-density of MDN parameter samples under the learned diagonal Gaussian posterior.
        theta_samples: np.ndarray, shape [n_samples, n_components * 3]
        Returns: np.ndarray, shape [n_samples]
        """
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        if not layers:
            raise RuntimeError("No BayesianLinear layers found in model.")
        layer = layers[-1]
        mu = layer.bias_mu.detach().cpu().numpy()
        std = np.exp(np.clip(layer.bias_log_sigma.detach().cpu().numpy(), -10.0, 2.0))
        var = std ** 2
        dim = mu.size
        theta_samples = np.atleast_2d(theta_samples)
        logp = (
            -0.5 * dim * np.log(2 * np.pi)
            - 0.5 * np.sum(np.log(var))
            - 0.5 * np.sum((theta_samples - mu) ** 2 / var, axis=1)
        )
        return logp

    def predict_mode(self, X, context='predict'):
        """
        Predict the mode (most probable value) for each input X.
        """
        # Use mean of variational posterior for output layer bias as mode
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        if not layers:
            raise RuntimeError("No BayesianLinear layers found in model.")
        layer = layers[-1]
        return layer.bias_mu.detach().cpu().numpy()

    def predict_moments(self, X, context='predict'):

        """
        Predict mean and variance for each input X.
        """
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        if not layers:
            raise RuntimeError("No BayesianLinear layers found in model.")
        layer = layers[-1]
        mean = layer.bias_mu.detach().cpu().numpy()
        std = np.exp(np.clip(layer.bias_log_sigma.detach().cpu().numpy(), -10.0, 2.0))
        var = std ** 2
        return mean, var

    def sample_output_layer_parameters(self, n_samples=100):
        """
        Sample weights and biases from the output layer posterior.
        Returns: list of (weights, bias) tuples, each for one sample.
        """
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        if not layers:
            raise RuntimeError("No BayesianLinear layers found in model.")
        layer = layers[-1]
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                w = layer.sample_weights().cpu().numpy()
                b = layer.sample_bias().cpu().numpy()
                samples.append((w, b))
        return samples  # List of (weights, bias)


    def hidden_representation(self, X):
        """
        Propagate X through all layers except the final BayesianLinear.
        Returns the hidden representation for X.
        """
        X = np.asarray(X, dtype=np.float32).reshape(1, -1)
        x = torch.from_numpy(X).to(self.device)
        for layer in list(self.model)[:-1]:
            x = layer(x)
        return x


    def mdn_params_for_input(self, X, w, b):
        """
        Compute MDN parameters for input X given sampled output layer weights and bias.
        X: shape [input_dim]
        w: shape [output_dim, hidden_dim]
        b: shape [output_dim]
        Returns: [n_components, 3] array
        """
        h = self.hidden_representation(X)  # shape [1, hidden_dim]
        w_torch = torch.from_numpy(w).to(self.device)
        b_torch = torch.from_numpy(b).to(self.device)
        mdn_params = torch.nn.functional.linear(h, w_torch, b_torch)
        mdn_params = mdn_params.reshape(self.n_components, 3)
        return mdn_params.cpu().detach().numpy()

    def mdn_parameter_log_density(self, theta_samples):
        """
        Compute log-density of MDN parameter samples under the learned diagonal Gaussian posterior.
        theta_samples: np.ndarray, shape [n_samples, n_components * 3]
        Returns: np.ndarray, shape [n_samples]
        """
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        if not layers:
            raise RuntimeError("No BayesianLinear layers found in model.")
        layer = layers[-1]
        mu = layer.bias_mu.detach().cpu().numpy()
        std = np.exp(np.clip(layer.bias_log_sigma.detach().cpu().numpy(), -10.0, 2.0))
        var = std ** 2
        dim = mu.size
        theta_samples = np.atleast_2d(theta_samples)
        logp = (
            -0.5 * dim * np.log(2 * np.pi)
            - 0.5 * np.sum(np.log(var))
            - 0.5 * np.sum((theta_samples - mu) ** 2 / var, axis=1)
        )
        return logp
    
    # ...existing code...

    def sample_full_network_parameters(self, n_samples=100):
        """
        Sample weights and biases for all BayesianLinear layers.
        Returns: list of parameter dicts, each with keys 'weights' and 'biases' (list per layer).
        """
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                weights = []
                biases = []
                for layer in layers:
                    w = layer.sample_weights().cpu().numpy()
                    b = layer.sample_bias().cpu().numpy()
                    weights.append(w)
                    biases.append(b)
                samples.append({'weights': weights, 'biases': biases})
        return samples  # List of dicts

    def mdn_params_for_input_full(self, X, param_sample):
        """
        Propagate input X through the network using sampled weights/biases for all layers.
        param_sample: dict with 'weights' and 'biases' (list per layer)
        Returns: [n_components, 3] array
        """
        X = np.asarray(X, dtype=np.float32).reshape(1, -1)
        x = torch.from_numpy(X).to(self.device)
        layers = [layer for layer in self.model if isinstance(layer, BayesianLinear)]
        n_layers = len(layers)
        for i, layer in enumerate(layers):
            w = torch.from_numpy(param_sample['weights'][i]).to(self.device)
            b = torch.from_numpy(param_sample['biases'][i]).to(self.device)
            x = torch.nn.functional.linear(x, w, b)
            if i < n_layers - 1:
                x = torch.relu(x)
        mdn_params = x.reshape(self.n_components, 3)
        return mdn_params.cpu().numpy()

# ...existing code...

# ...existing code...

class BayesianLinear(nn.Module):
    """
    Linear layer with weight distributions (variational inference).
    Implements q(w) = N(w_mu, diag(exp(2*w_log_sigma)))
    Prior   p(w) = N(0, prior_sigma²*I)
    Uses reparameterization trick for gradient-based learning.
    """
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.01)
        self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -5.0))

    def sample_weights(self):
        eps = torch.randn_like(self.weight_mu, device=self.weight_mu.device)
        w = self.weight_mu + torch.exp(torch.clamp(self.weight_log_sigma, min=-10.0, max=2.0)) * eps
        return w.detach()

    def sample_bias(self):
        eps = torch.randn_like(self.bias_mu, device=self.bias_mu.device)
        b = self.bias_mu + torch.exp(torch.clamp(self.bias_log_sigma, min=-10.0, max=2.0)) * eps
        return b.detach()

    def forward(self, x):
        eps_w = torch.randn_like(self.weight_mu, device=x.device)
        eps_b = torch.randn_like(self.bias_mu, device=x.device)
        w_log_sigma_clamped = torch.clamp(self.weight_log_sigma, min=-10.0, max=2.0)
        b_log_sigma_clamped = torch.clamp(self.bias_log_sigma, min=-10.0, max=2.0)
        w = self.weight_mu + torch.exp(w_log_sigma_clamped) * eps_w
        b = self.bias_mu + torch.exp(b_log_sigma_clamped) * eps_b
        return torch.nn.functional.linear(x, w, b)

    def kl_divergence(self):
        w_log_sigma_clamped = torch.clamp(self.weight_log_sigma, min=-10.0, max=2.0)
        b_log_sigma_clamped = torch.clamp(self.bias_log_sigma, min=-10.0, max=2.0)
        sigma_w_sq = torch.exp(2 * w_log_sigma_clamped)
        kl_w = 0.5 * torch.sum(
            np.log(self.prior_sigma ** 2)
            - 2 * w_log_sigma_clamped
            + (sigma_w_sq + self.weight_mu ** 2) / (self.prior_sigma ** 2)
            - 1
        )
        sigma_b_sq = torch.exp(2 * b_log_sigma_clamped)
        kl_b = 0.5 * torch.sum(
            np.log(self.prior_sigma ** 2)
            - 2 * b_log_sigma_clamped
            + (sigma_b_sq + self.bias_mu ** 2) / (self.prior_sigma ** 2)
            - 1
        )
        return kl_w + kl_b
