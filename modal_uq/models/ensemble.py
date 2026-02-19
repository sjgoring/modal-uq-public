
import numpy as np
from .base import ModelBase, MarginalizationConfig
from ..registry import register, build

@register('model','ensemble')
class Ensemble(ModelBase):
    def __init__(self, base_model='mdn', base_params=None, n_members=5, bootstrap=True, seed=42, marginalization=None):
        super().__init__(marginalization=marginalization)
        self.base_model = base_model
        self.base_params = base_params or {}
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.seed = seed
        self.members = []
        self._y_min = None; self._y_max = None
        self._member_losses = None  # For selection by criterion

    def fit(self, X, y, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        self.members = []
        N = len(X)
        from copy import deepcopy
        for _ in range(self.n_members):
            model = build('model', self.base_model, **deepcopy(self.base_params))
            if self.bootstrap:
                idx = rng.integers(0, N, size=N)
                X_m, y_m = X[idx], y[idx]
            else:
                X_m, y_m = X, y
            model.fit(X_m, y_m, X_val, y_val)
            self.members.append(model)
        self._y_min = float(y.min()); self._y_max = float(y.max())
        
        # Compute member losses for criterion-based selection
        self._compute_member_losses(X, y)

    def _compute_member_losses(self, X, y, y_grid=None):
        """Compute losses for each member for criterion-based selection."""
        if y_grid is None:
            y_grid = self.default_y_grid(X)
        self._member_losses = []
        for member in self.members:
            dens = member.predict_density(X, y_grid)  # [N, G]
            # Evaluate density at true labels for each sample
            dens_at_y = []
            for i in range(len(y)):
                density_at_i = np.interp(y[i], y_grid, dens[i], left=0, right=0)
                dens_at_y.append(density_at_i)
            dens_at_y = np.array(dens_at_y)
            dens_at_y = np.clip(dens_at_y, 1e-12, None)  # Avoid log(0)
            nll = -np.mean(np.log(dens_at_y))  # Negative log likelihood
            self._member_losses.append(nll)
        self._member_losses = np.array(self._member_losses)
    
    def _select_member_by_criterion(self, criterion):
        """Select member index based on criterion.
        
        Parameters
        ----------
        criterion : str
            Criterion for selection: 'mle' (best by NLL), 'map' (posterior probability)
            
        Returns
        -------
        idx : int
            Index of selected member
        """
        if self._member_losses is None:
            raise RuntimeError("Member losses not computed. Call fit() first.")
        
        if criterion == 'mle':
            return np.argmin(self._member_losses)
        elif criterion == 'map':
            # Convert losses to posterior weights (lower loss = higher weight)
            weights = np.exp(-self._member_losses)
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
        
        # Get member densities
        member_dens = [m.predict_density(X, y_grid) for m in self.members]
        member_dens = np.stack(member_dens, axis=0)  # [M, N, G]
        
        if strategy == 'bma_expected':
            # Average over members
            return np.mean(member_dens, axis=0)  # [N, G]
        
        elif strategy == 'point_estimate':
            # Select single member
            idx = self._select_member_by_criterion(config.point_estimate_criterion)
            return member_dens[idx]  # [N, G]
        
        elif strategy == 'posterior_weighted':
            # Return weighted average (weights based on likelihoods)
            if self._member_losses is None:
                raise RuntimeError("Member losses not computed. Call fit() first.")
            weights = np.exp(-self._member_losses)  # [M]
            weights = weights / weights.sum()  # Normalize
            return np.average(member_dens, axis=0, weights=weights)  # [N, G]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def predict_density_samples(self, X, y_grid, context='predict', n_samples: int = None):
        """Get ensemble member densities as samples.
        
        Parameters
        ----------
        X : array
            Input features
        y_grid : array
            Output grid
        context : {'predict', 'approximate'}, default='predict'
            Marginalization context
        n_samples : int, optional
            Ignored; ensemble has fixed number of members
            
        Returns
        -------
        dens : array of shape [S, N, G]
            Sampled densities where S is number of members
        """
        config = self.get_marginalization_config()
        
        # Select which strategy to use based on context
        strategy = config.predict if context == 'predict' else config.approximate
        
        # Get member densities
        member_dens = [m.predict_density(X, y_grid) for m in self.members]
        member_dens = np.stack(member_dens, axis=0)  # [M, N, G]
        
        if strategy == 'bma_expected':
            # Return averaged density as single sample
            avg_dens = np.mean(member_dens, axis=0)  # [N, G]
            return avg_dens[None, :, :]  # [1, N, G]
        
        elif strategy == 'point_estimate':
            # Return single member density
            idx = self._select_member_by_criterion(config.point_estimate_criterion)
            return member_dens[idx][None, :, :]  # [1, N, G]
        
        elif strategy == 'posterior_weighted':
            # Return individual member densities (unaveraged)
            return member_dens  # [M, N, G]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_member_parameters(self):
        """Return member indices as 'parameter samples' for meta-QUEST."""
        return np.arange(len(self.members)).reshape(-1, 1)