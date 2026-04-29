
import numpy as np
from .base import ModelBase
from ..registry import register, build

@register('model','ensemble')
class Ensemble(ModelBase):
    def __init__(self, base_model='mdn', base_params=None, n_members=5, bootstrap=True, seed=42, inferential_choice=None):
        super().__init__(inferential_choice=inferential_choice)
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
            if self.base_model == 'mdn':
                model.fit(X_m, y_m, X_val, y_val)
            elif self.base_model == 'condgmm':
                model.fit(X_m, y_m)
            else:
                raise NotImplementedError(f"Base model '{self.base_model}' is not supported in Ensemble.")
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
        """Predict density collection using canonical inferential choice.
        
        Parameters
        ----------
        X : array
            Input features
        y_grid : array
            Output grid
        context : {'predict', 'approximate'}, default='predict'
            inferential_choice context
            
        Returns
        -------
        dens : array
            [N,G] for posterior_predictive, [S,N,G] for bma.
        """
        strategy = self.resolve_inferential_choice(context=context)

        # Get member densities [M,N,G]
        member_dens = [m.predict_density(X, y_grid) for m in self.members]
        member_dens = np.stack(member_dens, axis=0)

        if strategy == 'posterior_predictive':
            # For posterior predictive, we weight members by their posterior probabilities. As this is a simple ensemble without explicit Bayesian updating, we can use uniform weights or weights based on member performance. Here, we use uniform weights for simplicity.
            return np.mean(member_dens, axis=0)
        if strategy == 'bma':
            # Pass full densities to functions that can handle them. Example usage: uncertainty measures will be calculated per member and then averaged.
            return member_dens
        raise NotImplementedError(
            f"inferential_choice '{strategy}' is not implemented for {self.__class__.__name__}."
        )

    def default_theta_grid(self, X, num_points=100):
        """Default grid for second-order distribution over parameters."""
        # Note, num_points is per dim.
        # For simplicity, we create a grid over the range of member parameters. This is a placeholder and can be improved with more sophisticated methods.
        if self._y_min is None or self._y_max is None:
            raise ValueError("Model must be fitted before creating default theta grid.")
        
        # check all members, for each parameter obtain the min and max.
        param_mins = []
        param_maxs = []
        for member in self.members:
            if hasattr(member, 'get_params'):
                params = member.get_params(X)  # [n_params, n_X]
                param_mins.append(params.min(axis=1))
                param_maxs.append(params.max(axis=1))
                
                print(params[:params.shape[0]//3,:].sum(axis=0), np.ones(params.shape[1]))
                if not np.isclose(params[:params.shape[0]//3,:].sum(axis=0), np.ones(params.shape[1]), atol=1e-5).all():  
                    raise ValueError("First third of parameters are expected to be weights that sum to 1. Check get_params output.")
                # else:
                #     print("Weight parameters check passed: first third of parameters sum to 1 across members.")
                #     quit()
            else:
                raise NotImplementedError(
                    f"Members of {self.__class__.__name__} do not implement get_params, so default theta grid cannot be created."
                )
        param_mins = np.stack(param_mins, axis=0)  # [n_members, n_params]
        param_maxs = np.stack(param_maxs, axis=0)  # [n_members, n_params]
        overall_mins = param_mins.min(axis=0)  # [n_params]
        overall_maxs = param_maxs.max(axis=0)  # [n_params]
        # Create a grid for each parameter, with broad padding
        grids = []
        for min_val, max_val in zip(overall_mins, overall_maxs):
            pad = 0.1 * (max_val - min_val + 1e-6)
            grid = np.linspace(min_val - pad, max_val + pad, num_points)
            grids.append(grid)
        # Returns a grid of shape [n_params, num_points]
        # Drop first param (e.g. weights) as it is redundant, to avoid rank degeneracy in KDE.
        # Check performed above to ensure first n_members params are weights that sum to 1, so dropping first param should not lose information about the parameter space. This is a common issue in mixture models where one component's weight can be inferred from the others.
        out =   np.stack(grids, axis=0) # shape [n_params, num_points]
        return out[1:, :]  # Drop first param (redundant weights)
        
    def get_second_order_distribution(self, X, theta_grid=None):
        """Provide second-order distribution payload for uncertainty measures."""
        # Note that here we fit a density to the otherwise dirac mixture. Note this is only intended to be used for epistemic uncertainty measurement.
        if theta_grid is None:
            theta_grid = self.default_theta_grid(X)

        if hasattr(self.members[0], 'get_params'):
            params = np.stack([member.get_params(X) for member in self.members], axis=-1)  # [n_params, n_X, n_members]
        else:
            raise NotImplementedError(
                f"Members of {self.__class__.__name__} do not implement get_params, so second-order distribution cannot be constructed."
            )
        # Fit a KDE to each condition parameter distribution across members.
        theta_dens_by_X = []
        for idx in range(X.shape[0]):
            # drop first param, as weights always have 1 redundant dim. can cause rank degeneracy and KDE issues.
            params_at_idx = params[1:, idx, :]  # [n_params-1, n_members]
            # params_at_idx = params[:, idx, :]  # [n_params, n_members]
            # use SciPy Gaussian KDE to estimate joint density over theta_grid.
            print(params_at_idx, params_at_idx.shape)
            # print("Quitting at file: ensemble.py, line 122 - check if get_params is returning expected shapes and values.")
            # quit()
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(params_at_idx)
            # Evaluate KDE on theta_grid to get density values. This will be used as the second-order distribution over parameters.    
            except Exception as e:
                raise RuntimeError(f"Error fitting KDE for sample {idx}: {e}")
            dens = kde(theta_grid)  # [n_params, num_points]
            # normalized_dens = dens / np.trapz(dens, theta_grid, axis=-1, keepdims=True)  # Normalize density
            # stack  this density for each X to get [n_X, n_params, num_points]
            theta_dens_by_X.append(dens) 
        theta_dens_by_X = np.stack(theta_dens_by_X, axis=0)
        return (theta_dens_by_X, theta_grid)
        
        # strategy = self.resolve_inferential_choice(context=context)
        # if strategy == 'posterior_predictive':
        #     dens = self.predict_density(X, y_grid, context=context)
        #     return {'densities': dens[None, ...], 'weights': np.array([1.0])}

        # member_dens = self.predict_density(X, y_grid, context=context)
        # n = member_dens.shape[0]
        # return {'densities': member_dens, 'weights': np.ones(n) / max(n, 1)}

    def get_member_parameters(self):
        """Return member indices as 'parameter samples' for integrated volume."""
        return np.arange(len(self.members)).reshape(-1, 1)