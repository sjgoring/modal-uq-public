
import numpy as np
from .base import InferentialChoiceConfig
from .base import ModelBase
from ..registry import register, build
from joblib import Parallel, delayed
from copy import deepcopy

@register('model','ensemble')
class Ensemble(ModelBase):
    def __init__(self, base_model='condgmm', base_params=None, n_members=5, bootstrap=True, seed=42, inferential_choice=None, n_jobs=None):
        super().__init__(inferential_choice=inferential_choice)
        self.base_model = base_model
        self.base_params = base_params or {}
        self.n_members = n_members
        self.bootstrap = bootstrap
        self.seed = seed
        self.members = []
        self._y_min = None; self._y_max = None
        self._member_losses = None  # For selection by criterion
        self.n_jobs = n_jobs  # Number of parallel jobs for compute-heavy operations
        self.cfg = None

        if self.base_model == 'condgmm':
            default_cfg = InferentialChoiceConfig(
                predict='posterior_predictive',
                approximate='point_estimate',
                point_estimate_criterion='mle',
            )
            quest_cfg = InferentialChoiceConfig(
                predict='bma',
                approximate='posterior_predictive',
                point_estimate_criterion='mle',
            )
            cfg = self.get_inferential_choice_config()
            if inferential_choice is None:
                self._inferential_choice_config = default_cfg
            elif (cfg.predict, cfg.approximate, cfg.point_estimate_criterion) not in {
                (default_cfg.predict, default_cfg.approximate, default_cfg.point_estimate_criterion),
                (quest_cfg.predict, quest_cfg.approximate, quest_cfg.point_estimate_criterion),
            }:
                raise NotImplementedError(
                    "Ensemble with base_model='condgmm' currently only supports inferential_choice "
                    "predict='posterior_predictive', approximate='point_estimate', "
                    "point_estimate_criterion='mle' or "
                    "predict='bma', approximate='posterior_predictive', "
                    "point_estimate_criterion='mle'."
                )
            self.cfg = cfg

    def fit(self, X, y, X_val=None, y_val=None):
        print("  Fitting ensemble with {} members...".format(self.n_members))
        rng = np.random.default_rng(self.seed)
        N = len(X)

        # Build one prototype so workers deepcopy it instead of calling build()
        # (loky workers start with an empty registry, so build() would fail there)
        prototype = build('model', self.base_model, **deepcopy(self.base_params))

        # Helper function for training a single member (parallelizable)
        def _train_member(member_idx):
            model = deepcopy(prototype)
            if self.bootstrap:
                # Use fixed RNG seed per member for reproducibility
                member_rng = np.random.default_rng(self.seed + member_idx)
                idx = member_rng.integers(0, N, size=N)
                X_m, y_m = X[idx], y[idx]
            else:
                X_m, y_m = X, y
            if self.base_model == 'mdn':
                model.fit(X_m, y_m, X_val, y_val)
            elif self.base_model == 'condgmm':
                model.fit(X_m, y_m)
            else:
                raise NotImplementedError(f"Base model '{self.base_model}' is not supported in Ensemble.")
            return model

        # Parallelize member training
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        if n_jobs == 1:
            # Serial execution
            self.members = []
            for member_idx in range(self.n_members):
                model = _train_member(member_idx)
                self.members.append(model)
                print("    Member {}/{} fitted".format(member_idx + 1, self.n_members))
        else:
            # Parallel execution — loky (process-based) avoids shared-state deadlocks
            # that occur with backend='threading' when sklearn/BLAS use their own locks
            self.members = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_train_member)(member_idx) for member_idx in range(self.n_members)
            )
            for member_idx in range(self.n_members):
                print("    Member {}/{} fitted".format(member_idx + 1, self.n_members))

        print("  [OK] Ensemble fitting complete")
        self._y_min = float(y.min()); self._y_max = float(y.max())
        
        # Compute member losses for criterion-based selection
        print (" Skipping member loss computation")
        # self._compute_member_losses(X, y)

    def _compute_member_losses(self, X, y, y_grid=None):
        """Compute losses for each member for criterion-based selection."""
        if y_grid is None:
            y_grid = self.default_y_grid(X)

        # Helper function for computing loss for a single member (parallelizable)
        def _compute_loss_for_member(member):
            dens = member.predict_density(X, y_grid)  # [N, G]
            # Evaluate density at true labels for each sample
            dens_at_y = np.array([
                np.interp(y[i], y_grid, dens[i], left=0, right=0)
                for i in range(len(y))
            ])
            dens_at_y = np.clip(dens_at_y, 1e-12, None)  # Avoid log(0)
            nll = -np.mean(np.log(dens_at_y))  # Negative log likelihood
            return nll

        # Parallelize loss computation
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        if n_jobs == 1:
            # Serial execution
            self._member_losses = [_compute_loss_for_member(member) for member in self.members]
        else:
            # Parallel execution
            self._member_losses = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_compute_loss_for_member)(member) for member in self.members
            )
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
        if self._member_losses is None and (self.cfg['predict'] == "point_esimate" or self.cfg['approximate'] == "point_esimate"):
            raise RuntimeError("Member losses not computed. but required for point-estimate inferential choice. Call fit() first.")
        
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

        if strategy == 'point_estimate':
            idx = self._select_member_by_criterion('mle')
            return member_dens[idx]

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
        # Function is deprecated in favor of get_second_order_distribution which uses KDE + MC sampling.
        raise NotImplementedError("default_theta_grid is deprecated. Use get_second_order_distribution for KDE + MC sampling of parameter space instead.")

        # Note, num_points is per dim.
        # For simplicity, we create a grid over the range of member parameters. This is a placeholder and can be improved with more sophisticated methods.
        if self._y_min is None or self._y_max is None:
            raise ValueError("Model must be fitted before creating default theta grid.")
        # warn if grid will be very large
        if num_points ** (self.members[0].get_params(X).shape[0] - 1) > 1e6:
            print(f"Warning: theta grid will have {num_points ** (self.members[0].get_params(X).shape[0] - 1)} points, which may be computationally expensive.")

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
        stacked =   np.stack(grids, axis=0) # shape [n_params, num_points]
        stacked  = stacked[1:, :]  # Drop first param (e.g. weights) to avoid rank degeneracy in KDE.
        # use the marginal grids for each parameter to create a full grid of parameter combinations. This will be used for KDE evaluation in get_second_order_distribution.
        mesh = np.mgrid(stacked)



        mesh = np.meshgrid(*stacked, indexing='ij')  # [n_params-1, num_points, ..., num_points]
        return mesh
        
    def get_second_order_distribution(self, X, n_mc_samples=100000, random_state=None):
        """
        Provide second-order (parameter-space) distribution via KDE + MC sampling.
        
        For each input X[i], learns a Gaussian KDE from empirical member parameters,
        then pre-samples from that KDE to provide both a density callable and MC samples
        for epistemic QUEST computation.
        
        Parameters
        ----------
        X : array
            Input features of shape (N, d_in)
        n_mc_samples : int, default=100000
            Number of Monte Carlo samples to draw from each KDE
        random_state : int or np.random.Generator, optional
            Random state for reproducibility
            
        Returns
        -------
        kdes_list : list of scipy.stats.gaussian_kde
            One fitted KDE per input X[i]
        samples_list : list of arrays
            Pre-sampled arrays from KDE, shape (n_mc_samples, n_params-1) per input
        """
        rng = np.random.default_rng(random_state)
        
        if hasattr(self.members[0], 'get_params'):
            params = np.stack([member.get_params(X) for member in self.members], axis=-1)  # [n_params, n_X, n_members]
        else:
            raise NotImplementedError(
                f"Members of {self.__class__.__name__} do not implement get_params, so second-order distribution cannot be constructed."
            )
        
        kdes_list = []
        samples_list = []
        
        from scipy.stats import gaussian_kde
        from modal_uq.utils.seed import derive_seed
        
        # Use the random_state as base seed (should already be resolved to int by caller)
        base_seed = random_state if isinstance(random_state, int) else 0
        
        for idx in range(X.shape[0]):
            # Drop first param (mixture weights) to avoid rank degeneracy from constraint (weights sum to 1)
            params_at_idx = params[1:, idx, :]  # [n_params-1, n_members]
            
            try:
                kde = gaussian_kde(params_at_idx)
            except Exception as e:
                raise RuntimeError(f"Error fitting KDE for sample {idx}: {e}")
            
            # Pre-sample from KDE for HDR thresholding using per-input derived seed for determinism
            input_seed = derive_seed(base_seed, "kde_sample", idx)
            mc_samples = kde.resample(size=n_mc_samples, seed=input_seed)  # [n_params-1, n_mc_samples]
            mc_samples = mc_samples.T  # -> [n_mc_samples, n_params-1]
            
            kdes_list.append(kde)
            samples_list.append(mc_samples)
        
        return kdes_list, samples_list

    def get_member_parameters(self):
        """Return member indices as 'parameter samples' for integrated volume."""
        return np.arange(len(self.members)).reshape(-1, 1)

    def sample_output(self, X, n_samples, rng):
        """
        Sample outputs from the ensemble predictive distribution.

        Requires that members implement `sample_output`. Returns (n_samples, N, d).
        """
        if len(self.members) == 0:
            raise RuntimeError("Ensemble has no members. Call fit() first.")
        # Check members implement sample_output
        if not all(hasattr(m, 'sample_output') for m in self.members):
            raise NotImplementedError("All ensemble members must implement sample_output for ensemble sampling.")
        rng = np.random.default_rng(rng)
        # Draw member indices for each sample: shape (n_samples,)
        member_idx = rng.integers(0, len(self.members), size=n_samples)
        samples_list = []
        # For efficiency, request n_samples from each member and then select
        # but simplest: call sample_output for each member once with n_samples and select
        member_samples = [m.sample_output(X, n_samples, rng) for m in self.members]
        # member_samples: list of arrays (n_samples, N, d)
        # assemble final samples by selecting per-sample member
        n_samples_local = n_samples
        N = X.shape[0]
        d = member_samples[0].shape[2]
        out = np.zeros((n_samples_local, N, d))
        for s in range(n_samples_local):
            m = member_idx[s]
            out[s] = member_samples[m][s]
        return out

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        """
        Build per-input bounds by aggregating member bounds.

        Returns (N, d, 2).
        """
        # Prefer members providing output_bounds
        member_bounds = []
        for m in self.members:
            if hasattr(m, 'output_bounds'):
                member_bounds.append(m.output_bounds(X, q_low=q_low, q_high=q_high, pad_frac=pad_frac, n_samples=n_samples, rng=rng))
            else:
                raise NotImplementedError("All ensemble members must implement output_bounds to build ensemble bounds.")
        # member_bounds: list of arrays (N, d, 2)
        stacked = np.stack(member_bounds, axis=0)  # (n_members, N, d, 2)
        lows = stacked[:, :, :, 0].min(axis=0)  # (N, d)
        highs = stacked[:, :, :, 1].max(axis=0)  # (N, d)
        return np.stack([lows, highs], axis=-1)

    def density_function_for_input(self, X):
        """
        Return a density function averaging member densities: density_func(points, input_index)
        """
        if not all(hasattr(m, 'density_function_for_input') for m in self.members):
            raise NotImplementedError("All members must implement density_function_for_input to build ensemble density.")
        member_funcs = [m.density_function_for_input(X) for m in self.members]

        def density_func(points, input_index=0):
            # Average densities across members for the given input index
            vals = np.stack([f(points, input_index) for f in member_funcs], axis=0)
            return np.mean(vals, axis=0)

        return density_func