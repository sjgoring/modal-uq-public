import numpy as np
from cgmm import ConditionalGMMRegressor

from .base import InferentialChoiceConfig
from .base import ModelBase
from ..registry import register
from  collections.abc import Iterable
from sklearn.model_selection import train_test_split
from ..utils.seed import resolve_seed

@register('model','oracle')

class Oracle(ModelBase):
    """Oracle model"""
    def __init__(self, inferential_choice=None, data_set=None, y_pad: float = 1.0, **kwargs):
        super().__init__(inferential_choice)
        default_cfg = InferentialChoiceConfig(
            predict='posterior_predictive',
            approximate='point_estimate',
            point_estimate_criterion='mle',
        )
        bma_cfg = InferentialChoiceConfig(
            predict='bma',
            approximate='posterior_predictive',
            point_estimate_criterion='mle',
        )
        cfg = self.get_inferential_choice_config()
        if inferential_choice is None:
            self._inferential_choice_config = default_cfg
        elif (cfg.predict, cfg.approximate, cfg.point_estimate_criterion) not in {
            (default_cfg.predict, default_cfg.approximate, default_cfg.point_estimate_criterion),
            (bma_cfg.predict, bma_cfg.approximate, bma_cfg.point_estimate_criterion),
        }:
            raise NotImplementedError(
                "Oracle currently only supports inferential_choice "
                "predict='posterior_predictive', approximate='point_estimate', "
                "point_estimate_criterion='mle' or "
                "predict='bma', approximate='posterior_predictive', "
                "point_estimate_criterion='mle'."
            )
        self._y_min = None
        self._y_max = None
        self.data_set = data_set
        self.y_grid = None
        self.y_pad = y_pad
        self.X_raw, self.y_raw = None, None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

        if self.data_set == None:
            raise ValueError("Oracle model requires a dataset object")

    def fit(self, X, y):
        # Don't require fitting. Will pass results straight from the DS.
        # Does Tom get some values from fit? Check this.
        self._y_min = self.data_set.y_min
        self._y_max = self.data_set.y_max
        pass

    def _compute_member_losses(self, X, y, y_grid=None):
        raise NotImplementedError("Oracle does not implement _compute_member_losses, as it does not have members")

    def predict_density(self, X, y, context='predict'):
        strategy = self.resolve_inferential_choice(context=context)
        if strategy not in {'bma', 'posterior_predictive'}:
            raise NotImplementedError(f"Oracle does not support strategy {strategy}")
        # Minimal compatibility layer: some datasets (MPE) expect empirical
        # trajectory samples as the second argument to `gt_dens`, whereas the
        # public model API passes a canonical `y_grid` here. To avoid changing
        # global grid plumbing, special-case MPE to supply its stored empirical
        # samples while keeping the caller-provided `y` as the evaluation grid.
        dens = None
        try:
            # import locally to avoid potential circular imports
            from ..datasets.mpe import MpeDataset
        except Exception:
            MpeDataset = None

        if MpeDataset is not None and isinstance(self.data_set, MpeDataset):
            # Prefer dataset's cached per-row trajectory samples (test split if available).
            # Avoid using `or` on NumPy arrays (ambiguous truth value). Explicitly
            # check attributes for None instead.
            empirical_y = None
            if hasattr(self.data_set, 'y_test') and getattr(self.data_set, 'y_test') is not None:
                empirical_y = self.data_set.y_test
            elif hasattr(self.data_set, 'y_raw') and getattr(self.data_set, 'y_raw') is not None:
                empirical_y = self.data_set.y_raw
            # If empirical samples are available, call gt_dens with them and the
            # caller-provided `y` as the evaluation grid.
            if empirical_y is not None:
                dens = self.data_set.gt_dens(X, empirical_y, y)

        if dens is None:
            # Default behaviour: pass through (for grid-native datasets)
            dens = self.data_set.gt_dens(X, y)

        if strategy == 'bma':
            # Exactly repeat the density for each X, to provide expect BMA behaviour whilst maintaining Oracle knowledge.
            n_estimates = 5
            if isinstance(dens, Iterable):
                dens = np.stack([dens] * n_estimates, axis=0)  # (n_estimates, N, y_grid_size)
            else:
                dens = np.repeat(dens[None, :, :], n_estimates, axis=0)  # (n_estimates, N, y_grid_size)
        return dens

    def get_params(self, X):
        raise NotImplementedError("Oracle does not implement get_params, as the 2nd order distribution is a degenerate Dirac")
    
    def default_theta_grid():
        raise NotImplementedError("Oracle does not implement default_theta_grid, as it does not have members")
        # Indeed, do we even need theta grid at all for any model.
        # TODO: check if 1D HDR by grid is only for y_grid or if it can also take theta_grid.

    def get_second_order_distribution(self, X, y_grid, context='predict'):
        raise NotImplementedError("Oracle does not implement get_second_order_distribution, as the 2nd order distribution is a degenerate Dirac")
    
    def get_member_parameters(self):
        # TODO: Deprecated method?
        raise NotImplementedError("Oracle does not implement get_member_parameters, as it does not have members")
    
    def sample_output(self, X, n_samples, rng):
        # In the Oracle case, sampling from the predictive distribution is just sampling from the GT density.   
        # Use the dataset's cached y_grid where available to avoid callers needing
        # to supply it and to preserve existing grid caches.
        y_grid = getattr(self.data_set, 'y_grid', None)
        if y_grid is None:
            raise ValueError("Oracle.sample_output requires the dataset to provide a y_grid")

        dens = self.predict_density(X, y_grid)
        # Collapse BMA stack if present
        if dens.ndim == 3:
            dens = dens.mean(axis=0)
        N = X.shape[0]
        # Ensure densities are normalized along the grid axis
        try:
            dens = dens / (dens.sum(axis=1, keepdims=True) + 1e-12)
        except Exception:
            pass

        # Build samples shape (n_samples, N, d) where d==1 for scalar outputs
        samples = np.zeros((n_samples, N, 1))
        for i in range(N):
            probs = dens[i]
            # rng.choice expects 1D probs summing to 1
            draws = rng.choice(y_grid, size=n_samples, p=probs)
            samples[:, i, 0] = draws
        return samples
    
    def output_bounds(self, X, q_low=0.001, q_high=1 - 0.001, pad_frac=0.05, n_samples=10000, rng=None):
        # Copied from ensemble, but sampling from the Oracle predictive density instead of an ensemble of members.
        rng = np.random.default_rng(rng)
        samples = self.sample_output(X, n_samples, rng)
        # samples shape: (n_samples, N, d)
        lows = np.quantile(samples, q_low, axis=0)  # (N, d)
        highs = np.quantile(samples, q_high, axis=0)  # (N, d)
        pad = (highs - lows) * pad_frac
        bounds = np.stack([lows - pad, highs + pad], axis=-1)  # (N, d, 2)
        return bounds
    
    def density_function_for_input(self, X):
        """
        Return a density callable: density_func(points, input_index)

        `points` shape: (M, d)
        Returns: (M,) densities for the specified input_index.
        """
        # Compute densities on the dataset's canonical grid to avoid requiring
        # callers to pass the grid through. This preserves cached grids.
        y_grid = getattr(self.data_set, 'y_grid', None)
        if y_grid is None:
            raise ValueError("density_function_for_input requires the dataset to provide a y_grid")

        dens = self.predict_density(X, y_grid)
        if dens.ndim == 3:
            dens = dens.mean(axis=0)

        def density_func(points, input_index):
            if not np.array_equal(points, y_grid):
                raise ValueError("density_function_for_input only supports points equal to the dataset's y_grid")
            predictive_dens = dens[input_index]  # (y_grid_size,)
            return predictive_dens
        return density_func
    