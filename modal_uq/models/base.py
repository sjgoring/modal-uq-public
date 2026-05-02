
from abc import ABC, abstractmethod
import numpy as np
import scipy.integrate as integrate
import warnings


class InferentialChoiceConfig:
    """Configuration for inferential_choice strategies in stochastic models.
    
    Parameters
    ----------
    predict : str
        inferential_choice strategy for prediction: 'posterior_predictive' or 'bma'.
        Legacy aliases ('bma_expected', 'posterior_weighted', 'point_estimate')
        are accepted for compatibility.
    approximate : str
        inferential_choice strategy for approximation: 'posterior_predictive' or 'bma'.
        Legacy aliases ('bma_expected', 'posterior_weighted', 'point_estimate')
        are accepted for compatibility.
    point_estimate_criterion : str
        Criterion for selecting parameters when strategy is 'point_estimate':
        'mle' (maximum likelihood, default), 'map', 'mean', or 'median'.
    """
    
    VALID_STRATEGIES = {'posterior_predictive', 'bma', 'point_estimate', 'bma_expected', 'posterior_weighted'}
    CANONICAL_STRATEGIES = {'posterior_predictive', 'bma'}
    STRATEGY_ALIASES = {
        'bma_expected': 'bma',
        'posterior_weighted': 'bma',
    }
    VALID_CRITERIA = {'mle', 'map', 'mean', 'median'}
    
    def __init__(self, predict: str = 'posterior_predictive', approximate: str = 'posterior_predictive', 
                 point_estimate_criterion: str = 'mle'):
        # Validate strategies
        if predict not in self.VALID_STRATEGIES:
            raise ValueError(f"predict must be one of {self.VALID_STRATEGIES}, got {predict}")
        if approximate not in self.VALID_STRATEGIES:
            raise ValueError(f"approximate must be one of {self.VALID_STRATEGIES}, got {approximate}")
        
        # Validate point_estimate_criterion
        if point_estimate_criterion not in self.VALID_CRITERIA:
            raise ValueError(f"point_estimate_criterion must be one of {self.VALID_CRITERIA}, got {point_estimate_criterion}")
        
        self.predict = predict
        self.approximate = approximate
        self.point_estimate_criterion = point_estimate_criterion

    @staticmethod
    def canonicalize_strategy(strategy: str) -> str:
        """Map legacy strategy labels to canonical values."""
        if strategy in InferentialChoiceConfig.STRATEGY_ALIASES:
            return InferentialChoiceConfig.STRATEGY_ALIASES[strategy]
        return strategy
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary (e.g., from JSON config).
        
        Parameters
        ----------
        config_dict : dict
            Dictionary with keys: predict, approximate, point_estimate_criterion
            
        Returns
        -------
        InferentialChoiceConfig
        """
        if config_dict is None:
            return cls()
        if isinstance(config_dict, cls):
            return config_dict
        return cls(**config_dict)
    
    def __repr__(self):
        return (f"InferentialChoiceConfig(predict='{self.predict}', "
                f"approximate='{self.approximate}', "
                f"point_estimate_criterion='{self.point_estimate_criterion}')")


class ModelBase(ABC):
    """Base API for predictive density models.

        Optional stochastic extensions:
            - predict_density: can return either [N,G] or [S,N,G]
            - inferential_choice: configuration for dual inferential_choice contexts (predict vs. approximate)
    
    Parameters
    ----------
    inferential_choice : dict or InferentialChoiceConfig, optional
        inferential_choice strategy configuration. Converted to InferentialChoiceConfig.
        Only used by stochastic models.
    """
    
    def __init__(self, inferential_choice=None):
        self._inferential_choice_config = InferentialChoiceConfig.from_dict(inferential_choice)
    
    def get_inferential_choice_config(self):
        """Get the inferential_choice configuration for this model."""
        return self._inferential_choice_config

    def _validate_context(self, context: str):
        if context not in {'predict', 'approximate'}:
            raise ValueError(f"context must be one of {{'predict', 'approximate'}}, got {context}")

    def resolve_inferential_choice(self, context='predict'):
        """Resolve inferential choice for a context to a canonical implemented mode."""
        self._validate_context(context)
        cfg = self.get_inferential_choice_config()
        raw = cfg.predict if context == 'predict' else cfg.approximate
        canonical = InferentialChoiceConfig.canonicalize_strategy(raw)

        if raw in InferentialChoiceConfig.STRATEGY_ALIASES:
            warnings.warn(
                f"inferential_choice='{raw}' is deprecated; using '{canonical}'.",
                UserWarning,
                stacklevel=2,
            )

        if canonical == 'point_estimate':
            warnings.warn(
                "inferential_choice='point_estimate' is compatibility-only and maps to "
                "'posterior_predictive' in active code paths.",
                UserWarning,
                stacklevel=2,
            )
            canonical = 'posterior_predictive'

        if canonical not in InferentialChoiceConfig.CANONICAL_STRATEGIES:
            raise NotImplementedError(
                f"inferential_choice='{raw}' is not implemented. "
                "Implemented choices: {'posterior_predictive', 'bma'}."
            )

        return canonical
    
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None): ...

    @abstractmethod
    def predict_density(self, X, y_grid, context='predict'): ...

    def get_second_order_distribution(self, X, y_grid, context='predict'):
        """Return a second-order distribution representation for uncertainty estimation.

        Stochastic models should override this to provide parameter-induced distributional
        variability. Deterministic models should keep the default NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a second-order distribution provider."
        )

    def predict_mode(self, X, y_grid, context='predict'):
        dens = self.predict_density(X, y_grid, context=context)
        if dens.ndim == 3:
            dens = np.mean(dens, axis=0)
        idx = dens.argmax(axis=1)
        return y_grid[idx]

    def default_y_grid(self, X, grid_points=512, y_pad=1.0):
        lo, hi = -1.0, 1.0
        try:
            lo, hi = float(self._y_min), float(self._y_max)
        except Exception:
            pass
        pad = y_pad * (hi - lo + 1e-6)
        return np.linspace(lo - pad, hi + pad, grid_points)

    def predict_density_samples(self, X, y_grid, context='predict', n_samples: int = 20):
        """Compatibility wrapper over ``predict_density``.

        The active API is ``predict_density`` returning either [N,G] or [S,N,G].
        This method is retained for compatibility and always returns [S,N,G].
        """
        warnings.warn(
            "predict_density_samples is deprecated. Use predict_density and handle [N,G] or [S,N,G].",
            UserWarning,
            stacklevel=2,
        )
        dens = self.predict_density(X, y_grid, context=context)
        if dens.ndim == 3:
            return dens
        if dens.ndim != 2:
            raise ValueError("predict_density must return [N,G] or [S,N,G].")
        return np.repeat(dens[None, ...], n_samples, axis=0)

    def predict_moments(self, X, y_grid, context='predict'):
        dens = self.predict_density(X, y_grid, context=context)
        if dens.ndim == 3:
            dens = np.mean(dens, axis=0)
        dens = dens / (integrate.trapezoid(dens, y_grid, axis=1)[:, None] + 1e-12)
        Ey  = integrate.trapezoid(dens * y_grid[None,:], y_grid, axis=1)
        Ey2 = integrate.trapezoid(dens * (y_grid[None,:]**2), y_grid, axis=1)
        Var = Ey2 - Ey**2
        return Ey, Var

    # --- Monte Carlo adapters for HDR/volume estimation ---
    def sample_output(self, X, n_samples, rng):
        """
        Optional: sample outputs y from the model conditional on inputs X.

        Signature: sampler(X, n_samples, rng) -> samples shaped (n_samples, N, d)
        Must be implemented by models that support Monte Carlo HDR estimation.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement sample_output(X, n_samples, rng).")

    def output_bounds(self, X, q_low=1e-3, q_high=1-1e-3, pad_frac=0.05, n_samples=10000, rng=None):
        """
        Optional: return per-input axis-aligned bounds covering output support.

        Expected shape: (N, d, 2) where bounds[i, j] = [low, high].
        Implementations may construct bounds by sampling and using quantiles.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement output_bounds(X, ...).")

    def density_function_for_input(self, X):
        """
        Optional: return a callable density function for the provided inputs.

        The returned callable should have signature `density_func(points, input_index)` or
        accept points and an optional input index. A simple form is `density_func(points, input_index=0)`
        returning an array of densities for `points` (shape (M,)).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement density_function_for_input(X).")
