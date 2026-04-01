
from abc import ABC, abstractmethod
import numpy as np
import scipy.integrate as integrate


class MarginalizationConfig:
    """Configuration for marginalization strategies in stochastic models.
    
    Parameters
    ----------
    predict : str
        Marginalization strategy for prediction: 'bma_expected', 'posterior_weighted', 
        or 'point_estimate'. Includes epistemic uncertainty.
    approximate : str
        Marginalization strategy for approximating true DGP: 'point_estimate', 
        'bma_expected', or 'posterior_weighted'. Usually point_estimate.
    point_estimate_criterion : str
        Criterion for selecting parameters when strategy is 'point_estimate':
        'mle' (maximum likelihood, default), 'map', 'mean', or 'median'.
    """
    
    VALID_STRATEGIES = {'point_estimate', 'bma_expected', 'posterior_weighted'}
    VALID_CRITERIA = {'mle', 'map', 'mean', 'median'}
    
    def __init__(self, predict: str = 'bma_expected', approximate: str = 'point_estimate', 
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
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary (e.g., from JSON config).
        
        Parameters
        ----------
        config_dict : dict
            Dictionary with keys: predict, approximate, point_estimate_criterion
            
        Returns
        -------
        MarginalizationConfig
        """
        if config_dict is None:
            return cls()
        if isinstance(config_dict, cls):
            return config_dict
        return cls(**config_dict)
    
    def __repr__(self):
        return (f"MarginalizationConfig(predict='{self.predict}', "
                f"approximate='{self.approximate}', "
                f"point_estimate_criterion='{self.point_estimate_criterion}')")


class ModelBase(ABC):
    """Base API for predictive density models.

    Optional stochastic extensions:
      - predict_density_samples: return multiple parameter-sampled densities [S,N,G]
      - predict_mixture_params: return deterministic mixture params (for MDN)
      - marginalization: configuration for dual marginalization contexts (predict vs. approximate)
    
    Parameters
    ----------
    marginalization : dict or MarginalizationConfig, optional
        Marginalization strategy configuration. Converted to MarginalizationConfig.
        Only used by stochastic models.
    """
    
    def __init__(self, marginalization=None):
        self._marginalization_config = MarginalizationConfig.from_dict(marginalization)
    
    def get_marginalization_config(self):
        """Get the marginalization configuration for this model."""
        return self._marginalization_config
    
    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None): ...

    @abstractmethod
    def predict_density(self, X, y_grid, context='predict'): ...

    def predict_mode(self, X, y_grid, context='predict'):
        dens = self.predict_density(X, y_grid, context=context)
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
        """Return parameter-sampled densities.
        
        Parameters
        ----------
        X : array
            Input features
        y_grid : array
            Output grid
        context : {'predict', 'approximate'}, default='predict'
            Marginalization context:
            - 'predict': Include epistemic uncertainty (marginalize over parameters)
            - 'approximate': Best guess of true DGP (point estimate of parameters)
        n_samples : int
            Number of parameter samples to draw
            
        Returns
        -------
        dens : array of shape [S, N, G]
            Density samples where S=n_samples, N=number of inputs, G=grid size
        """
        dens = self.predict_density(X, y_grid, context=context)
        return np.repeat(dens[None, ...], n_samples, axis=0)

    def predict_moments(self, X, y_grid, context='predict'):
        dens = self.predict_density(X, y_grid, context=context)
        dens = dens / (integrate.trapezoid(dens, y_grid, axis=1)[:, None] + 1e-12)
        Ey  = integrate.trapezoid(dens * y_grid[None,:], y_grid, axis=1)
        Ey2 = integrate.trapezoid(dens * (y_grid[None,:]**2), y_grid, axis=1)
        Var = Ey2 - Ey**2
        return Ey, Var
