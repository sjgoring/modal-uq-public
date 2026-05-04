
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from ..utils.seed import resolve_seed

@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

class DatasetSpec(ABC):
    X_train: ...; y_train: ...; X_val: ...; y_val: ...; X_test: ...; y_test: ...

    @abstractmethod
    def __init__(self, *args, **kwargs): ...

    def _standardize_inplace(self):
        from sklearn.preprocessing import StandardScaler
        self.x_scaler = StandardScaler().fit(self.X_train)
        self.X_train = self.x_scaler.transform(self.X_train)
        self.X_val   = self.x_scaler.transform(self.X_val)
        self.X_test  = self.x_scaler.transform(self.X_test)
        self.y_scaler = None

    def _apply_split(self, X, y, split_cfg):
        strat = (split_cfg or {}).get('strategy', 'random')
        if strat == 'random':
            split_params = (split_cfg or {}).get('params', {})
            train = split_params.get('train', 0.6)
            val = split_params.get('val', 0.2)
            split_seed = split_params.get('split_seed', split_params.get('seed', split_params.get('random_state')))
            split_seed = resolve_seed(split_seed)
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=1-train, random_state=split_seed)
            X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val/(train), random_state=split_seed)
            return X_tr, y_tr, X_va, y_va, X_te, y_te
        elif strat == 'by_threshold_band':
            low = split_cfg['params']['low']; high = split_cfg['params']['high']
            mask = (X.squeeze() >= low) & (X.squeeze() <= high)
            X_id, y_id = X[mask], y[mask]
            X_ood, y_ood = X[~mask], y[~mask]
            split_params = (split_cfg or {}).get('params', {})
            split_seed = split_params.get('split_seed', split_params.get('seed', split_params.get('random_state')))
            split_seed = resolve_seed(split_seed)
            from sklearn.model_selection import train_test_split
            X_tr, X_va, y_tr, y_va = train_test_split(X_id, y_id, test_size=0.2, random_state=split_seed)
            return X_tr, y_tr, X_va, y_va, X_ood, y_ood
        else:
            raise NotImplementedError(f"Unknown split strategy: {strat}")
