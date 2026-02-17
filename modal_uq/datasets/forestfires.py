
import pandas as pd
from .base import DatasetSpec
from ..registry import register

@register('dataset','forestfires')
class ForestFires(DatasetSpec):
    def __init__(self, path, target, features, split, standardize=True):
        df = pd.read_csv(path)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes
        X = df[features].values.astype(float)
        y = df[target].values.astype(float)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self._apply_split(X, y, split)
        if standardize:
            self._standardize_inplace()
