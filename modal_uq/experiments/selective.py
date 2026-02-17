
import numpy as np
from .base import ExperimentBase
from ..analysis.correlation import compute_uncertainty_scores, correlation_suite, save_correlation_artifacts

class SelectivePrediction(ExperimentBase):
    def run(self):
        self.pgt.fit(self.ds.X_train, self.ds.y_train)
        self.model.fit(self.ds.X_train, self.ds.y_train, self.ds.X_val, self.ds.y_val)
        y_mode_true = np.array([self.pgt.conditional_mode(x) for x in self.ds.X_test])
        y_grid = self.model.default_y_grid(self.ds.X_test)
        y_mode_pred = self.model.predict_mode(self.ds.X_test, y_grid)
        unc_cfg = self.cfg.get('uncertainty')
        if unc_cfg:
            df_scores = compute_uncertainty_scores(unc_cfg['measures'], self.model, self.ds.X_test, self.ds.y_test)
            corr_cfg = self.cfg.get('correlation', {})
            if corr_cfg.get('enabled', False):
                corrs, _ = correlation_suite(df_scores, corr_cfg.get('method', ['pearson','spearman']))
                out_dir = corr_cfg.get('output_dir', 'runs/_correlation')
                save_correlation_artifacts(df_scores, corrs, out_dir)
        self.results = {'y_mode_true': y_mode_true, 'y_mode_pred': y_mode_pred}
