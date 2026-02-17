
from .base import ExperimentBase
from ..analysis.correlation import compute_uncertainty_scores, correlation_suite, save_correlation_artifacts

class OODExperiment(ExperimentBase):
    def run(self):
        self.pgt.fit(self.ds.X_train, self.ds.y_train)
        self.model.fit(self.ds.X_train, self.ds.y_train, self.ds.X_val, self.ds.y_val)
        unc_cfg = self.cfg.get('uncertainty')
        if unc_cfg:
            df_id = compute_uncertainty_scores(unc_cfg['measures'], self.model, self.ds.X_val, self.ds.y_val)
            df_ood= compute_uncertainty_scores(unc_cfg['measures'], self.model, self.ds.X_test, self.ds.y_test)
            corr_cfg = self.cfg.get('correlation', {})
            if corr_cfg.get('enabled', False):
                corrs_id, _  = correlation_suite(df_id,  corr_cfg.get('method', ['pearson','spearman']))
                corrs_ood, _ = correlation_suite(df_ood, corr_cfg.get('method', ['pearson','spearman']))
                out_dir = corr_cfg.get('output_dir', 'runs/_correlation')
                save_correlation_artifacts(df_id,  corrs_id,  out_dir + '/id')
                save_correlation_artifacts(df_ood, corrs_ood, out_dir + '/ood')
        self.results = {}

    def report(self):
        pass
