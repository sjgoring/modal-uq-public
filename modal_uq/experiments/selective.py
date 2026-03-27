
import os
import numpy as np
from .base import ExperimentBase
from ..analysis.correlation import compute_uncertainty_scores, correlation_suite, save_correlation_artifacts
from ..metrics.selective import risk_coverage, aurc
from ..metrics import mode_errors
from ..utils import io as io_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ..datasets.synthetic_constant_var import SyntheticConstantVarDataset
import datetime

class SelectivePrediction(ExperimentBase):
    def run(self):
        print("Running Selective Prediction experiment...")
        # GT is available from experimental setup, so want to override this here.
        # In fact, this should probably be a feature of experiment base.
        if self.ds.needs_pseudo_ground_truth:
            # raise ValueError("Dataset indicates psuedo ground truth is needed, Feature not yet implemented. Use synthetic data with known GT.")
            # For experiment: Selective_Faithful only. Todo: General re-write.
            self.pgt.fit(self.ds.X_train, self.ds.y_train)
            self.model.fit(self.ds.X_train, self.ds.y_train, self.ds.X_val, self.ds.y_val)
            y_mode_true = np.array([self.pgt.conditional_mode(x) for x in self.ds.X_test])
            y_grid = self.model.default_y_grid(self.ds.X_test)
            y_mode_pred = self.model.predict_mode(self.ds.X_test, y_grid)
            self.X_test, self.y_test = self.ds.X_test, self.ds.y_test
        else:
            
            if type(self.ds) is SyntheticConstantVarDataset:
                # No mu, pi, sigma fns.
                X, y, _, _, _ = self.ds.get_data()
                self.X_train, self.y_train = X, y # bodge for initial results.
                self.X_test, self.y_test = X, y

            else:
                # For experiment: Selective_Synthetic only. Todo: General re-write.
                X, y, _, _ = self.ds.get_data(pi_fn=self.ds.test_pi_fn, mu_fn=self.ds.test_mu_fn, sigma_fn=self.ds.test_sigma_fn, noise_fn=self.ds.test_no_fn)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
            
            self.y_train = np.expand_dims(self.y_train, axis=1) # Bodge to fix training for MDN. Todo: replace.
            self.model.fit(self.X_train, self.y_train)
            y_grid = self.model.default_y_grid(self.X_test)
            y_mode_pred = self.model.predict_mode(self.X_test, y_grid)

            if type(self.ds) is SyntheticConstantVarDataset:
                y_mode_true = self.ds.gt(self.X_test)
            else:
                y_mode_true = self.ds.gt(self.X_test, self.ds.test_mu_fn, self.ds.test_pi_fn, self.ds.test_sigma_fn)


            
        self.results = {'y_mode_true': y_mode_true, 'y_mode_pred': y_mode_pred}
        
        unc_cfg = self.cfg.get('uncertainty')
        print("Uncertainty config:", unc_cfg)
        if unc_cfg:
            # print("Computing uncertainty scores for selective analysis...")
            # print(self.X_test.shape, self.y_test.shape)
            # print(self.X_test[:5], self.y_test[:5])
            df_scores = compute_uncertainty_scores(unc_cfg['measures'], self.model, self.X_test, self.y_test)
            corr_cfg = self.cfg.get('correlation', {})
            if corr_cfg.get('enabled', False):
                corrs, _ = correlation_suite(df_scores, corr_cfg.get('method', ['pearson','spearman']))
                out_dir = corr_cfg.get('output_dir', 'runs/_correlation')
                save_correlation_artifacts(df_scores, corrs, out_dir)
            # Run selective analysis and save AUROC-style (risk vs % abstention) plots
            print("Running selective analysis...")
            try:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                print(ts)
                out_dir = os.path.join('runs', f"_selective\{ts}")
                print(out_dir)
                self.run_selective_analysis(df_scores, out_dir=out_dir)
            except Exception:
                # Do not fail experiment run if selective analysis errors; keep original behavior
                print(Exception)
                pass

    def _get_risk_fn(self, metric_name):
        name = (metric_name or '').lower()
        if name in ('moae', 'modal_absolute_error', 'mae'):
            return mode_errors.modal_absolute_error
        if name in ('mose', 'modal_squared_error', 'mse'):
            return mode_errors.modal_squared_error
        raise ValueError(f"Unknown selective metric '{metric_name}'")

    def compute_selective_curve(self, df_scores, measure_label, metric_name='moae', steps=100):
        print("Test: selective.py, compute_selective_curve() - test prints")
        if 'y_mode_pred' not in self.results:
            raise RuntimeError('y_mode_pred missing from self.results; run model first')
        if measure_label not in df_scores.columns:
            raise ValueError(f"Measure '{measure_label}' not found in df_scores columns")
        y_true = np.asarray(self.y_test)
        y_pred = np.asarray(self.results['y_mode_pred'])
        uncertainty = np.asarray(df_scores[measure_label].values)
        risk_fn = self._get_risk_fn(metric_name)
        coverages, risks = risk_coverage(y_true, y_pred, uncertainty, risk_fn, steps=steps)
        abstention = 100.0 * (1.0 - coverages)
        area = aurc(coverages, risks)
        return {
            'measure': measure_label,
            'metric': metric_name,
            'abstention': abstention,
            'risk': risks,
            'coverages': coverages,
            'aurc': area
        }

    def plot_selective_curve(self, curve, out_dir='runs/_selective', filename=None):
        os.makedirs(out_dir, exist_ok=True)
        measure = curve.get('measure')
        metric = curve.get('metric')
        if filename is None:
            safe_measure = str(measure).replace(' ', '_')
            safe_metric = str(metric).replace(' ', '_')
            filename = f"selective_{safe_measure}_{safe_metric}.png"
        path = os.path.join(out_dir, filename)
        plt.figure(figsize=(6,4))
        plt.plot(curve['abstention'], curve['risk'], marker='o', linewidth=1)
        plt.gca().invert_xaxis()
        plt.xlabel('% abstention')
        plt.ylabel(metric)
        plt.title(f"Selective curve: {measure} ({metric}), AURC={curve.get('aurc'):.4f}")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def run_selective_analysis(self, df_scores, metrics=None, measures=None, steps=100, out_dir=None):
        # Default metrics from config primary list
        metrics = metrics or self.cfg.get('metrics', {}).get('primary', [])
        if out_dir is None:
            out_dir = 'runs/_selective'
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        # Default measures: all columns in df_scores
        measures = measures or list(df_scores.columns)
        for measure in measures:
            results.setdefault(measure, {})
            for metric in metrics:
                try:
                    curve = self.compute_selective_curve(df_scores, measure, metric, steps=steps)
                    png = self.plot_selective_curve(curve, out_dir=out_dir)
                    results[measure][metric] = {'aurc': curve['aurc'], 'png': png}
                except Exception as e:
                    results[measure][metric] = {'error': str(e)}
        # Save summary JSON
        summary_path = os.path.join(out_dir, 'selective_summary.json')
        io_utils.write_json(results, summary_path)
        return results
