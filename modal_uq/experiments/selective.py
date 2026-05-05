
import os
import numpy as np
import pandas as pd
from .base import ExperimentBase
from ..analysis.correlation import compute_uncertainty_scores, correlation_suite, save_correlation_artifacts
from ..metrics.selective import risk_coverage, aurc
from ..metrics import mode_errors
from ..utils import io as io_utils
from ..utils.seed import resolve_seed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ..datasets.synthetic_constant_var import SyntheticConstantVarDataset
import datetime

class SelectivePrediction(ExperimentBase):
    def run(self):
        print("Running Selective Prediction experiment...")
        print("[PHASE: Dataset Preparation] Starting...")
        # GT is available from experimental setup, so want to override this here.
        # In fact, this should probably be a feature of experiment base.
        if self.ds.needs_pseudo_ground_truth:
            # raise ValueError("Dataset indicates psuedo ground truth is needed, Feature not yet implemented. Use synthetic data with known GT.")
            # For experiment: Selective_Faithful only. Todo: General re-write.
            self.pgt.fit(self.ds.X_train, self.ds.y_train)
            self.model.fit(self.ds.X_train, self.ds.y_train, self.ds.X_val, self.ds.y_val)
            y_mode_true = np.array([self.pgt.conditional_mode(x) for x in self.ds.X_test])
            y_grid = self.model.default_y_grid(self.ds.X_test)
            self.y_grid = y_grid  # Cache y_grid to avoid recomputation
            y_mode_pred = self.model.predict_mode(self.ds.X_test, y_grid)
            self.X_test, self.y_test = self.ds.X_test, self.ds.y_test
        else:
            
            if type(self.ds) is SyntheticConstantVarDataset:
                # No mu, pi, sigma fns.
                X, y, _, _, _, y_grid = self.ds.get_data() # get n_samples
                self.X_train, self.y_train = X, y
                self.X_test, self.y_test, _, _, _, y_grid_test = self.ds.get_data() # get n_samples worth again, but this time keep only 0.2 for testing. Todo: Re-write to be more efficient and less hacky.
                split_seed = self.cfg.get('dataset', {}).get('params', {}).get('split_seed')
                split_seed = resolve_seed(split_seed)
                _, self.X_test, _, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.2, random_state=split_seed)
                y_grid = y_grid_test

            else:
                # For experiment: Selective_Synthetic only. Todo: General re-write.
                X, y, _, _ = self.ds.get_data(pi_fn=self.ds.test_pi_fn, mu_fn=self.ds.test_mu_fn, sigma_fn=self.ds.test_sigma_fn, noise_fn=self.ds.test_no_fn)
                split_seed = self.cfg.get('dataset', {}).get('params', {}).get('split_seed')
                split_seed = resolve_seed(split_seed)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed)
            
            self.y_train = np.expand_dims(self.y_train, axis=1) # Bodge to fix training for MDN. Todo: replace.
            print("[PHASE: Dataset Preparation] Complete")
            print("[PHASE: Model Training] Starting...")
            self.model.fit(self.X_train, self.y_train)
            print("[PHASE: Model Training] Complete [OK]")
            y_grid = self.model.default_y_grid(self.X_test)
            self.y_grid = y_grid  # Cache y_grid to avoid recomputation
            y_mode_pred = self.model.predict_mode(self.X_test, y_grid)

            true_dens = self.ds.gt_dens(self.X_test, y_grid)
            if type(self.ds) is SyntheticConstantVarDataset:
                y_mode_true = y_grid[true_dens.argmax(axis=1)]
            else:
                y_mode_true = self.ds.gt(self.X_test, self.ds.test_mu_fn, self.ds.test_pi_fn, self.ds.test_sigma_fn)


            
        self.results = {'y_mode_true': y_mode_true, 'y_mode_pred': y_mode_pred, 'y_grid': y_grid, 'true_dens': true_dens, 'est_dens': self.model.predict_density(self.X_test, y_grid, context='predict')}
        
        unc_cfg = self.cfg.get('uncertainty')
        print("Uncertainty config:", unc_cfg)
        if unc_cfg:
            print("[PHASE: Uncertainty Scoring] Starting... ({} measures)".format(len(unc_cfg['measures'])))
            # print(self.X_test.shape, self.y_test.shape)
            # print(self.X_test[:5], self.y_test[:5])
            # Pass canonical y_grid to uncertainty measures if available (for grid alignment)
            y_grid_arg = self.y_grid if hasattr(self, 'y_grid') and self.y_grid is not None else None
            df_scores = compute_uncertainty_scores(unc_cfg['measures'], self.model, self.X_test, self.y_test, y_grid=y_grid_arg)
            print("[PHASE: Uncertainty Scoring] Complete [OK]")
            corr_cfg = self.cfg.get('correlation', {})
            if corr_cfg.get('enabled', False):
                print("[PHASE: Correlation Analysis] Starting...")
                corrs, _ = correlation_suite(df_scores, corr_cfg.get('method', ['pearson','spearman']))
                out_dir = corr_cfg.get('output_dir', 'runs/_correlation')
                save_correlation_artifacts(df_scores, corrs, out_dir)
                print("[PHASE: Correlation Analysis] Complete [OK]")
            else:
                print("[PHASE: Correlation Analysis] Skipped (disabled)")
            # Run selective analysis and save AUROC-style (risk vs % abstention) plots
            print("[PHASE: Selective Analysis] Starting...")
            try:
                out_dir = self.cfg.get('experiment', {}).get('run_root')
                if out_dir is None:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                    print(ts)
                    out_dir = os.path.join('runs', '_selective', ts)
                print("[PHASE: Selective Analysis] Output directory: {}".format(out_dir))
                self.run_selective_analysis(df_scores, out_dir=out_dir)
                print("[PHASE: Selective Analysis] Complete [OK]")
            except Exception:
                # Do not fail experiment run if selective analysis errors; keep original behavior
                print(Exception)
                pass

    def _get_risk_fn(self, metric_name):
        name = (metric_name or '').lower()
        if name in ('moae', 'modal_absolute_error', 'mae'):
            return mode_errors.modal_absolute_error, {}
        if name in ('mose', 'modal_squared_error', 'mse'):
            return mode_errors.modal_squared_error, {}
        if name in ('lr_true', 'likelihood_ratio_true'):
            return mode_errors.likelihood_ratio_measure, {"reference_dist": "true"}
        if name in ('lr_est', 'likelihood_ratio_est'):
            return mode_errors.likelihood_ratio_measure, {"reference_dist": "est"}
        if name in ('coverage_true', 'coverage_reference_true'):
            return mode_errors.modal_coverage_measure, {"reference_dist": "true"}
        if name in ('coverage_est', 'coverage_reference_est'):
            return mode_errors.modal_coverage_measure, {"reference_dist": "est"}
        raise ValueError(f"Unknown selective metric '{metric_name}'")

    def _is_distance_metric(self, metric_name):
        """Return True for distance metrics (absolute / squared errors)."""
        if metric_name is None:
            return False
        name = str(metric_name).lower()
        return name in ('moae', 'modal_absolute_error', 'mae', 'mose', 'modal_squared_error', 'mse')

    def _requires_grid_aligned_truth(self, metric_name):
        """Return True for metrics that expect mode-aligned truth on the canonical y_grid."""
        if metric_name is None:
            return False
        name = str(metric_name).lower()
        return name in ('lr_true', 'likelihood_ratio_true', 'lr_est', 'likelihood_ratio_est', 'coverage_true', 'coverage_reference_true', 'coverage_est', 'coverage_reference_est')

    def compute_selective_curve(self, df_scores, measure_label, metric_name='moae', steps=100):
        print("Test: selective.py, compute_selective_curve() - test prints")
        if 'y_mode_pred' not in self.results:
            raise RuntimeError('y_mode_pred missing from self.results; run model first')
        if measure_label not in df_scores.columns:
            raise ValueError(f"Measure '{measure_label}' not found in df_scores columns")
        if self._requires_grid_aligned_truth(metric_name) and 'y_mode_true' in self.results:
            y_true = np.asarray(self.results['y_mode_true'])
        else:
            y_true = np.asarray(self.y_test)
        y_pred = np.asarray(self.results['y_mode_pred'])
        uncertainty = np.asarray(df_scores[measure_label].values)
        risk_fn, risk_fn_kwargs = self._get_risk_fn(metric_name)
        # print("debug-risk_fn", risk_fn)
        # print( "debug: risk_fn_kwargs", risk_fn_kwargs)
        if hasattr(self, 'y_grid') and self.y_grid is not None:
            y_grid = self.y_grid
        else:
            y_grid = self.model.default_y_grid(self.X_test)
        # print("Debug - print risk coverage inputs:")
        # print("y_true", y_true[:5], "y_pred", y_pred[:5], "uncertainty", uncertainty[:5], "y_grid", y_grid[:5], "risk_fn",risk_fn,"risk_fn_kwargs", risk_fn_kwargs, "coverage_steps", s)
        coverages, risks = risk_coverage(y_true, y_pred, uncertainty, risk_fn, risk_fn_kwargs, steps=steps, y_grid=y_grid, true_dens=self.results['true_dens'], est_dens=self.results['est_dens'])
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

        # order abstention, risk pairs by increasing abstention for plotting
        order = np.argsort(curve['abstention'])
        curve['abstention'] = curve['abstention'][order]
        curve['risk'] = curve['risk'][order]

        if filename is None:
            safe_measure = str(measure).replace(' ', '_')
            safe_metric = str(metric).replace(' ', '_')
            filename = f"selective_{safe_measure}_{safe_metric}.png"
        path = os.path.join(out_dir, filename)
        plt.figure(figsize=(6,4))
        plt.plot(curve['abstention'], curve['risk'], marker='o', linewidth=1)
        try:
            if not self._is_distance_metric(metric):
                plt.yscale('log')
        except Exception:
            pass
        plt.xlabel('% abstention')
        plt.ylabel(metric)
        plt.title(f"Selective curve: {measure} ({metric}), AURC={curve.get('aurc'):.4f}")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_measures_combined(self, curves_by_measure, metric, out_dir='runs/_selective', filename=None):
        """Plot multiple measures' selective curves for a single metric on one figure.

        curves_by_measure: dict mapping measure -> curve dict (with 'abstention', 'risk', optionally 'aurc')
        metric: name of the metric (used for title/filename)
        """
        os.makedirs(out_dir, exist_ok=True)
        safe_metric = str(metric).replace(' ', '_')
        if filename is None:
            filename = f"selective_measures_{safe_metric}.png"
        path = os.path.join(out_dir, filename)

        plt.figure(figsize=(8,6))
        plotted_any = False
        for measure, curve in curves_by_measure.items():
            if not isinstance(curve, dict) or 'abstention' not in curve or 'risk' not in curve:
                continue
            try:
                order = np.argsort(curve['abstention'])
                abst = np.asarray(curve['abstention'])[order]
                risk = np.asarray(curve['risk'])[order]
            except Exception:
                continue
            auc = curve.get('aurc')
            label = f"{measure}"
            if auc is not None:
                try:
                    label = f"{measure} (AURC={auc:.4f})"
                except Exception:
                    pass
            plt.plot(abst, risk, marker='o', linewidth=1, label=label)
            plotted_any = True

        if not plotted_any:
            plt.close()
            return None

        try:
            if not self._is_distance_metric(metric):
                plt.yscale('log')
        except Exception:
            pass

        plt.xlabel('% abstention')
        plt.ylabel(metric)
        plt.title(f"Selective curves (all measures) - {metric}")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def _save_combined_curve_csv(self, curves_dict, metric, metric_dir):
        rows = []
        for measure, curve in curves_dict.items():
            if not isinstance(curve, dict) or 'abstention' not in curve or 'risk' not in curve:
                continue
            order = np.argsort(curve['abstention'])
            abstention = np.asarray(curve['abstention'])[order]
            risk = np.asarray(curve['risk'])[order]
            aurc = curve.get('aurc')
            for idx, (abst, risk_value) in enumerate(zip(abstention, risk)):
                rows.append({
                    'metric': metric,
                    'measure': measure,
                    'point_index': int(idx),
                    'abstention': float(abst),
                    'risk': float(risk_value),
                    'aurc': float(aurc) if aurc is not None else np.nan,
                })

        if not rows:
            return None

        csv_path = os.path.join(metric_dir, 'combined_curve_data.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def run_selective_analysis(self, df_scores, metrics=None, measures=None, steps=100, out_dir=None):
        # Default metrics from config primary list
        metrics = metrics or self.cfg.get('metrics', {}).get('primary', [])
        if out_dir is None:
            out_dir = self.cfg.get('experiment', {}).get('run_root', 'runs/_selective')
        os.makedirs(out_dir, exist_ok=True)
        results = {}
        # Default measures: all columns in df_scores
        measures = measures or list(df_scores.columns)
        # collect curves per metric so we can plot all measures together per metric
        curves_by_metric = {metric: {} for metric in metrics}
        for measure in measures:
            results.setdefault(measure, {})
            for metric in metrics:
                try:
                    # create a subdirectory for this metric inside out_dir
                    safe_metric = str(metric).replace(' ', '_')
                    metric_dir = os.path.join(out_dir, safe_metric)
                    os.makedirs(metric_dir, exist_ok=True)

                    curve = self.compute_selective_curve(df_scores, measure, metric, steps=steps)
                    png = self.plot_selective_curve(curve, out_dir=metric_dir)
                    results[measure][metric] = {'aurc': curve['aurc'], 'png': png}
                    curves_by_metric[metric][measure] = curve
                except Exception as e:
                    results[measure][metric] = {'error': str(e)}
                    curves_by_metric[metric][measure] = {'error': str(e)}

        # For each metric, plot all measures on the same graph inside that metric's folder
        results.setdefault('_combined_by_metric', {})
        for metric, curves_dict in curves_by_metric.items():
            try:
                safe_metric = str(metric).replace(' ', '_')
                metric_dir = os.path.join(out_dir, safe_metric)
                os.makedirs(metric_dir, exist_ok=True)
                combined_png = self.plot_measures_combined(curves_dict, metric, out_dir=metric_dir)
                combined_csv = self._save_combined_curve_csv(curves_dict, metric, metric_dir)
                if combined_png is not None:
                    results['_combined_by_metric'][metric] = {'png': combined_png, 'csv': combined_csv}
            except Exception as e:
                results['_combined_by_metric'][metric] = {'error': str(e)}
        # Save summary JSON
        summary_path = os.path.join(out_dir, 'selective_summary.json')
        io_utils.write_json(results, summary_path)
        return results
