
from .base import ExperimentBase
import os
from ..analysis.correlation import compute_uncertainty_scores, correlation_suite, save_correlation_artifacts
from ..analysis import ood_gen, ood_metrics, plotting
from ..utils.logging import get_logger
from ..utils.io import write_json


class OODExperiment(ExperimentBase):
    def run(self):
        logger = get_logger(__name__)

        # Fit model on training data
        self.model.fit(self.ds.X_train, self.ds.y_train, getattr(self.ds, 'X_val', None), getattr(self.ds, 'y_val', None))

        unc_cfg = self.cfg.get('uncertainty')
        if not unc_cfg:
            logger.info("No uncertainty configuration found; skipping OOD analysis.")
            self.results = {}
            return

        # Determine ID inputs (prefer validation, fall back to test)
        X_id = getattr(self.ds, 'X_val', None)
        y_id = getattr(self.ds, 'y_val', None)
        if X_id is None:
            X_id = getattr(self.ds, 'X_test', None)
            y_id = getattr(self.ds, 'y_test', None)
        if X_id is None:
            logger.error("No validation or test set available for OOD analysis.")
            self.results = {}
            return

        # OOD generation parameters from config
        ood_cfg = self.cfg.get('experiment', {}).get('ood', {})
        method = ood_cfg.get('method', 'shift')
        params = ood_cfg.get('params', {})
        n_ood = int(ood_cfg.get('n_ood_samples', max(100, len(X_id))))
        seed = ood_cfg.get('seed', self.cfg.get('experiment', {}).get('seed', None))

        # Generate OOD inputs
        try:
            if method == 'shift':
                X_ood = ood_gen.shift_features(X_id, n_ood, rng=seed, **params)
            elif method == 'noise':
                X_ood = ood_gen.add_noise(X_id, n_ood, rng=seed, **params)
            elif method == 'extrapolate':
                X_ood = ood_gen.extrapolate(X_id, n_ood, rng=seed, **params)
            elif method == 'mix':
                X_ood = ood_gen.mix_region(X_id, n_ood, rng=seed, **params)
            else:
                logger.warning("Unknown OOD method '%s', falling back to shift.", method)
                X_ood = ood_gen.shift_features(X_id, n_ood, rng=seed, **params)
        except Exception as e:
            logger.error("Failed to generate OOD inputs: %s", e)
            self.results = {}
            return

        # Compute uncertainty scores for ID and OOD
        try:
            df_id = compute_uncertainty_scores(unc_cfg['measures'], self.model, X_id, y_id)
        except Exception as e:
            logger.error("Failed to compute ID uncertainty scores: %s", e)
            self.results = {}
            return

        try:
            df_ood = compute_uncertainty_scores(unc_cfg['measures'], self.model, X_ood, None)
        except Exception as e:
            logger.error("Failed to compute OOD uncertainty scores: %s", e)
            self.results = {}
            return

        # Prepare output directory
        run_root = self.cfg.get('experiment', {}).get('run_root')
        if run_root is None:
            logger.warning("No run_root set in config; writing outputs to 'runs/_ood' instead")
            run_root = os.path.join('runs', '_ood')
        os.makedirs(run_root, exist_ok=True)

        # Save score tables
        id_csv = os.path.join(run_root, 'uncertainty_scores_id.csv')
        ood_csv = os.path.join(run_root, 'uncertainty_scores_ood.csv')
        try:
            df_id.to_csv(id_csv, index=False)
            df_ood.to_csv(ood_csv, index=False)
        except Exception as e:
            logger.warning("Failed to save uncertainty score CSVs: %s", e)

        # Correlation artifacts
        corr_cfg = self.cfg.get('correlation', {})
        if corr_cfg.get('enabled', False):
            try:
                corrs_id, _ = correlation_suite(df_id, corr_cfg.get('method', ['pearson', 'spearman']))
                corrs_ood, _ = correlation_suite(df_ood, corr_cfg.get('method', ['pearson', 'spearman']))
                out_dir = os.path.join(run_root, 'correlation', 'ood')
                save_correlation_artifacts(df_id, corrs_id, os.path.join(out_dir, 'id'))
                save_correlation_artifacts(df_ood, corrs_ood, os.path.join(out_dir, 'ood'))
            except Exception as e:
                logger.warning("Failed to compute/save correlation artifacts: %s", e)

        # Compute AUROC / AUPR using sklearn via ood_metrics helper
        invert = ood_cfg.get('invert', {})
        try:
            metrics = ood_metrics.compute_ood_detection_metrics(df_id, df_ood, invert=invert)
            summary_path = os.path.join(run_root, 'ood_summary.json')
            try:
                write_json(metrics, summary_path)
            except Exception:
                logger.warning("Failed to write ood summary JSON to %s", summary_path)
        except Exception as e:
            logger.error("Failed to compute OOD detection metrics: %s", e)
            metrics = {}

        # plotting
        plotting_cfg = self.cfg.get('experiment', {}).get('plotting', {'enabled': True})
        if plotting_cfg.get('enabled', True) and run_root:
            plots_dir = os.path.join(run_root, plotting_cfg.get('dir', 'plots'), 'ood')
            os.makedirs(plots_dir, exist_ok=True)
            measures = list(df_id.columns)
            try:
                plotting.plot_ood_score_distributions(df_id, df_ood, measures, os.path.join(plots_dir, 'score_distributions.png'))
            except Exception as e:
                logger.warning("Failed to create OOD score distribution plot: %s", e)
            try:
                plotting.plot_ood_roc_pr(df_id, df_ood, measures, os.path.join(plots_dir, 'roc_pr_curves.png'))
            except Exception as e:
                logger.warning("Failed to create OOD ROC/PR plot: %s", e)
            try:
                plotting.plot_ood_summary_bars(metrics, os.path.join(plots_dir, 'summary_bars.png'))
            except Exception as e:
                logger.warning("Failed to create OOD summary bars plot: %s", e)

        self.results = {'df_id_path': id_csv, 'df_ood_path': ood_csv, 'metrics': metrics}
