
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .base import ExperimentBase
from ..analysis.correlation import compute_uncertainty_scores
from ..metrics.common import nll_from_density_at_truth
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.seed import resolve_seed
from ..analysis import plotting
from ..datasets.synthetic_constant_var import SyntheticConstantVarDataset
from ..datasets.mpe import MpeDataset

class ActiveLearning(ExperimentBase):
    def __init__(self, ds, pgt, model, metrics, cfg, n_jobs=None):
        """Initialize active_learning experiment with canonical y_grid caching.
        
        Caches the dataset-provided y_grid if available (for synthetic datasets with canonical grid).
        """
        super().__init__(ds, pgt, model, metrics, cfg, n_jobs)
        self.y_grid = getattr(ds, 'y_grid', None)

    def _al_config(self):
        return self.cfg.get('experiment', {}).get('al', {})

    def _supported_dataset_kind(self):
        if type(self.ds) is SyntheticConstantVarDataset:
            return 'synthetic_constant_var'
        if type(self.ds) is MpeDataset:
            return 'mpe'
        raise NotImplementedError(
            "Active learning currently supports only SyntheticConstantVarDataset and MpeDataset."
        )

    def _prepare_dataset(self):
        kind = self._supported_dataset_kind()
        if kind == 'synthetic_constant_var':
            X_pool = np.asarray(self.ds.X_train).copy()
            y_pool = np.asarray(self.ds.y_train).copy()
            X_eval = np.asarray(self.ds.X_test).copy()
            y_eval = np.asarray(self.ds.y_test).copy()
            y_grid = np.asarray(self.ds.y_grid)
            fit_y = np.expand_dims(y_pool, axis=1)
            eval_bundle = {'X': X_eval, 'y': y_eval, 'kind': kind}
            return X_pool, y_pool, fit_y, y_grid, eval_bundle

        n_traj = int(self.ds.y_train.shape[1])
        X_pool = np.repeat(np.asarray(self.ds.X_train), n_traj, axis=0)
        y_pool = np.asarray(self.ds.y_train).reshape(-1)
        X_eval = np.asarray(self.ds.X_test).copy()
        y_eval = np.asarray(self.ds.y_test).copy()
        y_grid = np.asarray(self.ds.y_grid)
        fit_y = np.expand_dims(y_pool, axis=1)
        eval_bundle = {'X': X_eval, 'y': y_eval, 'kind': kind, 'n_traj': n_traj}
        return X_pool, y_pool, fit_y, y_grid, eval_bundle

    def _budget_schedule(self, n_pool, init_size, rounds):
        if rounds <= 0:
            raise ValueError('rounds must be positive')
        if init_size <= 0:
            raise ValueError('init_size must be positive')
        if init_size >= n_pool:
            raise ValueError('init_size must be smaller than the pool size')

        remaining = n_pool - init_size
        base = remaining // rounds
        remainder = remaining % rounds
        return [base + 1 if idx < remainder else base for idx in range(rounds)]

    def _eval_nll(self, X_eval, y_eval, y_grid, dataset_kind, n_traj=None):
        dens = self.model.predict_density(X_eval, y_grid, context='predict')
        if dataset_kind == 'mpe':
            if n_traj is None:
                raise ValueError('n_traj is required for MPE NLL evaluation')
            y_eval_flat = np.asarray(y_eval).reshape(-1)
            X_eval_flat = np.repeat(np.asarray(X_eval), n_traj, axis=0)
            dens = self.model.predict_density(X_eval_flat, y_grid, context='predict')
            nll = nll_from_density_at_truth(dens, y_grid, y_eval_flat)
            return float(np.mean(nll))

        nll = nll_from_density_at_truth(dens, y_grid, np.asarray(y_eval))
        return float(np.mean(nll))

    def _measure_label(self, spec):
        params = dict(spec.get('params', {}))
        return params.get('label', spec.get('name'))

    def _nll_run_root(self):
        run_root = self.cfg.get('experiment', {}).get('run_root')
        if run_root:
            return os.path.join(run_root, 'nll')
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        return os.path.join('runs', '_active', ts, 'nll')

    def _score_measure(self, spec, X):
        y_grid_arg = self.y_grid if self.y_grid is not None else None
        df_scores = compute_uncertainty_scores([spec], self.model, X, y=None, y_grid=y_grid_arg)
        measure_label = self._measure_label(spec)
        if measure_label not in df_scores.columns:
            raise RuntimeError(f"Uncertainty measure '{measure_label}' did not produce a score column")
        return np.asarray(df_scores[measure_label].values), measure_label

    def _fit_model(self, X, y):
        try:
            self.model.fit(X, y, getattr(self.ds, 'X_val', None), getattr(self.ds, 'y_val', None))
        except TypeError:
            self.model.fit(X, y)

    def _run_single_measure(self, spec, X_pool, fit_y, X_eval, y_eval, y_grid, budget_schedule, dataset_kind, labeled_indices, unlabeled_indices, n_traj=None):
        measure_label = self._measure_label(spec)

        history_rows = []
        curves = []

        if len(labeled_indices) == 0:
            raise ValueError('init_size must be positive')

        self._fit_model(X_pool[labeled_indices], fit_y[labeled_indices])
        curves.append({'point_index': 0, 'labelled_budget': len(labeled_indices), 'nll': self._eval_nll(X_eval, y_eval, y_grid, dataset_kind, n_traj=n_traj)})

        for round_idx, batch_size in enumerate(budget_schedule, start=1):
            if batch_size <= 0:
                curves.append({'point_index': round_idx, 'labelled_budget': len(labeled_indices), 'nll': self._eval_nll(X_eval, y_eval, y_grid, dataset_kind, n_traj=n_traj)})
                continue

            if len(unlabeled_indices) < batch_size:
                batch_size = len(unlabeled_indices)
            if batch_size == 0:
                curves.append({'point_index': round_idx, 'labelled_budget': len(labeled_indices), 'nll': self._eval_nll(X_eval, y_eval, y_grid, dataset_kind, n_traj=n_traj)})
                continue

            scores, _ = self._score_measure(spec, X_pool[unlabeled_indices])
            order = np.argsort(-scores)
            selected_local = order[:batch_size]
            selected_indices = [unlabeled_indices[idx] for idx in selected_local]

            labeled_indices.extend(selected_indices)
            selected_set = set(selected_indices)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected_set]

            self._fit_model(X_pool[labeled_indices], fit_y[labeled_indices])
            curves.append({'point_index': round_idx, 'labelled_budget': len(labeled_indices), 'nll': self._eval_nll(X_eval, y_eval, y_grid, dataset_kind, n_traj=n_traj)})
            history_rows.append({'round': round_idx - 1, 'selected': selected_indices, 'batch_size': batch_size, 'labelled_budget': len(labeled_indices)})

        return {'measure': measure_label, 'curve': curves, 'history': history_rows}

    def _save_nll_outputs(self, run_root, per_measure_results):
        os.makedirs(run_root, exist_ok=True)
        plot_rows = []
        summary = {'metric': 'nll', 'measures': {}}
        curves_by_measure = {}

        for result in per_measure_results:
            measure = result['measure']
            curve = result['curve']
            curves_by_measure[measure] = curve

            frame = pd.DataFrame(curve).set_index('labelled_budget')
            plot_path = os.path.join(run_root, f'learning_curve_{measure}.png')
            plotting.plot_al_learning_curve(
                frame[['nll']],
                plot_path,
                title=f'Learning curve: {measure} (nll)',
                x_label='Labelled budget',
                y_label='Negative log likelihood'
            )

            summary['measures'][measure] = {'plot': plot_path, 'final_nll': float(frame['nll'].iloc[-1]), 'points': curve}

            for row in curve:
                plot_rows.append({
                    'metric': 'nll',
                    'measure': measure,
                    'point_index': int(row['point_index']),
                    'labelled_budget': int(row['labelled_budget']),
                    'nll': float(row['nll']),
                })

        combined_path = os.path.join(run_root, 'learning_curve_measures.png')
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted_any = False
        for measure, curve in curves_by_measure.items():
            x_values = [row['labelled_budget'] for row in curve]
            y_values = [row['nll'] for row in curve]
            ax.plot(x_values, y_values, marker='o', linewidth=1, label=measure)
            plotted_any = True
        if plotted_any:
            ax.set_xlabel('Labelled budget')
            ax.set_ylabel('Negative log likelihood')
            ax.set_title('Learning curves (all measures) - nll')
            ax.legend(loc='best')
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()
            fig.savefig(combined_path, dpi=150)
        plt.close(fig)

        csv_path = os.path.join(run_root, 'combined_curve_data.csv')
        pd.DataFrame(plot_rows).to_csv(csv_path, index=False)
        summary['combined_plot'] = combined_path
        summary['combined_csv'] = csv_path
        write_json(summary, os.path.join(run_root, 'nll_summary.json'))
        return summary
    
    def run(self):
        logger = get_logger(__name__)
        al_cfg = self._al_config()
        init_size = int(al_cfg.get('init_size', 20))
        rounds = int(al_cfg.get('rounds', 10))

        seed = resolve_seed(self.cfg.get('experiment', {}).get('seed'))
        rng = np.random.default_rng(seed)

        X_pool, y_pool, fit_y, y_grid, eval_bundle = self._prepare_dataset()
        n_pool = len(X_pool)
        budget_schedule = self._budget_schedule(n_pool, init_size, rounds)
        pool_order = rng.permutation(n_pool)
        labeled_indices = pool_order[:init_size].tolist()
        unlabeled_indices = pool_order[init_size:].tolist()

        unc_cfg = self.cfg.get('uncertainty', {})
        measure_specs = list(unc_cfg.get('measures', []))
        if not measure_specs:
            raise ValueError('Active learning requires at least one uncertainty measure')

        dataset_kind = eval_bundle['kind']
        X_eval = eval_bundle['X']
        y_eval = eval_bundle['y']
        n_traj = eval_bundle.get('n_traj')

        per_measure_results = []
        for spec in measure_specs:
            measure_result = self._run_single_measure(
                spec,
                X_pool,
                fit_y,
                X_eval,
                y_eval,
                y_grid,
                budget_schedule,
                dataset_kind,
                labeled_indices.copy(),
                unlabeled_indices.copy(),
                n_traj=n_traj,
            )
            per_measure_results.append(measure_result)

        run_root = self._nll_run_root()
        summary = self._save_nll_outputs(run_root, per_measure_results)
        self.results = summary
        logger.info('Active learning finished; measures=%d, output=%s', len(per_measure_results), run_root)
