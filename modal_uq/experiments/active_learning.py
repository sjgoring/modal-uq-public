
import os
import numpy as np
from .base import ExperimentBase
from ..registry import build
from ..analysis.correlation import compute_uncertainty_scores, correlation_suite, save_correlation_artifacts
from ..utils.io import write_json
from ..utils.logging import get_logger
from ..utils.seed import resolve_seed
from ..analysis import plotting

class ActiveLearning(ExperimentBase):
    def __init__(self, ds, pgt, model, metrics, cfg, n_jobs=None):
        """Initialize active_learning experiment with canonical y_grid caching.
        
        Caches the dataset-provided y_grid if available (for synthetic datasets with canonical grid).
        """
        super().__init__(ds, pgt, model, metrics, cfg, n_jobs)
        # Cache canonical y_grid from dataset if available (for uncertainty scoring consistency)
        self.y_grid = getattr(ds, 'y_grid', None)
    
    def run(self):
        logger = get_logger(__name__)
        al_cfg = self.cfg.get('experiment', {}).get('al', {"init_size": 20, "batch": 5, "rounds": 10, "acquisition": "variance"})
        init_size = al_cfg.get('init_size', 20)
        batch = al_cfg.get('batch', 5)
        rounds = al_cfg.get('rounds', 10)
        acq_name = al_cfg.get('acquisition','variance')
        acq_params = al_cfg.get('acquisition_params', {})
        acq = build('acquisition', acq_name, **acq_params)

        # reproducible RNG from experiment seed
        seed = resolve_seed(self.cfg.get('experiment', {}).get('seed'))
        rng = np.random.default_rng(seed)

        if not hasattr(self.ds, 'X_train') or not hasattr(self.ds, 'y_train'):
            self.ds._setup_test_train_split()  # ensure dataset has train/test split for AL loop

        X_pool, y_pool = self.ds.X_train.copy(), self.ds.y_train.copy()
        n_pool = len(X_pool)
        idx = rng.permutation(n_pool)
        L_idx = idx[:init_size].tolist()
        U_idx = idx[init_size:].tolist()

        # tracking structures
        acq_scores_by_round = []
        selection_matrix = np.zeros((n_pool, rounds), dtype=bool)
        metric_rows = []
        df_by_round = []

        # initial fit on labelled set
        history = []
        if len(L_idx) > 0:
            X_L_init, y_L_init = X_pool[L_idx], y_pool[L_idx]
            self.model.fit(X_L_init, y_L_init)

        # evaluation set for metrics
        X_eval = getattr(self.ds, 'X_test', None)
        y_eval = getattr(self.ds, 'y_test', None)

        for r in range(rounds):
            if len(U_idx) == 0:
                logger.info("Unlabelled pool empty at round %s, stopping.", r)
                break

            # cap batch size to remaining pool
            curr_batch = min(batch, len(U_idx))

            # compute acquisition scores for unlabeled pool
            scores = acq.score(self.model, X_pool[U_idx])
            scores = np.asarray(scores).ravel()
            if scores.ndim != 1 or scores.shape[0] != len(U_idx):
                raise RuntimeError(f"Acquisition `score` must return 1D array of length {len(U_idx)}; got shape {scores.shape}")

            acq_scores_by_round.append(scores.copy())

            top = np.argsort(-scores)[:curr_batch]
            new_idx = [U_idx[i] for i in top]
            selected_scores = [float(scores[i]) for i in top]

            # mark selections in matrix
            for i in new_idx:
                selection_matrix[i, r] = True

            # update labelled / unlabelled sets
            L_idx.extend(new_idx)
            U_idx = [u for i,u in enumerate(U_idx) if i not in top]

            # fit model with newly extended labelled set
            X_L, y_L = X_pool[L_idx], y_pool[L_idx]
            self.model.fit(X_L, y_L)

            # compute evaluation metric and uncertainty scores on held-out eval set if available
            if X_eval is not None and y_eval is not None:
                try:
                    # try predict_moments to get mean prediction
                    y_grid = self.model.default_y_grid(X_eval)
                    Ey, Var = self.model.predict_moments(X_eval, y_grid)
                    preds = Ey
                except Exception:
                    pred_fn = getattr(self.model, 'predict', None)
                    if pred_fn is not None:
                        preds = pred_fn(X_eval)
                    else:
                        preds = None

                if preds is not None:
                    rmse = float(np.sqrt(np.mean((preds - y_eval)**2)))
                    metric_rows.append({'round': r, 'rmse': rmse, 'L_size': len(L_idx)})
                else:
                    metric_rows.append({'round': r, 'L_size': len(L_idx)})

                # compute uncertainty scores
                unc_cfg = self.cfg.get('uncertainty')
                if unc_cfg:
                    try:
                        y_grid_arg = self.y_grid if hasattr(self, 'y_grid') and self.y_grid is not None else None
                        df_scores = compute_uncertainty_scores(unc_cfg['measures'], self.model, X_eval, y_eval, y_grid=y_grid_arg)
                        errors = np.abs(preds - y_eval) if preds is not None else np.zeros(len(X_eval))
                        df_by_round.append((df_scores, errors))
                    except Exception as e:
                        logger.warning("Failed to compute per-round uncertainty scores at round %s: %s", r, e)

            history.append({'round': r, 'L_size': len(L_idx), 'selected': new_idx, 'selected_scores': selected_scores})
            logger.info("Round %d: selected %d samples, labelled size now %d", r, len(new_idx), len(L_idx))

        unc_cfg = self.cfg.get('uncertainty')
        # persist results
        self.results = {'history': history}
        run_root = self.cfg.get('experiment', {}).get('run_root')
        if run_root:
            write_json(self.results, os.path.join(run_root, 'al_history.json'))
            # save acquisition scores and selection matrix
            try:
                import numpy as _np
                _np.save(os.path.join(run_root, 'acq_scores_by_round.npy'), acq_scores_by_round, allow_pickle=True)
                _np.save(os.path.join(run_root, 'selection_matrix.npy'), selection_matrix)
            except Exception:
                logger.warning("Failed to persist AL arrays to run_root")

        if unc_cfg and len(U_idx) > 0:
            y_grid_arg = self.y_grid if hasattr(self, 'y_grid') and self.y_grid is not None else None
            df_scores = compute_uncertainty_scores(unc_cfg['measures'], self.model, X_pool[U_idx], y_pool[U_idx], y_grid=y_grid_arg)
            corr_cfg = self.cfg.get('correlation', {})
            if corr_cfg.get('enabled', False):
                corrs, _ = correlation_suite(df_scores, corr_cfg.get('method', ['pearson','spearman']))
                if run_root:
                    out_dir = os.path.join(run_root, 'correlation', 'al_pool')
                else:
                    out_dir = corr_cfg.get('output_dir', 'runs/_correlation') + '/al_pool'
                save_correlation_artifacts(df_scores, corrs, out_dir)

        # plotting
        plotting_cfg = self.cfg.get('experiment', {}).get('plotting', {'enabled': True})
        if plotting_cfg.get('enabled', True) and run_root:
            plots_dir = os.path.join(run_root, plotting_cfg.get('dir', 'plots'), 'al')
            os.makedirs(plots_dir, exist_ok=True)
            # learning curve
            try:
                import pandas as _pd
                if len(metric_rows) > 0:
                    metric_df = _pd.DataFrame(metric_rows).set_index('round')
                    plotting.plot_al_learning_curve(metric_df[['rmse']] if 'rmse' in metric_df.columns else metric_df,
                                                    os.path.join(plots_dir, 'learning_curve.png'))
            except Exception as e:
                logger.warning("Failed to create AL learning curve plot: %s", e)

            # acquisition distributions
            try:
                plotting.plot_al_acquisition_distributions(acq_scores_by_round, os.path.join(plots_dir, 'acquisition_dists.png'))
            except Exception as e:
                logger.warning("Failed to create AL acquisition distribution plot: %s", e)

            # selection timeline
            try:
                plotting.plot_al_selection_timeline(selection_matrix, os.path.join(plots_dir, 'selection_timeline.png'))
            except Exception as e:
                logger.warning("Failed to create AL selection timeline plot: %s", e)

            # uncertainty vs error for first measure
            try:
                if len(df_by_round) > 0:
                    first_measure = unc_cfg['measures'][0].get('params', {}).get('label', unc_cfg['measures'][0]['name'])
                    plotting.plot_al_uncertainty_vs_error(df_by_round, first_measure, os.path.join(plots_dir, f'uncertainty_vs_error_{first_measure}.png'))
            except Exception as e:
                logger.warning("Failed to create AL uncertainty vs error plot: %s", e)

        if unc_cfg and len(U_idx) > 0:
            df_scores = compute_uncertainty_scores(unc_cfg['measures'], self.model, X_pool[U_idx], y_pool[U_idx])
            corr_cfg = self.cfg.get('correlation', {})
            if corr_cfg.get('enabled', False):
                corrs, _ = correlation_suite(df_scores, corr_cfg.get('method', ['pearson','spearman']))
                if run_root:
                    out_dir = os.path.join(run_root, 'correlation', 'al_pool')
                else:
                    out_dir = corr_cfg.get('output_dir', 'runs/_correlation') + '/al_pool'
                save_correlation_artifacts(df_scores, corrs, out_dir)
        logger.info("Active learning finished; history length=%d", len(history))
