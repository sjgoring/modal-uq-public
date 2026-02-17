
import numpy as np
from .base import ExperimentBase
from ..registry import build
from ..analysis.correlation import compute_uncertainty_scores, correlation_suite, save_correlation_artifacts

class ActiveLearning(ExperimentBase):
    def run(self):
        al_cfg = self.cfg.get('experiment', {}).get('al', {"init_size": 20, "batch": 5, "rounds": 10, "acquisition": "variance"})
        init_size = al_cfg.get('init_size', 20)
        batch = al_cfg.get('batch', 5)
        rounds = al_cfg.get('rounds', 10)
        acq = build('acquisition', al_cfg.get('acquisition','variance'))

        X_pool, y_pool = self.ds.X_train.copy(), self.ds.y_train.copy()
        idx = np.random.default_rng(42).permutation(len(X_pool))
        L_idx = idx[:init_size].tolist()
        U_idx = idx[init_size:].tolist()

        history = []
        for r in range(rounds):
            X_L, y_L = X_pool[L_idx], y_pool[L_idx]
            self.model.fit(X_L, y_L)
            scores = acq.score(self.model, X_pool[U_idx])
            top = np.argsort(-scores)[:batch]
            new_idx = [U_idx[i] for i in top]
            L_idx.extend(new_idx)
            U_idx = [u for i,u in enumerate(U_idx) if i not in top]
            history.append({'round': r, 'L_size': len(L_idx)})

        unc_cfg = self.cfg.get('uncertainty')
        if unc_cfg and len(U_idx) > 0:
            df_scores = compute_uncertainty_scores(unc_cfg['measures'], self.model, X_pool[U_idx], y_pool[U_idx])
            corr_cfg = self.cfg.get('correlation', {})
            if corr_cfg.get('enabled', False):
                corrs, _ = correlation_suite(df_scores, corr_cfg.get('method', ['pearson','spearman']))
                out_dir = corr_cfg.get('output_dir', 'runs/_correlation')
                save_correlation_artifacts(df_scores, corrs, out_dir + '/al_pool')
        self.results = {'history': history}
