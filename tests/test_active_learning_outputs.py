"""Tests for the modernized active-learning output layout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class _DummyALModel:
    def fit(self, X, y, X_val=None, y_val=None):
        self._fit_shape = (len(X), len(y))
        return self

    def predict_density(self, X, y_grid, context='predict'):
        y_grid = np.asarray(y_grid)
        dens = np.exp(-0.5 * (y_grid[None, :] ** 2))
        dens = dens / (np.trapz(dens, y_grid, axis=1)[:, None] + 1e-12)
        return np.repeat(dens, len(X), axis=0)


def test_active_learning_writes_nll_folder_layout(tmp_path, monkeypatch):
    from modal_uq.datasets.synthetic_constant_var import SyntheticConstantVarDataset
    from modal_uq.experiments.active_learning import ActiveLearning

    ds = SyntheticConstantVarDataset(n_samples=36, y_grid_size=64, seed=11, split_seed=11)
    model = _DummyALModel()
    cfg = {
        'experiment': {
            'seed': 13,
            'run_root': str(tmp_path),
            'al': {'init_size': 12, 'rounds': 3},
        },
        'uncertainty': {
            'measures': [
                {'name': 'variance', 'params': {'label': 'variance_total'}},
                {'name': 'differential_entropy', 'params': {'label': 'differential_entropy_total'}},
            ]
        },
    }

    def fake_compute_uncertainty_scores(measure_specs, model, X, y=None, y_grid=None):
        label = measure_specs[0].get('params', {}).get('label', measure_specs[0]['name'])
        values = np.linspace(len(X), 1, len(X))
        return pd.DataFrame({label: values})

    monkeypatch.setattr('modal_uq.experiments.active_learning.compute_uncertainty_scores', fake_compute_uncertainty_scores)

    al = ActiveLearning(ds, None, model, {}, cfg, n_jobs=1)
    al.run()

    nll_root = Path(tmp_path) / 'nll'
    assert (nll_root / 'learning_curve_variance_total.png').exists()
    assert (nll_root / 'learning_curve_differential_entropy_total.png').exists()
    assert (nll_root / 'learning_curve_measures.png').exists()
    assert (nll_root / 'combined_curve_data.csv').exists()
    assert (nll_root / 'nll_summary.json').exists()

    df = pd.read_csv(nll_root / 'combined_curve_data.csv')
    assert list(df.columns) == ['metric', 'measure', 'point_index', 'labelled_budget', 'nll']
    assert set(df['metric']) == {'nll'}
    assert set(df['measure']) == {'variance_total', 'differential_entropy_total'}
    assert len(df) == 2 * 4

    summary = json.loads((nll_root / 'nll_summary.json').read_text())
    assert summary['metric'] == 'nll'
    assert 'variance_total' in summary['measures']
    assert 'differential_entropy_total' in summary['measures']