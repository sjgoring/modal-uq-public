
import numpy as np
import pandas as pd
from ..registry import build
from ..utils import plot as plot_utils


def compute_uncertainty_scores(measure_specs, model, X, y=None):
    records = {}
    for spec in measure_specs:
        u = build('uncertainty', spec['name'], **spec.get('params', {}))
        needs_y = spec['name'] in {'nll'}
        s = u.score(model, X, y if needs_y else None)
        records[spec['name']] = s
    return pd.DataFrame(records)


def correlation_suite(df_scores, methods=("pearson","spearman","kendall","distance"), bootstrap=None):
    corrs = {}
    for m in methods:
        if m == 'distance':
            corrs[m] = _distance_corr_matrix(df_scores.values, list(df_scores.columns))
        else:
            corrs[m] = df_scores.corr(method=m)
    cis = None
    return corrs, cis


def _distance_corr_matrix(X, cols):
    import itertools
    M = X.shape[1]
    out = np.ones((M,M))
    for i,j in itertools.combinations(range(M), 2):
        out[i,j] = out[j,i] = _distance_corr(X[:,i], X[:,j])
    import pandas as pd
    return pd.DataFrame(out, columns=cols, index=cols)


def _distance_corr(x, y):
    from scipy.spatial.distance import pdist, squareform
    A = squareform(pdist(x[:,None], metric='euclidean'))
    B = squareform(pdist(y[:,None], metric='euclidean'))
    A -= A.mean(0)[None,:] + A.mean(1)[:,None] - A.mean()
    B -= B.mean(0)[None,:] + B.mean(1)[:,None] - B.mean()
    dcov = (A*B).mean()
    dvarx = (A*A).mean()**0.5
    dvary = (B*B).mean()**0.5
    return 0.0 if dvarx==0 or dvary==0 else dcov/(dvarx*dvary)


def save_correlation_artifacts(df_scores, corrs, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    df_scores.to_csv(os.path.join(out_dir, 'uncertainty_scores.csv'), index=False)
    for name, mat in corrs.items():
        mat.to_csv(os.path.join(out_dir, f'corr_{name}.csv'))
        plot_utils.heatmap(mat, title=f'Correlation ({name})', save_path=os.path.join(out_dir, f'corr_{name}.png'))
