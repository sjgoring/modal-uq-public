
import numpy as np
import pandas as pd
from ..registry import build
from ..utils import plot as plot_utils


def compute_uncertainty_scores(measure_specs, model, X, y=None):
    import pandas as pd
    from ..registry import build

    out = {}
    for spec in measure_specs:
        # 1) Copy params so we can pop without mutating the original
        params = dict(spec.get('params', {}))

        # 2) Pop the optional label so it doesn't get passed to the constructor
        label = params.pop('label', None)

        # 3) Build the scorer with the cleaned params
        u = build('uncertainty', spec['name'], **params)

        # 4) Compute the score
        # print("Test prints - correlation.py - compute_uncertainty_scores")
        # print(X.shape, y.shape if y is not None else None)
        # print(X[:5], y[:5] if y is not None else None)
        # 23:10 02/03 - appears ok.
        s = u.score(model, X, y)

        # 5) Name the column: label (if provided) else measure name
        key = label if label else spec['name']
        out[key] = s

    return pd.DataFrame(out)


def correlation_suite(df_scores, methods=("pearson","spearman","kendall","distance"), bootstrap=None):
    corrs = {}
    for m in methods:
        if m == 'distance':
            corrs[m] = _distance_corr_matrix(df_scores.values, list(df_scores.columns))
        else:
            corrs[m] = df_scores.corr(method=m)
    return corrs, None


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
    import os, datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = os.path.join(out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    
    df_scores.to_csv(os.path.join(out_dir, 'uncertainty_scores.csv'), index=False)
    for name, mat in corrs.items():
        mat.to_csv(os.path.join(out_dir, f'corr_{name}.csv'))
        plot_utils.heatmap(mat, title=f'Correlation ({name})', save_path=os.path.join(out_dir, f'corr_{name}.png'))
