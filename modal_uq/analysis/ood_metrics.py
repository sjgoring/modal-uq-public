import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)

def compute_ood_detection_metrics(df_id, df_ood, invert=None):
    """Compute AUROC and AUPR for each measure comparing ID vs OOD scores.

    Parameters
    - df_id, df_ood: pandas.DataFrame with same columns (measures)
    - invert: dict mapping column -> True if lower score implies OOD (will negate scores)

    Returns
    - dict mapping measure -> {'auroc': float or None, 'aupr': float or None, 'n_id': int, 'n_ood': int}
    """
    invert = invert or {}
    out = {}
    cols = list(df_id.columns)
    for col in cols:
        try:
            s_id = np.asarray(df_id[col].values).ravel()
            s_ood = np.asarray(df_ood[col].values).ravel()
        except Exception as e:
            logger.warning("Skipping column %s: could not extract values: %s", col, e)
            continue

        scores = np.concatenate([s_id, s_ood])
        if invert.get(col, False):
            scores = -scores
        labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])

        # Handle trivial cases
        if len(np.unique(labels)) < 2:
            out[col] = {'auroc': None, 'aupr': None, 'n_id': int(len(s_id)), 'n_ood': int(len(s_ood))}
            continue

        try:
            auroc = float(roc_auc_score(labels, scores))
        except Exception as e:
            logger.warning("AUROC computation failed for %s: %s", col, e)
            auroc = None
        try:
            aupr = float(average_precision_score(labels, scores))
        except Exception as e:
            logger.warning("AUPR computation failed for %s: %s", col, e)
            aupr = None

        out[col] = {'auroc': auroc, 'aupr': aupr, 'n_id': int(len(s_id)), 'n_ood': int(len(s_ood))}

    return out
