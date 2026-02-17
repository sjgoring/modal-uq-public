
from sklearn.metrics import roc_auc_score, average_precision_score

def auroc(y_scores_id, y_scores_ood):
    import numpy as np
    y = np.concatenate([np.zeros_like(y_scores_id), np.ones_like(y_scores_ood)])
    s = np.concatenate([y_scores_id, y_scores_ood])
    return float(roc_auc_score(y, s))

def auprc(y_scores_id, y_scores_ood):
    import numpy as np
    y = np.concatenate([np.zeros_like(y_scores_id), np.ones_like(y_scores_ood)])
    s = np.concatenate([y_scores_id, y_scores_ood])
    return float(average_precision_score(y, s))
