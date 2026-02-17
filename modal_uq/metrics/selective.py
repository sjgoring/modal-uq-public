
import numpy as np

def risk_coverage(y_true, y_pred, uncertainty, risk_fn, steps=20):
    order = np.argsort(-uncertainty)
    risks, coverages = [], []
    n = len(y_true)
    for k in range(steps+1):
        keep = order[k*n//(steps+1):]
        if len(keep) == 0:
            risks.append(np.nan); coverages.append(0.0)
            continue
        risks.append(float(risk_fn(y_true[keep], y_pred[keep])))
        coverages.append(len(keep)/n)
    return np.array(coverages), np.array(risks)

def aurc(coverages, risks):
    return float(np.trapz(risks, coverages))
